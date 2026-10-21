"""LiteLLM handler giving tool calling to a vLLM server that has it disabled.

A vLLM server started without ``--enable-auto-tool-choice --tool-call-parser``
rejects any request carrying ``tools`` with HTTP 400. A Qwen3.8 model still
emits Hermes-format ``<tool_call>`` blocks reliably when the
tools are described in the prompt (measured 12/12 across two placements), so
what is missing is only the server-side translation layer.

This handler is that layer, on the client side:

  request   OpenAI ``tools``/``tool_choice``  ->  Hermes ``<tools>`` system block
  response  ``<tool_call>{...}</tool_call>``  ->  OpenAI ``tool_calls``

Upstream is always called without ``tools`` and without streaming. Streaming
clients get the assembled reply replayed as chunks: tool-call syntax cannot be
parsed reliably from a partial token stream, and Prime Agent needs whole tool
calls, not fragments.

LiteLLM's own ``add_function_to_prompt`` does not cover this case -- it is
skipped for openai-compatible providers, and it never parses the reply back
into tool calls, so the caller sees prose where it expects a tool call.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import os
import random
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

import litellm
from litellm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    GenericStreamingChunk,
    ModelResponse,
)

# The closing tag is optional: a reply truncated at the output cap ends
# mid-JSON with no </tool_call> at all. Without this the block is invisible to
# the parser, so it cannot even be reported as unreadable and retried.
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|\Z)", re.DOTALL)
# S2: the model sometimes closes a Hermes block with tags from the XML tool
# dialect it also saw in training.
TRAILING_XML_RE = re.compile(r"(?:\s*</(?:invoke|parameter|function|tool_call)>)+\s*\Z")
# Qwen sometimes falls back to the XML tool syntax it also saw in training.
XML_CALL_RE = re.compile(r"<function=([A-Za-z0-9_.-]+)>(.*?)</function>", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=([A-Za-z0-9_.-]+)>(.*?)</parameter>", re.DOTALL)
XML_BARE_RE = re.compile(r"<function=([A-Za-z0-9_.-]+)(?=>|\s|\Z)")
LOGGER = logging.getLogger("LiteLLM Proxy.hermes_tools")
# Emit these content-free metrics under the launcher's default logging setup.
# The inherited LiteLLM handler still honors LITELLM_LOG=WARNING/ERROR.
LOGGER.setLevel(logging.INFO)

TOOL_PROMPT_HEAD = (
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>\n"
)
TOOL_PROMPT_TAIL = (
    "\n</tools>\n\nFor each function call, return a json object with function "
    "name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def _tool_prompt(tools: list[dict], tool_choice: Any) -> str:
    signatures = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else tool
        if function:
            signatures.append(json.dumps(function, ensure_ascii=False))
    prompt = TOOL_PROMPT_HEAD + "\n".join(signatures) + TOOL_PROMPT_TAIL
    # "required"/a named tool are instructions to the server we are replacing,
    # so they have to become instructions to the model instead.
    if tool_choice == "required":
        prompt += "\n\nYou MUST call at least one function in your reply."
    elif isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            prompt += f"\n\nYou MUST call the function `{name}` in your reply."
    elif tool_choice == "none":
        prompt = ""
    return prompt


def _render_messages(messages: list[dict], tool_prompt: str) -> list[dict]:
    """Fold tool results back into a shape a tool-less server understands."""
    rendered: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            # No tool role exists upstream; present the result as user content
            # tagged the way Qwen is trained to read it back.
            rendered.append({
                "role": "user",
                "content": f"<tool_response>\n{message.get('content')}\n</tool_response>",
            })
            continue
        if role == "assistant" and message.get("tool_calls"):
            blocks = [
                "<tool_call>\n" + json.dumps({
                    "name": (c.get("function") or {}).get("name"),
                    "arguments": _loads(( c.get("function") or {}).get("arguments")),
                }, ensure_ascii=False) + "\n</tool_call>"
                for c in message["tool_calls"]
            ]
            text = message.get("content") or ""
            rendered.append({"role": "assistant",
                             "content": (text + "\n" + "\n".join(blocks)).strip()})
            continue
        rendered.append(dict(message))

    if not tool_prompt:
        return rendered
    for message in rendered:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.rstrip() + "\n\n" + tool_prompt
            elif isinstance(content, list):
                # A new list, not append: the shallow copy still shares the
                # caller's list, and a retry would append the prompt twice.
                message["content"] = [*content,
                                      {"type": "text", "text": "\n\n" + tool_prompt}]
            return rendered
    return [{"role": "system", "content": tool_prompt}, *rendered]


def _loads(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    return raw if raw is not None else {}


def _strip_reasoning(content: str) -> tuple[str, bool]:
    """Return visible content and whether the tool/reasoning boundary is ambiguous.

    A closing tag inside an argument is data. With Qwen's implicit opening
    tag, a complete tool block before </think> could be either a reasoning
    example or a real call followed by a misplaced closer. Do not guess:
    preserve it for the retry classifier without making it executable.
    Explicit, fully closed reasoning can contain example calls; those remain
    reasoning rather than executable tools.
    """
    explicit = content.lstrip().startswith("<think>")
    if "</think>" not in content:
        return content, explicit
    tools = list(TOOL_CALL_RE.finditer(content))
    for closer in re.finditer("</think>", content):
        # A literal tag in any tool's arguments cannot delimit reasoning.
        if any(tool.start() <= closer.start() < tool.end() for tool in tools):
            continue
        if tools and tools[0].start() < closer.start() and not explicit:
            return content, True
        return content[closer.end():].lstrip(), False
    # No closing delimiter outside a tool block: an explicit reasoning prefix
    # is still open, so its examples must not become executable calls.
    return content, explicit


def _split_tool_calls(content: str, known: set[str] | None = None,
                      required: dict[str, set[str]] | None = None) -> tuple[str, list[dict]]:
    """Pull Hermes tool calls out of the reply and drop the thinking block."""
    if not content:
        return "", []
    content, ambiguous_reasoning = _strip_reasoning(content)
    if ambiguous_reasoning:
        return content.strip(), []
    calls = []
    unparsed = []
    for match in TOOL_CALL_RE.finditer(content):
        parsed = _parse_one_call(match.group(1), known, required)
        if parsed is None:
            unparsed.append(match.group(0))
            continue
        name, arguments = parsed
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "index": len(calls),
            "function": {
                "name": name,
                "arguments": arguments if isinstance(arguments, str)
                             else json.dumps(arguments, ensure_ascii=False),
            },
        })
    if calls or unparsed:
        # Blocks that could not be parsed stay in the visible content rather
        # than vanishing: a stalled agent that can see its own malformed call
        # is recoverable, one whose call silently disappeared is not.
        content = TOOL_CALL_RE.sub("", content)
        if unparsed:
            content = (content + "\n" + "\n".join(unparsed)).strip()
    return content.strip(), calls


# Observed constantly in real agent traffic: {"name": ipython, ...} -- the tool
# name emitted as a bare word. Quoting it recovers a fully parseable object,
# which is far more reliable than salvaging the arguments with regexes.
BARE_NAME_RE = re.compile(
    r'("(?:name|function)"\s*:\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*[,}])')


def _balance_brackets(raw: str) -> str:
    """Close an object the model mis-closed, string-aware.

    Two real failure modes, both fatal before this existed: the reply is cut
    off mid-object, and the closer is typed as ``)`` instead of ``}`` (seen as
    ``...stderr[:1000])\\n"})``). Brackets inside a JSON string are left alone,
    so a ``)`` in a shell command or a line of Python is never touched.
    """
    depth = 0
    in_string = False
    escaped = False
    out = []
    for char in raw:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            out.append(char)
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char in ")" and depth > 0:
            # Outside a string a ')' cannot be valid JSON, and at depth it is
            # always a mis-typed closing brace.
            char = "}"
            depth -= 1
        out.append(char)
    repaired = "".join(out)
    if depth > 0:
        repaired += "}" * depth
    return repaired


def _repair_json(raw: str) -> Any | None:
    """Best-effort decode: the model truncates, nests, unquotes and mis-closes."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw)
    # Drop foreign XML closers, and the degenerate </invoke> repetition the
    # model sometimes emits until it exhausts the output cap.
    stripped = TRAILING_XML_RE.sub("", raw)
    raw = stripped if stripped else raw
    variants = [raw]
    quoted = BARE_NAME_RE.sub(r'\1"\2"\3', raw)
    if quoted != raw:
        variants.append(quoted)
    for base in list(variants):
        balanced = _balance_brackets(base)
        if balanced != base:
            variants.append(balanced)
    for base in variants:
        for candidate in (base, *(base + "}" * n for n in range(1, 4))):
            try:
                return json.loads(candidate)
            except ValueError:
                continue
    return None


NAME_RE = re.compile(r'"(?:name|function)"\s*:\s*"?([A-Za-z0-9_.-]+)"?')
ARGS_RE = re.compile(r'"(?:arguments|parameters)"\s*:\s*\{(.*)', re.DOTALL)
# A key, then everything up to the next key or the end -- so an unescaped quote
# inside the value is kept rather than terminating it.
PAIR_RE = re.compile(
    r'"([A-Za-z0-9_.-]+)"\s*:\s*"(.*?)"\s*(?=,\s*"[A-Za-z0-9_.-]+"\s*:|\s*$)',
    re.DOTALL,
)


def _args_body(block: str) -> str | None:
    match = ARGS_RE.search(block)
    if not match:
        return None
    return match.group(1).rstrip().rstrip("}").rstrip()


def _lenient_call(block: str) -> tuple[str, dict] | None:
    name = NAME_RE.search(block)
    if not name:
        return None
    body = _args_body(block)
    arguments: dict[str, Any] = {}
    if body is not None:
        arguments = dict(PAIR_RE.findall(body))
        for key, raw in re.findall(
            r'"([A-Za-z0-9_.-]+)"\s*:\s*(-?\d+(?:\.\d+)?|true|false|null)', body
        ):
            arguments.setdefault(key, json.loads(raw))
        if body.strip() and not arguments:
            # The block carried arguments this salvage could not read. Emitting
            # the call anyway would run it with no arguments -- for a shell tool
            # that is a different action, not a degraded one. Refuse instead.
            return None
    return name.group(1), arguments


def _parse_one_call(block: str, known: set[str] | None = None,
                    required: dict[str, set[str]] | None = None) -> tuple[str, dict] | None:
    """Validate every recovery path against the tools offered on this turn."""
    parsed = _recover_one_call(block, known or set())
    if parsed is None:
        return None
    name, arguments = parsed
    if not name or (known is not None and name not in known):
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    # Do not reject {} unconditionally: tools with no required arguments are
    # valid. Full schema/type validation still belongs to the tool executor.
    if not (required or {}).get(name, set()).issubset(arguments):
        return None
    return name, arguments


def _recover_one_call(block: str, known: set[str]) -> tuple[str, Any] | None:
    """Recover (name, arguments) from the several shapes Qwen actually emits."""
    payload = _repair_json(block)
    if isinstance(payload, dict):
        # Nested {"function": {"name": ..., "arguments": {...}}}.
        inner = payload.get("function")
        if isinstance(inner, dict):
            payload = inner

        arguments = payload.get("arguments", payload.get("parameters", {}))
        # Canonical {"name": ..., "arguments": {...}}.
        name = payload.get("name")
        if isinstance(name, str):
            return name, arguments
        if isinstance(payload.get("function"), str):
            return payload["function"], arguments

        # The model renames the name key -- "assistant_name", "tool_name",
        # "function_name" have all been observed. Resolve by value rather than
        # by guessing key spellings: whichever string value names a tool that
        # was actually offered is the call.
        for key, value in payload.items():
            if key in {"arguments", "parameters"}:
                continue
            if isinstance(value, str) and value in known:
                return value, arguments
        # No declared tools to match against (or none matched): a payload whose
        # only other entry is a string is unambiguous enough to accept.
        others = [(k, v) for k, v in payload.items()
                  if k not in {"arguments", "parameters"}]
        if len(others) == 1 and isinstance(others[0][1], str):
            return others[0][1], arguments

    # Valid shape, invalid JSON: the model regularly forgets to escape quotes
    # inside an argument value ({"command": "echo "hi""}). The official vLLM
    # parser drops these too, but the intent is unambiguous, so recover it.
    lenient = _lenient_call(block)
    if lenient:
        return lenient

    # The XML tool syntax, whole or partial.
    xml = XML_CALL_RE.search(block)
    if xml:
        return xml.group(1), {k: v.strip() for k, v in XML_PARAM_RE.findall(xml.group(2))}
    bare = XML_BARE_RE.search(block)
    if bare:
        params = {k: v.strip() for k, v in XML_PARAM_RE.findall(block)}
        return bare.group(1), params
    return None


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
# Measured: the deployment serves ~32 requests in flight and 429s beyond that.
# Queueing here is cheaper than a retry storm from a 40-VM evaluation fleet, and
# far cheaper than a 429 reaching an agent as a lost step. Keep the proxy at one
# worker process or this gate is per-process and lets N times as much through.
MAX_INFLIGHT = int(os.environ.get("HERMES_MAX_INFLIGHT", "28"))
_SYNC_GATE = threading.Semaphore(MAX_INFLIGHT)
_ASYNC_GATE: asyncio.Semaphore | None = None
_ASYNC_GATE_LOCK = threading.Lock()


def _async_gate() -> asyncio.Semaphore:
    """Created lazily: a Semaphore binds to the running loop on first use."""
    global _ASYNC_GATE
    with _ASYNC_GATE_LOCK:
        if _ASYNC_GATE is None:
            _ASYNC_GATE = asyncio.Semaphore(MAX_INFLIGHT)
    return _ASYNC_GATE
MAX_ATTEMPTS = int(os.environ.get("HERMES_MAX_ATTEMPTS", "6"))


def _backoff(attempt: int, response: httpx.Response | None) -> float:
    """Honour Retry-After when the server sends one, else exponential + jitter."""
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(float(header), 60.0)
            except ValueError:
                pass
    return min(2 ** attempt, 30.0) * (0.5 + random.random())


def _should_retry(response: httpx.Response) -> bool:
    return response.status_code in RETRY_STATUS


def _raise_upstream_error(reply: httpx.Response, model: str) -> None:
    """Keep the upstream status and diagnostic text through LiteLLM mapping.

    HTTPStatusError's string omits the body and it has no status_code
    attribute, so LiteLLM maps it to a 500 APIConnectionError. In particular
    that hides context overflow from the caller's compact-and-retry logic.
    A native LiteLLM exception passes through its mapper unchanged.
    """
    if reply.is_success:
        return
    error = litellm.exceptions.APIError(
        status_code=reply.status_code,
        message=reply.text or reply.reason_phrase,
        llm_provider="hermes-tools", model=model, request=reply.request,
    )
    error.response = reply
    try:
        error.body = reply.json()
    except ValueError:
        error.body = reply.text
    raise error


def _log_upstream_attempt(request_id: str, attempt: int, queue_seconds: float,
                          started: float, reply: httpx.Response | None) -> dict[str, Any]:
    """Log durations and token counts, never prompts, outputs or credentials."""
    metrics: dict[str, Any] = {
        "request_id": request_id,
        "attempt": attempt + 1,
        "queue_ms": round(queue_seconds * 1000, 1) if attempt == 0 else 0,
        "upstream_ms": round((time.monotonic() - started) * 1000, 1),
        "status": reply.status_code if reply is not None else None,
    }
    raw_usage: dict[str, Any] = {}
    provider_response_id = None
    if reply is not None and reply.is_success:
        try:
            body = reply.json()
            usage = body.get("usage") if isinstance(body, dict) else None
            if isinstance(usage, dict):
                raw_usage = usage
                provider_response_id = body.get("id")
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        metrics[key] = value
        except ValueError:
            pass
    LOGGER.info("hermes_upstream_attempt %s", json.dumps(metrics, sort_keys=True))
    return {"request_id": request_id, "attempt": attempt + 1, "status": metrics["status"],
            "usage": raw_usage, "provider_response_id": provider_response_id}


class HermesToolsLLM(CustomLLM):
    """Speak OpenAI tool calling to the client, Hermes prompt syntax upstream."""

    def _upstream(self, model: str, api_base: str | None, api_key: str | None):
        base = (api_base or os.environ.get("HERMES_UPSTREAM_BASE_URL", "")).rstrip("/")
        key = api_key or os.environ.get("HERMES_UPSTREAM_API_KEY", "")
        if not base:
            raise litellm.exceptions.APIError(
                status_code=500, message="HERMES_UPSTREAM_BASE_URL is not set",
                llm_provider="hermes-tools", model=model)
        return f"{base}/chat/completions", {"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"}

    @staticmethod
    def _known_names(optional_params: dict) -> set[str]:
        if (optional_params or {}).get("tool_choice") == "none":
            return set()
        names = set()
        for tool in (optional_params or {}).get("tools") or []:
            function = tool.get("function") if tool.get("type") == "function" else tool
            name = (function or {}).get("name")
            if isinstance(name, str):
                names.add(name)
        return names

    @staticmethod
    def _required_arguments(optional_params: dict) -> dict[str, set[str]]:
        required = {}
        for tool in (optional_params or {}).get("tools") or []:
            function = tool.get("function") if tool.get("type") == "function" else tool
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            schema = function.get("parameters") or {}
            fields = schema.get("required", []) if isinstance(schema, dict) else []
            required[function["name"]] = {field for field in fields if isinstance(field, str)}
        return required

    def _payload(self, model: str, messages: list, optional_params: dict) -> dict:
        params = dict(optional_params or {})
        tools = params.pop("tools", None) or []
        tool_choice = params.pop("tool_choice", "auto")
        # Client-side concerns the upstream server neither needs nor accepts.
        for drop in ("stream", "stream_options", "parallel_tool_calls",
                     "functions", "function_call"):
            params.pop(drop, None)
        prompt = _tool_prompt(tools, tool_choice) if tools else ""
        return {"model": model.split("/", 1)[-1],
                "messages": _render_messages(messages, prompt), **params}

    @staticmethod
    def _has_unparsed_call(upstream: dict, known: set[str] | None = None,
                           required: dict[str, set[str]] | None = None) -> bool:
        """Did the model attempt a tool call this parser could not read?

        This is worth a retry rather than a shrug: Prime ends the whole episode
        the moment it sees an assistant message with no tool calls and
        stopReason "stop". One unreadable block therefore costs the entire
        task, not one turn.
        """
        # Without offered tools, literal tool syntax is ordinary assistant
        # text (e.g. an explanation), not a failed tool attempt to resample.
        if known is not None and not known:
            return False
        choice = (upstream.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        content, calls = _split_tool_calls(content, known, required)
        return not calls and ("<tool_call" in content or "<function=" in content)

    def _to_response(self, upstream: dict, model_response: ModelResponse,
                     model: str, known: set[str] | None = None,
                     required: dict[str, set[str]] | None = None,
                     attempts: list[dict[str, Any]] | None = None) -> ModelResponse:
        choice = (upstream.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content, calls = _split_tool_calls(message.get("content") or "", known, required)
        model_response.id = upstream.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
        model_response.created = upstream.get("created") or int(time.time())
        model_response.model = model
        block = model_response.choices[0]
        block.message.content = content or None
        block.finish_reason = "tool_calls" if calls else (
            choice.get("finish_reason") or "stop")
        if calls:
            # Typed objects, not the raw dicts the streaming path uses: handed a
            # dict, the proxy's response serialisation drops function.name and
            # the client receives a nameless tool call.
            block.message.tool_calls = [
                ChatCompletionMessageToolCall(
                    id=call["id"], type="function",
                    function=Function(name=call["function"]["name"],
                                      arguments=call["function"]["arguments"]),
                )
                for call in calls
            ]
        usage = upstream.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        # LiteLLM fills missing counts with zero and rebuilds streaming usage,
        # dropping extra top-level fields. The details object preserves extras,
        # so carry the original usage there for lossless accounting over SSE.
        details = dict(usage.get("prompt_tokens_details") or {})
        details["provider_usage"] = usage
        if attempts is not None:
            details["proxy_attempts"] = attempts
        model_response.usage = litellm.Usage(  # type: ignore[attr-defined]
            **{**usage, "prompt_tokens_details": details},
        )
        return model_response

    def _checked_response(self, reply: httpx.Response, model_response: ModelResponse,
                          model: str, known: set[str],
                          required: dict[str, set[str]],
                          attempts: list[dict[str, Any]]) -> ModelResponse:
        _raise_upstream_error(reply, model)
        upstream = reply.json()
        if self._has_unparsed_call(upstream, known, required):
            # Returning stop here would make the harness mark an unreadable
            # tool attempt as a successful final answer.
            raise litellm.exceptions.APIError(
                status_code=422, llm_provider="hermes-tools", model=model,
                message=f"No valid tool call after {MAX_ATTEMPTS} attempts: "
                        "malformed syntax, ambiguous reasoning boundary, unknown tool, "
                        "or missing required arguments.",
                request=reply.request,
            )
        return self._to_response(upstream, model_response, model, known, required, attempts)

    def completion(self, *args, **kwargs) -> ModelResponse:
        model, messages = kwargs["model"], kwargs["messages"]
        url, headers = self._upstream(model, kwargs.get("api_base"), kwargs.get("api_key"))
        params = kwargs.get("optional_params") or {}
        known = self._known_names(params)
        required = self._required_arguments(params)
        payload = self._payload(model, messages, params)
        # A shared deployment rate-limits hard at the concurrency an evaluation
        # fleet produces. A 429 that reaches the agent is a lost task step, so
        # absorb it here rather than letting it surface as a failed episode.
        request_id, queued_at = uuid.uuid4().hex, time.monotonic()
        attempts = []
        with _SYNC_GATE:
            queue_seconds = time.monotonic() - queued_at
            with httpx.Client(timeout=kwargs.get("timeout") or 1800) as client:
                for attempt in range(MAX_ATTEMPTS):
                    reply, started = None, time.monotonic()
                    try:
                        reply = client.post(url, headers=headers, json=payload)
                    finally:
                        attempts.append(_log_upstream_attempt(request_id, attempt, queue_seconds, started, reply))
                    if attempt == MAX_ATTEMPTS - 1:
                        break
                    if _should_retry(reply):
                        time.sleep(_backoff(attempt, reply))
                        continue
                    if reply.status_code == 200 and self._has_unparsed_call(reply.json(), known, required):
                        time.sleep(0.5)
                        continue
                    break
        return self._checked_response(reply, kwargs["model_response"], model, known, required, attempts)

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        model, messages = kwargs["model"], kwargs["messages"]
        url, headers = self._upstream(model, kwargs.get("api_base"), kwargs.get("api_key"))
        params = kwargs.get("optional_params") or {}
        known = self._known_names(params)
        required = self._required_arguments(params)
        payload = self._payload(model, messages, params)
        request_id, queued_at = uuid.uuid4().hex, time.monotonic()
        attempts = []
        async with _async_gate():
            queue_seconds = time.monotonic() - queued_at
            async with httpx.AsyncClient(timeout=kwargs.get("timeout") or 1800) as client:
                for attempt in range(MAX_ATTEMPTS):
                    reply, started = None, time.monotonic()
                    try:
                        reply = await client.post(url, headers=headers, json=payload)
                    finally:
                        attempts.append(_log_upstream_attempt(request_id, attempt, queue_seconds, started, reply))
                    if attempt == MAX_ATTEMPTS - 1:
                        break
                    if _should_retry(reply):
                        await asyncio.sleep(_backoff(attempt, reply))
                        continue
                    if reply.status_code == 200 and self._has_unparsed_call(reply.json(), known, required):
                        await asyncio.sleep(0.5)
                        continue
                    break
        return self._checked_response(reply, kwargs["model_response"], model, known, required, attempts)

    def _chunks(self, response: ModelResponse) -> list[GenericStreamingChunk]:
        block = response.choices[0]
        # ChatCompletionToolCallChunk is a TypedDict, so the typed calls on the
        # message have to come back as plain dicts -- and each needs the index
        # that the non-streaming shape does not carry.
        calls = [
            {"id": call.id, "type": "function", "index": index,
             "function": {"name": call.function.name,
                          "arguments": call.function.arguments}}
            for index, call in enumerate(getattr(block.message, "tool_calls", None) or [])
        ]
        usage = response.usage.model_dump(exclude_none=True)
        chunks: list[GenericStreamingChunk] = []
        text = block.message.content or ""
        if text:
            chunks.append({"text": text, "tool_use": None, "is_finished": False,
                           "finish_reason": "", "usage": None, "index": 0})
        for call in calls:
            chunks.append({"text": "", "tool_use": call, "is_finished": False,
                           "finish_reason": "", "usage": None, "index": 0})
        chunks.append({"text": "", "tool_use": None, "is_finished": True,
                       "finish_reason": block.finish_reason, "usage": usage, "index": 0})
        return chunks

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        return iter(self._chunks(self.completion(*args, **kwargs)))

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        for chunk in self._chunks(await self.acompletion(*args, **kwargs)):
            yield chunk


hermes_tools = HermesToolsLLM()
