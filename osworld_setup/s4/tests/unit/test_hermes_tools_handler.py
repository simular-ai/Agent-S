"""The tool-call parser fronting a vLLM server with tool calling disabled.

Every shape here was observed coming out of
a Qwen3.8-27B deployment during a 50-sample survey; the
canonical form covered 45 of them and the rest are the tail this parser
exists to absorb.
"""
import json
import asyncio
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "proxy"))

litellm = pytest.importorskip("litellm", reason="LiteLLM proxy venv only")
from hermes_tools_handler import (  # noqa: E402
    _render_messages,
    _split_tool_calls,
    _tool_prompt,
)
import hermes_tools_handler as handler  # noqa: E402
import httpx  # noqa: E402


def one_call(raw):
    _, calls = _split_tool_calls(raw)
    assert len(calls) == 1, calls
    function = calls[0]["function"]
    return function["name"], json.loads(function["arguments"])


def test_canonical_shape():
    assert one_call('<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>') == (
        "bash", {"command": "ls"})


def test_nested_function_shape_with_missing_brace():
    """Qwen both nests the call and truncates the closing brace."""
    assert one_call(
        '<tool_call>{"function": {"name": "bash", "arguments": {"command": "pwd"}}</tool_call>'
    ) == ("bash", {"command": "pwd"})


def test_unescaped_quotes_in_an_argument_are_recovered():
    """The model forgets to escape quotes inside a shell command. The intent is
    unambiguous, so the call is salvaged rather than dropped."""
    name, args = one_call(
        '<tool_call>{"name": "bash", "arguments": {"command": "echo "hi" && who"}}</tool_call>')
    assert name == "bash"
    assert args["command"] == 'echo "hi" && who'


def test_xml_tool_syntax():
    assert one_call(
        "<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
    ) == ("bash", {"command": "pwd"})


def test_non_string_arguments_keep_their_type():
    assert one_call('<tool_call>{"name":"wait","arguments":{"seconds":5,"loud":true}}</tool_call>'
                    ) == ("wait", {"seconds": 5, "loud": True})


def test_several_calls_are_indexed_in_order():
    _, calls = _split_tool_calls(
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{}}</tool_call>')
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert [c["index"] for c in calls] == [0, 1]


def test_thinking_block_is_stripped():
    content, calls = _split_tool_calls(
        'deliberating</think>\n<tool_call>{"name":"a","arguments":{}}</tool_call>')
    assert content == ""
    assert calls[0]["function"]["name"] == "a"


def test_unparseable_block_yields_no_call_and_stays_visible():
    """Never invent a call. An agent that can see its own malformed output can
    recover; one whose call vanished silently cannot."""
    content, calls = _split_tool_calls('<tool_call>{"function_bash"}</tool_call>')
    assert calls == []
    assert "function_bash" in content


def test_arguments_that_cannot_be_salvaged_do_not_become_an_empty_call():
    """Running a shell tool with no command is a different action, not a
    degraded one."""
    _, calls = _split_tool_calls(
        '<tool_call>{"name": "bash", "arguments": {!!! unreadable !!!}}</tool_call>')
    assert calls == []


def test_tool_choice_required_is_expressed_in_the_prompt():
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    assert "MUST call at least one function" in _tool_prompt(tools, "required")
    assert "MUST call the function `bash`" in _tool_prompt(
        tools, {"type": "function", "function": {"name": "bash"}})
    assert _tool_prompt(tools, "none") == ""


def test_tool_results_are_folded_into_a_shape_the_server_accepts():
    """There is no tool role upstream, so results come back as tagged user text
    and prior tool calls as the syntax the model emitted."""
    rendered = _render_messages([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command":"ls"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
    ], "TOOLS")
    assert rendered[0] == {"role": "system", "content": "TOOLS"}
    assert "<tool_call>" in rendered[2]["content"]
    assert rendered[3]["role"] == "user"
    assert "<tool_response>" in rendered[3]["content"]


def test_tool_prompt_appends_to_an_existing_system_message():
    rendered = _render_messages(
        [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "go"}],
        "TOOLS")
    assert len(rendered) == 2
    assert rendered[0]["content"] == "Be helpful.\n\nTOOLS"


def test_bare_word_tool_name_is_quoted_and_recovered():
    """Seen constantly in real agent traffic: the tool name emitted unquoted,
    which makes the whole object invalid JSON."""
    name, args = one_call(
        '<tool_call>{"name": ipython, "arguments": {"code": "print(1)"}}</tool_call>')
    assert name == "ipython"
    assert args == {"code": "print(1)"}


def test_bare_word_name_keeps_full_json_argument_fidelity():
    """Quoting the name lets the arguments go through a real JSON parse, so
    newlines and nested quotes survive exactly."""
    _, args = one_call(
        '<tool_call>{"name": ipython, "arguments": {"code": "a = \\"x\\"\\nprint(a)"}}</tool_call>')
    assert args["code"] == 'a = "x"\nprint(a)'


def test_rendering_does_not_mutate_the_callers_messages():
    """A retry re-renders the same message objects; appending in place would
    stack the tool prompt once per attempt."""
    messages = [{"role": "system", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": "go"}]
    _render_messages(messages, "TOOLS")
    _render_messages(messages, "TOOLS")
    assert messages[0]["content"] == [{"type": "text", "text": "hi"}]


def test_renamed_name_key_is_resolved_by_the_declared_tools():
    """The model renames the name key -- assistant_name, tool_name and
    function_name have all been seen. Resolve by which string value names a
    tool that was actually offered, instead of guessing key spellings."""
    known = {"ipython", "bash"}
    for block in (
        '<tool_call>{"assistant_name": "ipython", "arguments": {"code": "print(1)"}}</tool_call>',
        '<tool_call>{"tool_name": "ipython", "arguments": {"code": "print(1)"}}</tool_call>',
        '<tool_call>{"function_name": "ipython", "arguments": {"code": "print(1)"}}</tool_call>',
    ):
        _, calls = _split_tool_calls(block, known)
        assert calls[0]["function"]["name"] == "ipython", block
        assert json.loads(calls[0]["function"]["arguments"]) == {"code": "print(1)"}


def test_a_declared_tool_name_wins_over_an_argument_that_looks_like_one():
    known = {"bash"}
    _, calls = _split_tool_calls(
        '<tool_call>{"tool_name": "bash", "arguments": {"command": "ipython"}}</tool_call>',
        known)
    assert calls[0]["function"]["name"] == "bash"


def test_known_names_are_read_off_the_request_tools():
    from hermes_tools_handler import HermesToolsLLM
    assert HermesToolsLLM._known_names({"tools": [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "ipython"}},
    ]}) == {"bash", "ipython"}


def test_object_closed_with_a_paren_is_repaired():
    """Observed in a real episode: the model closes the arguments object with
    ')' instead of '}'. Before the repair this parsed to nothing, Prime saw a
    plain text reply, and the whole task ended after one turn."""
    name, args = one_call(
        '<tool_call>{"name": "ipython", "arguments": {"code": "print(1)"})</tool_call>')
    assert name == "ipython"
    assert args == {"code": "print(1)"}


def test_parens_inside_a_string_are_never_touched():
    """The repair is string-aware: shell and Python code is full of parens."""
    _, args = one_call(
        '<tool_call>{"name":"ipython","arguments":{"code":"f(g(x))  # )))"}}</tool_call>')
    assert args["code"] == "f(g(x))  # )))"


def test_truncated_object_is_completed():
    name, args = one_call(
        '<tool_call>{"name":"bash","arguments":{"command":"ls"}</tool_call>')
    assert (name, args) == ("bash", {"command": "ls"})


def test_unparsed_call_is_reported_so_it_can_be_retried():
    """Prime ends the episode on an assistant message with no tool calls, so an
    unreadable block has to be caught before it is returned."""
    from hermes_tools_handler import HermesToolsLLM
    unreadable = {"choices": [{"message": {"content": '<tool_call>{"!!!"}</tool_call>'}}]}
    readable = {"choices": [{"message": {
        "content": '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>'}}]}
    prose = {"choices": [{"message": {"content": "Here is the answer."}}]}
    assert HermesToolsLLM._has_unparsed_call(unreadable) is True
    assert HermesToolsLLM._has_unparsed_call(readable) is False
    assert HermesToolsLLM._has_unparsed_call(prose) is False


def test_foreign_xml_closers_are_stripped():
    """The model mixes the XML tool dialect into a Hermes block, closing it
    with </invoke></parameter></function>."""
    name, args = one_call(
        '<tool_call>\n{"name": "ipython", "arguments": {"code": "x=1"}\n}\n'
        '</invoke>\n\n</parameter>\n</function>')
    assert (name, args) == ("ipython", {"code": "x=1"})


def test_degenerate_closer_repetition_is_stripped():
    """Seen once: a valid call followed by </invoke> repeated until the output
    cap was exhausted."""
    name, args = one_call(
        '<tool_call>\n{"name":"ipython","arguments":{"code":"r.stdout"})'
        + "\n</invoke>" * 50)
    assert (name, args) == ("ipython", {"code": "r.stdout"})


def test_block_without_a_closing_tag_is_still_seen():
    """A reply truncated at the output cap has no </tool_call> at all. It
    cannot be repaired -- the JSON is cut mid-string -- but it MUST still be
    visible to the parser, or it cannot be reported and retried and the
    episode dies silently."""
    from hermes_tools_handler import HermesToolsLLM
    truncated = ('<tool_call>\n{"name": "ipython", "arguments": {"code": '
                 '"import json\\ns = json.load(open(\'/tmp/x\'))\\nfor r in s:\\n    print(r[\'en')
    _, calls = _split_tool_calls(truncated, {"ipython"})
    assert calls == []
    assert HermesToolsLLM._has_unparsed_call(
        {"choices": [{"message": {"content": truncated}}]}) is True


def test_two_blocks_where_the_first_lacks_a_closing_tag():
    """The optional closer must not swallow a following block."""
    _, calls = _split_tool_calls(
        '<tool_call>{"name":"a","arguments":{}}'
        '<tool_call>{"name":"b","arguments":{}}</tool_call>', {"a", "b"})
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


@pytest.mark.parametrize("block", [
    '{"name":"phantom","arguments":{}}',
    '{"function":{"name":"phantom","arguments":{}}}',
    '{"function":"phantom","arguments":{}}',
    '{"tool_name":"phantom","arguments":{}}',
    '{"name": phantom,"arguments":{"code":"echo "hi""}}',
    '<function=phantom><parameter=code>print(1)</parameter></function>',
    '<function=phantom><parameter=code>print(1)</parameter>',
    'function call',
])
def test_all_recovery_paths_reject_unoffered_tools(block):
    raw = f"<tool_call>{block}</tool_call>"
    content, calls = _split_tool_calls(raw, {"ipython"})
    assert calls == []
    assert block in content
    assert handler.HermesToolsLLM._has_unparsed_call(
        {"choices": [{"message": {"content": raw}}]}, {"ipython"})


@pytest.mark.parametrize("block", [
    '{"name":"ipython"}',
    '{"name":"ipython","arguments":{}}',
    '{"name":"ipython","arguments":{"other":"value"}}',
    '{"name":"ipython","arguments":"not JSON"}',
    '{"name":"ipython","arguments":[]}',
    '{"name":"ipython","arguments":null}',
    '<function=ipython></function>',
])
def test_missing_required_arguments_are_not_executable(block):
    raw = f"<tool_call>{block}</tool_call>"
    content, calls = _split_tool_calls(raw, {"ipython"}, {"ipython": {"code"}})
    assert calls == []
    assert block in content


@pytest.mark.parametrize("arguments", ["", ',"arguments":{}'])
def test_zero_argument_tool_is_still_valid(arguments):
    _, calls = _split_tool_calls(
        '<tool_call>{"name":"noop"' + arguments + "}</tool_call>",
        {"noop"}, {"noop": set()})
    assert calls[0]["function"] == {"name": "noop", "arguments": "{}"}


def test_json_encoded_arguments_keep_their_content():
    arguments = {"code": 'print("hi")'}
    raw = "<tool_call>" + json.dumps({
        "name": "ipython", "arguments": json.dumps(arguments),
    }) + "</tool_call>"
    _, calls = _split_tool_calls(raw, {"ipython"}, {"ipython": {"code"}})
    assert json.loads(calls[0]["function"]["arguments"]) == arguments


def test_no_offered_tools_means_no_executable_calls():
    assert _split_tool_calls('<tool_call>{"name":"noop"}</tool_call>', set())[1] == []
    assert handler.HermesToolsLLM._known_names({
        "tools": [{"type": "function", "function": {"name": "noop"}}],
        "tool_choice": "none",
    }) == set()


@pytest.mark.parametrize("prefix", ["", "deliberating\n"])
def test_late_implicit_think_closer_is_visible_but_not_executable(prefix):
    raw = (prefix + '<tool_call>{"name":"ipython","arguments":{"code":"print(1)"}}'
           "</tool_call></think>")
    assert _split_tool_calls(raw, {"ipython"}) == (raw, [])
    assert handler.HermesToolsLLM._has_unparsed_call(
        {"choices": [{"message": {"content": raw}}]}, {"ipython"})


@pytest.mark.parametrize("code", ["raise RuntimeError()", 'print("</think>")'])
def test_implicit_reasoning_example_cannot_be_promoted_to_a_real_call(code):
    raw = ("Maybe I could use <tool_call>" + json.dumps({
        "name": "ipython", "arguments": {"code": code},
    }) + "</tool_call>, but that is wrong.</think>Final answer.")
    assert _split_tool_calls(raw, {"ipython"}) == (raw, [])
    assert handler.HermesToolsLLM._has_unparsed_call(
        {"choices": [{"message": {"content": raw}}]}, {"ipython"})


def test_think_closer_inside_an_argument_is_data():
    code = 'print("</think>")'
    raw = "<tool_call>" + json.dumps({
        "name": "ipython", "arguments": {"code": code},
    }) + "</tool_call>"
    assert one_call(raw) == ("ipython", {"code": code})


def test_think_closer_inside_later_calls_does_not_delete_earlier_calls():
    calls = ['<tool_call>{"name":"noop"}</tool_call>', "<tool_call>" + json.dumps({
        "name": "ipython", "arguments": {"code": 'print("</think>")'},
    }) + "</tool_call>"]
    _, parsed = _split_tool_calls("".join(calls), {"noop", "ipython"})
    assert [call["function"]["name"] for call in parsed] == ["noop", "ipython"]


def test_explicit_reasoning_example_is_not_executed_or_retried():
    raw = '<think>Maybe <tool_call>{"name":"example"}</tool_call></think>Done.'
    assert _split_tool_calls(raw, {"example"}) == ("Done.", [])
    assert not handler.HermesToolsLLM._has_unparsed_call(
        {"choices": [{"message": {"content": raw}}]}, {"example"})


@pytest.mark.parametrize("suffix", ["", "</think>"])
def test_unclosed_explicit_reasoning_does_not_execute_tool_examples(suffix):
    # A missing tool closer can hide the reasoning closer inside the parser's
    # unfinished block. Either truncation must remain non-executable.
    raw = '<think>Maybe <tool_call>{"name":"example","arguments":{}}' + suffix
    content, calls = _split_tool_calls(raw, {"example"})
    assert calls == []
    assert content == raw
    assert handler.HermesToolsLLM._has_unparsed_call(
        {"choices": [{"message": {"content": raw}}]}, {"example"})


def test_literal_think_closer_inside_reasoning_example_is_not_a_boundary():
    raw = "<think>Example: <tool_call>" + json.dumps({
        "name": "ipython", "arguments": {"code": 'print("</think>")'},
    }) + "</tool_call></think>Done."
    assert _split_tool_calls(raw, {"ipython"}) == ("Done.", [])


def _success(content):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
    }


@pytest.fixture
def upstream_replies(monkeypatch):
    """Intercept both HTTP clients; no request ever leaves this process."""
    replies, requests = [], []

    def post(client, url, *, headers, json):
        requests.append(json)
        status, body = replies.pop(0)
        request = httpx.Request("POST", url, headers=headers)
        if isinstance(body, str):
            return httpx.Response(status, text=body, request=request)
        return httpx.Response(status, json=body, request=request)

    async def apost(client, url, *, headers, json):
        return post(client, url, headers=headers, json=json)

    async def no_sleep(delay):
        pass

    monkeypatch.setattr(httpx.Client, "post", post)
    monkeypatch.setattr(httpx.AsyncClient, "post", apost)
    monkeypatch.setattr(handler.time, "sleep", lambda delay: None)
    monkeypatch.setattr(handler.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(handler, "_ASYNC_GATE", None)
    monkeypatch.setattr(handler, "MAX_ATTEMPTS", 2)
    return replies, requests


def _completion_kwargs():
    return {
        "model": "hermes-tools/test-model",
        "messages": [{"role": "user", "content": "private prompt"}],
        "api_base": "https://upstream.invalid/v1",
        "api_key": "private-key",
        "optional_params": {"tools": [{
            "type": "function", "function": {
                "name": "ipython", "parameters": {
                    "type": "object", "required": ["code"],
                    "properties": {"code": {"type": "string"}},
                },
            },
        }]},
        "model_response": litellm.ModelResponse(),
    }


def _run_handler(mode, kwargs):
    llm = handler.HermesToolsLLM()
    if mode == "sync":
        return llm.completion(**kwargs)
    if mode == "stream":
        return list(llm.streaming(**kwargs))
    if mode == "async":
        return asyncio.run(llm.acompletion(**kwargs))

    async def consume():
        return [chunk async for chunk in llm.astreaming(**kwargs)]
    return asyncio.run(consume())


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "astream"])
@pytest.mark.parametrize("usage", [None, {"prompt_tokens": 42}, {
    "prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49,
    "prompt_tokens_details": {"cached_tokens": 12},
    "completion_tokens_details": {"reasoning_tokens": 3},
}])
def test_provider_usage_and_missing_fields_survive_proxy(upstream_replies, mode, usage):
    replies, _ = upstream_replies
    replies.append((200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                          "usage": usage}))
    response = _run_handler(mode, _completion_kwargs())
    result = response[-1]["usage"] if mode in {"stream", "astream"} else response.usage.model_dump()
    assert result["prompt_tokens_details"]["provider_usage"] == (usage or {})
    if usage and "completion_tokens_details" in usage:
        assert result["completion_tokens_details"]["reasoning_tokens"] == 3
        assert result["prompt_tokens_details"]["cached_tokens"] == 12


@pytest.mark.parametrize("usage", [None, {
    "prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49,
    "completion_tokens_details": {"reasoning_tokens": 3},
}])
def test_litellm_stream_keeps_raw_usage_for_accounting(monkeypatch, usage):
    upstream = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": usage}
    llm = handler.HermesToolsLLM()
    attempts = [{"status": 200, "usage": usage or {}}]
    response = llm._to_response(upstream, litellm.ModelResponse(), "test-model", known=set(),
                                attempts=attempts)
    monkeypatch.setattr(llm, "completion", lambda **kwargs: response)
    monkeypatch.setattr(litellm, "custom_provider_map", [
        {"provider": "usage-test", "custom_handler": llm},
    ])
    stream = litellm.completion(
        model="usage-test/test-model", messages=[{"role": "user", "content": "test"}],
        stream=True, stream_options={"include_usage": True},
    )
    chunks = [chunk for chunk in stream if getattr(chunk, "usage", None) is not None]
    assert len(chunks) == 1
    # This is the serialization shape used for SSE; absent usage must survive
    # exclude_none so it cannot turn into LiteLLM's synthetic zero counts.
    details = chunks[0].model_dump(exclude_none=True)["usage"]["prompt_tokens_details"]
    assert details["provider_usage"] == (usage or {})
    assert details["proxy_attempts"] == attempts


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "astream"])
@pytest.mark.parametrize("status", [400, 413])
def test_upstream_error_status_and_body_survive_litellm_mapping(upstream_replies, mode, status):
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type

    replies, requests = upstream_replies
    body = {"error": {"message": "This model's maximum context length is 262144 tokens."}}
    replies.append((status, body))
    with pytest.raises(litellm.APIError) as caught:
        _run_handler(mode, _completion_kwargs())
    mapped = exception_type(
        model="test-model", original_exception=caught.value,
        custom_llm_provider="hermes-tools")
    assert mapped.status_code == status
    assert mapped.body == body
    assert "maximum context length is 262144 tokens" in str(mapped)
    assert mapped.response.json() == body
    assert len(requests) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("invalid", [
    '<tool_call>{"name":"ipython"}</tool_call>',
    'Maybe <tool_call>{"name":"ipython","arguments":{"code":"example only"}}'
    "</tool_call>, but no.</think>Done.",
])
def test_invalid_call_is_resampled_using_request_schema(upstream_replies, mode, invalid, caplog):
    replies, requests = upstream_replies
    replies.extend([
        (200, _success(invalid)),
        (200, _success('<tool_call>{"name":"ipython","arguments":{"code":"private output"}}</tool_call>')),
    ])
    with caplog.at_level(logging.INFO, logger=handler.LOGGER.name):
        response = _run_handler(mode, _completion_kwargs())
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "ipython"
    attempts = response.usage.prompt_tokens_details.model_dump()["proxy_attempts"]
    assert len(attempts) == 2
    assert [attempt["usage"]["completion_tokens"] for attempt in attempts] == [7, 7]
    metrics = [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records if record.name == handler.LOGGER.name
    ]
    assert [metric["attempt"] for metric in metrics] == [1, 2]
    assert metrics[0]["request_id"] == metrics[1]["request_id"]
    assert all(metric["upstream_ms"] >= 0 for metric in metrics)
    assert metrics[0]["queue_ms"] >= 0
    assert metrics[0]["completion_tokens"] == 7
    assert "private prompt" not in caplog.text
    assert "private output" not in caplog.text
    assert "private-key" not in caplog.text


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "astream"])
@pytest.mark.parametrize("invalid", [
    '<tool_call>{"name":"phantom"}</tool_call>',
    'Maybe <tool_call>{"name":"ipython","arguments":{"code":"example only"}}'
    "</tool_call>, but no.</think>Done.",
])
def test_exhausted_parser_retries_cannot_become_successful_stop(upstream_replies, mode, invalid):
    replies, requests = upstream_replies
    replies.extend([(200, _success(invalid))] * 2)
    with pytest.raises(litellm.APIError, match="No valid tool call after 2 attempts") as caught:
        _run_handler(mode, _completion_kwargs())
    assert caught.value.status_code == 422
    assert len(requests) == 2


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_transient_http_errors_are_still_retried(upstream_replies, mode):
    replies, requests = upstream_replies
    replies.extend([(429, "slow down"), (200, _success("Done."))])
    response = _run_handler(mode, _completion_kwargs())
    assert response.choices[0].message.content == "Done."
    assert len(requests) == 2


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("tool_choice", [None, "none"])
def test_literal_tool_syntax_without_tools_is_ordinary_text(upstream_replies, mode, tool_choice):
    replies, requests = upstream_replies
    text = 'Example: <tool_call>{"name":"noop"}</tool_call>'
    replies.append((200, _success(text)))
    kwargs = _completion_kwargs()
    if tool_choice is None:
        kwargs["optional_params"] = {}
    else:
        kwargs["optional_params"]["tool_choice"] = tool_choice
    response = _run_handler(mode, kwargs)
    assert response.choices[0].finish_reason == "stop"
    assert "<tool_call>" in response.choices[0].message.content
    assert not response.choices[0].message.tool_calls
    assert len(requests) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
def test_public_litellm_completion_preserves_context_overflow(
        upstream_replies, monkeypatch, stream, is_async):
    replies, requests = upstream_replies
    replies.append((400, {"error": {"message": "maximum context length is 262144 tokens"}}))
    monkeypatch.setattr(litellm, "custom_provider_map", [{
        "provider": "hermes-tools", "custom_handler": handler.HermesToolsLLM(),
    }])
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    kwargs = _completion_kwargs()
    kwargs.pop("model_response")
    kwargs.pop("optional_params")

    async def acompletion():
        response = await litellm.acompletion(**kwargs, stream=stream, num_retries=0)
        if stream:
            return [chunk async for chunk in response]
        return response

    with pytest.raises(litellm.APIError, match="maximum context length is 262144 tokens") as caught:
        if is_async:
            asyncio.run(acompletion())
        else:
            response = litellm.completion(**kwargs, stream=stream, num_retries=0)
            if stream:
                list(response)
    assert caught.value.status_code == 400
    assert len(requests) == 1
