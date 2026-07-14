"""Strict parsing and validation for observation-only Agent S proposals."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable


MAX_CODE_CHARS = 4096
MAX_AST_NODES = 64
MAX_DESCRIPTION_CHARS = 512
MAX_TEXT_CHARS = 2048


class ActionParseError(ValueError):
    """Raised when model output is not exactly one supported literal action."""


@dataclass(frozen=True)
class ActionCall:
    """A validated action call that is safe to inspect but is never executed."""

    kind: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionProposal:
    """A grounded, immutable proposal returned to Codex for review."""

    proposal_id: str
    task_id: str
    step: int
    action: ActionCall
    summary: str
    screenshot_sha256: str
    created_at: str
    target: dict[str, int] | None = None
    risk_class: str = "proposal_only"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.to_dict()
        return payload


def _click(
    element_description: str,
    num_clicks: int = 1,
    button_type: str = "left",
    hold_keys: list[str] | None = None,
) -> None:
    """Signature-only action specification."""


def _type(
    element_description: str | None = None,
    text: str = "",
    overwrite: bool = False,
    enter: bool = False,
) -> None:
    """Signature-only action specification."""


def _scroll(element_description: str, clicks: int, shift: bool = False) -> None:
    """Signature-only action specification."""


def _hotkey(keys: list[str]) -> None:
    """Signature-only action specification."""


def _wait(time: float) -> None:
    """Signature-only action specification."""


def _done() -> None:
    """Signature-only action specification."""


def _fail() -> None:
    """Signature-only action specification."""


_ACTION_SIGNATURES: dict[str, Callable[..., None]] = {
    "click": _click,
    "type": _type,
    "scroll": _scroll,
    "hotkey": _hotkey,
    "wait": _wait,
    "done": _done,
    "fail": _fail,
}

_ALLOWED_HOTKEYS = {
    "alt",
    "backspace",
    "ctrl",
    "delete",
    "down",
    "end",
    "enter",
    "esc",
    "escape",
    "home",
    "left",
    "pagedown",
    "pageup",
    "right",
    "shift",
    "space",
    "tab",
    "up",
}
_ALLOWED_HOTKEYS.update(chr(code) for code in range(ord("a"), ord("z") + 1))
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_action_code(response: str) -> str:
    """Extract exactly one fenced action expression from a model response."""

    if len(response) > 32_000:
        raise ActionParseError("Model response is too large")
    matches = _CODE_BLOCK_RE.findall(response)
    if len(matches) != 1:
        raise ActionParseError("Response must contain exactly one fenced action")
    code = matches[0].strip()
    if not code or len(code) > MAX_CODE_CHARS:
        raise ActionParseError("Action is empty or too large")
    return code


def _literal(node: ast.AST, *, depth: int = 0) -> Any:
    if depth > 3:
        raise ActionParseError("Literal nesting is too deep")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        raise ActionParseError("Unsupported literal type")
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > 10:
            raise ActionParseError("Literal list is too long")
        return [_literal(item, depth=depth + 1) for item in node.elts]
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return -node.operand.value
    raise ActionParseError("Arguments must be simple literals")


def _require_description(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ActionParseError(f"{field} must be a string")
    value = value.strip()
    if not 8 <= len(value) <= MAX_DESCRIPTION_CHARS:
        raise ActionParseError(f"{field} must be 8-{MAX_DESCRIPTION_CHARS} characters")
    return value


def _validate(call: ActionCall) -> ActionCall:
    args = dict(call.arguments)
    if call.kind == "click":
        args["element_description"] = _require_description(
            args["element_description"], "element_description"
        )
        if type(args["num_clicks"]) is not int or not 1 <= args["num_clicks"] <= 2:
            raise ActionParseError("num_clicks must be 1 or 2")
        if args["button_type"] not in {"left", "middle", "right"}:
            raise ActionParseError("Unsupported mouse button")
        if args["hold_keys"] is None:
            args["hold_keys"] = []
        if not isinstance(args["hold_keys"], list) or len(args["hold_keys"]) > 3:
            raise ActionParseError("hold_keys must be a short list")
        if any(key not in _ALLOWED_HOTKEYS for key in args["hold_keys"]):
            raise ActionParseError("Unsupported hold key")
    elif call.kind == "type":
        if args["element_description"] is not None:
            args["element_description"] = _require_description(
                args["element_description"], "element_description"
            )
        if not isinstance(args["text"], str) or len(args["text"]) > MAX_TEXT_CHARS:
            raise ActionParseError(f"text must be at most {MAX_TEXT_CHARS} characters")
        if type(args["overwrite"]) is not bool or type(args["enter"]) is not bool:
            raise ActionParseError("overwrite and enter must be booleans")
    elif call.kind == "scroll":
        args["element_description"] = _require_description(
            args["element_description"], "element_description"
        )
        if type(args["clicks"]) is not int or args["clicks"] == 0:
            raise ActionParseError("clicks must be a non-zero integer")
        if not -10 <= args["clicks"] <= 10:
            raise ActionParseError("clicks must be between -10 and 10")
        if type(args["shift"]) is not bool:
            raise ActionParseError("shift must be a boolean")
    elif call.kind == "hotkey":
        keys = args["keys"]
        if not isinstance(keys, list) or not 1 <= len(keys) <= 3:
            raise ActionParseError("keys must contain 1-3 entries")
        if any(not isinstance(key, str) or key not in _ALLOWED_HOTKEYS for key in keys):
            raise ActionParseError("Unsupported hotkey")
    elif call.kind == "wait":
        value = args["time"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ActionParseError("time must be numeric")
        if not 0 <= float(value) <= 5:
            raise ActionParseError("time must be between 0 and 5 seconds")
    return ActionCall(kind=call.kind, arguments=args)


def parse_action_call(code: str) -> ActionCall:
    """Parse one exact ``agent.method(literals)`` expression without evaluation."""

    if not code or len(code) > MAX_CODE_CHARS:
        raise ActionParseError("Action is empty or too large")
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as exc:
        raise ActionParseError("Action is not a single expression") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ActionParseError("Action syntax is too complex")
    call = tree.body
    if not isinstance(call, ast.Call):
        raise ActionParseError("Action must be a function call")
    if not (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "agent"
    ):
        raise ActionParseError("Only agent methods are allowed")
    method = call.func.attr
    signature_target = _ACTION_SIGNATURES.get(method)
    if signature_target is None:
        raise ActionParseError(f"Unsupported action: {method}")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ActionParseError("Expanded keyword arguments are not allowed")
    keyword_names = [keyword.arg for keyword in call.keywords]
    if len(keyword_names) != len(set(keyword_names)):
        raise ActionParseError("Duplicate keyword argument")
    positional = [_literal(node) for node in call.args]
    keywords = {keyword.arg: _literal(keyword.value) for keyword in call.keywords}
    try:
        bound = inspect.signature(signature_target).bind(*positional, **keywords)
    except TypeError as exc:
        raise ActionParseError(f"Invalid {method} arguments") from exc
    bound.apply_defaults()
    return _validate(ActionCall(method, dict(bound.arguments)))
