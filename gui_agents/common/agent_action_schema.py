from __future__ import annotations

import json
import textwrap
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class AgentActionParseError(ValueError):
    """Raised when an agent action payload cannot be parsed or validated."""


class TargetSelector(BaseModel):
    a11y_id: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    description: Optional[str] = None

    class Config:
        extra = "forbid"

    @field_validator("a11y_id", "description")
    @classmethod
    def _strip_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text selectors must be non-empty when provided")
        return cleaned

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("Bounding boxes must contain exactly four integers")
        return tuple(int(v) for v in value)

    @model_validator(mode="after")
    def _check_exactly_one_selector(self) -> "TargetSelector":
        provided = [
            name
            for name in ("a11y_id", "bbox", "description")
            if getattr(self, name) not in (None, "")
        ]
        if len(provided) != 1:
            raise ValueError("Provide exactly one of a11y_id, bbox, or description in target selector")
        return self


class ArgsModel(BaseModel):
    class Config:
        extra = "forbid"


class ClickArgs(ArgsModel):
    target: TargetSelector
    button: str = Field(default="left")
    count: int = Field(default=1, ge=1, le=4)
    hold_keys: List[str] = Field(default_factory=list)

    @field_validator("button")
    @classmethod
    def _validate_button(cls, value: str) -> str:
        allowed = {"left", "right", "middle"}
        lowered = value.lower().strip()
        if lowered not in allowed:
            raise ValueError(f"Button must be one of {sorted(allowed)}")
        return lowered

    @field_validator("hold_keys", mode="after")
    @classmethod
    def _clean_keys(cls, values: List[str]) -> List[str]:
        cleaned = [item.strip() for item in values if item.strip()]
        if len(cleaned) != len(values):
            raise ValueError("Hold key entries must be non-empty strings")
        if len(cleaned) > 5:
            raise ValueError("Hold keys cannot exceed 5 entries")
        return cleaned


class DragAndDropArgs(ArgsModel):
    start: TargetSelector
    end: TargetSelector
    hold_keys: List[str] = Field(default_factory=list)

    @field_validator("hold_keys", mode="after")
    @classmethod
    def _clean_hold_keys(cls, values: List[str]) -> List[str]:
        cleaned = [item.strip() for item in values if item.strip()]
        if len(cleaned) != len(values):
            raise ValueError("Hold key entries must be non-empty strings")
        if len(cleaned) > 4:
            raise ValueError("Hold keys cannot exceed 4 entries")
        return cleaned


class TypeArgs(ArgsModel):
    text: str = Field(..., max_length=512)
    target: Optional[TargetSelector] = None
    overwrite: bool = False
    enter: bool = False

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip("\n"):
            raise ValueError("Typing text must be non-empty")
        return value


class HotkeyArgs(ArgsModel):
    keys: List[str]

    @field_validator("keys")
    @classmethod
    def _ensure_bounds(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("At least one key must be provided")
        if len(values) > 5:
            raise ValueError("Hotkey sequences cannot exceed 5 keys")
        cleaned = [item.strip() for item in values if item.strip()]
        if len(cleaned) != len(values):
            raise ValueError("Hotkey values must be non-empty strings")
        return cleaned


class HoldAndPressArgs(ArgsModel):
    hold_keys: List[str] = Field(default_factory=list)
    press_keys: List[str]

    @field_validator("hold_keys")
    @classmethod
    def _clean_hold(cls, values: List[str]) -> List[str]:
        cleaned = [item.strip() for item in values if item.strip()]
        if len(cleaned) != len(values):
            raise ValueError("Hold key entries must be non-empty strings")
        if len(cleaned) > 5:
            raise ValueError("Hold keys cannot exceed 5 entries")
        return cleaned

    @field_validator("press_keys")
    @classmethod
    def _clean_press(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("At least one key must be pressed")
        cleaned = [item.strip() for item in values if item.strip()]
        if len(cleaned) != len(values):
            raise ValueError("Press key entries must be non-empty strings")
        if len(cleaned) > 5:
            raise ValueError("Press sequences cannot exceed 5 keys")
        return cleaned


class ScrollArgs(ArgsModel):
    target: TargetSelector
    clicks: int
    shift: bool = False


class SwitchAppArgs(ArgsModel):
    app_id: str

    @field_validator("app_id")
    @classmethod
    def _clean_app(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Application identifier must be non-empty")
        return cleaned


class OpenAppArgs(ArgsModel):
    app_name: str

    @field_validator("app_name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Application name must be non-empty")
        return cleaned


class SaveToKnowledgeArgs(ArgsModel):
    text: List[str]

    @field_validator("text")
    @classmethod
    def _validate_saved_text(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("At least one string must be provided")
        if len(values) > 20:
            raise ValueError("Saved knowledge entries cannot exceed 20 strings")
        cleaned = [item.strip() for item in values if item.strip()]
        if len(cleaned) != len(values):
            raise ValueError("Saved knowledge entries must be non-empty strings")
        return cleaned


class HighlightTextSpanArgs(ArgsModel):
    start_phrase: str
    end_phrase: str
    button: str = Field(default="left")

    @field_validator("start_phrase", "end_phrase")
    @classmethod
    def _non_empty_phrase(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Highlight phrases must be non-empty")
        return cleaned

    @field_validator("button")
    @classmethod
    def _validate_button(cls, value: str) -> str:
        allowed = {"left", "right", "middle"}
        lowered = value.lower().strip()
        if lowered not in allowed:
            raise ValueError(f"Button must be one of {sorted(allowed)}")
        return lowered


class SetCellValuesArgs(ArgsModel):
    cell_values: Dict[str, Any]
    app_name: str
    sheet_name: str

    @field_validator("app_name", "sheet_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Spreadsheet metadata must be non-empty strings")
        return cleaned


class WaitArgs(ArgsModel):
    seconds: float = Field(..., ge=0.0, le=30.0)


class DoneArgs(ArgsModel):
    return_value: Optional[Any] = None


class FailArgs(ArgsModel):
    reason: Optional[str] = None


class ActionType(str, Enum):
    CLICK = "click"
    DBLCLICK = "dblclick"
    TYPE = "type"
    HOTKEY = "hotkey"
    WAIT = "wait"
    SCROLL = "scroll"
    SWITCH_APP = "switch_app"
    FOCUS_APP = "focus_app"
    SWITCH_APPLICATIONS = "switch_applications"
    OPEN_APP = "open_app"
    OPEN = "open"
    DRAG_AND_DROP = "drag_and_drop"
    SAVE_TO_KNOWLEDGE = "save_to_knowledge"
    HIGHLIGHT_TEXT_SPAN = "highlight_text_span"
    SET_CELL_VALUES = "set_cell_values"
    HOLD_AND_PRESS = "hold_and_press"
    DONE = "done"
    FAIL = "fail"


ACTION_ARGS_BY_TYPE: Dict[ActionType, type[ArgsModel]] = {
    ActionType.CLICK: ClickArgs,
    ActionType.DBLCLICK: ClickArgs,
    ActionType.TYPE: TypeArgs,
    ActionType.HOTKEY: HotkeyArgs,
    ActionType.WAIT: WaitArgs,
    ActionType.SCROLL: ScrollArgs,
    ActionType.SWITCH_APP: SwitchAppArgs,
    ActionType.FOCUS_APP: SwitchAppArgs,
    ActionType.SWITCH_APPLICATIONS: SwitchAppArgs,
    ActionType.OPEN_APP: OpenAppArgs,
    ActionType.OPEN: OpenAppArgs,
    ActionType.DRAG_AND_DROP: DragAndDropArgs,
    ActionType.SAVE_TO_KNOWLEDGE: SaveToKnowledgeArgs,
    ActionType.HIGHLIGHT_TEXT_SPAN: HighlightTextSpanArgs,
    ActionType.SET_CELL_VALUES: SetCellValuesArgs,
    ActionType.HOLD_AND_PRESS: HoldAndPressArgs,
    ActionType.DONE: DoneArgs,
    ActionType.FAIL: FailArgs,
}


class AgentActionMeta(BaseModel):
    idempotency_key: str
    roi_hash: Optional[str] = None
    explanation: Optional[str] = Field(default=None, max_length=280)

    class Config:
        extra = "forbid"

    @field_validator("idempotency_key")
    @classmethod
    def _clean_idempotency(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("idempotency_key must be a non-empty string")
        return cleaned

    @field_validator("roi_hash", "explanation")
    @classmethod
    def _trim_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class AgentAction(BaseModel):
    type: ActionType
    args: ArgsModel
    meta: AgentActionMeta

    class Config:
        extra = "forbid"

    @model_validator(mode="before")
    @classmethod
    def _apply_args_model(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise AgentActionParseError("AgentAction payload must be a JSON object")
        raw_type = data.get("type")
        if raw_type is None:
            raise AgentActionParseError("Missing action type")
        try:
            action_type = ActionType(raw_type)
        except ValueError as exc:
            raise AgentActionParseError(f"Unsupported action type: {raw_type}") from exc

        args_model = ACTION_ARGS_BY_TYPE.get(action_type)
        if args_model is None:
            raise AgentActionParseError(f"No argument model registered for action type: {action_type.value}")

        args_payload = data.get("args") or {}
        try:
            args_instance = args_model.model_validate(args_payload)
        except Exception as exc:
            raise AgentActionParseError(str(exc)) from exc

        data["type"] = action_type
        data["args"] = args_instance
        return data

    @model_validator(mode="after")
    def _normalize_args(self) -> "AgentAction":
        if self.type is ActionType.DBLCLICK and isinstance(self.args, ClickArgs):
            if self.args.count < 2:
                self.args.count = 2
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


_JSON_BLOCK_PREFIX = "```json"
_JSON_BLOCK_SUFFIX = "```"


def extract_agent_action_json(payload: str) -> str:
    """Return the first JSON object inside a fenced block or raise."""
    if not payload:
        raise AgentActionParseError("Empty response from model")
    lowered = payload.lower()
    start = lowered.find(_JSON_BLOCK_PREFIX)
    if start != -1:
        start = payload.find("{", start)
        if start == -1:
            raise AgentActionParseError("JSON block started but no object found")
        end = payload.find(_JSON_BLOCK_SUFFIX, start)
        if end == -1:
            raise AgentActionParseError("JSON block not properly terminated")
        return payload[start:end].strip()
    stripped = payload.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    raise AgentActionParseError("No JSON object found in model response")


def parse_agent_action(payload: str) -> AgentAction:
    """Parse and validate an AgentAction from model output."""
    json_blob = extract_agent_action_json(payload)
    try:
        data = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        raise AgentActionParseError(f"Invalid JSON action: {exc}") from exc
    try:
        return AgentAction.model_validate(data)
    except AgentActionParseError:
        raise
    except Exception as exc:
        raise AgentActionParseError(str(exc)) from exc


_TARGET_SELECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "a11y_id": {"type": "string"},
        "bbox": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        },
        "description": {"type": "string"},
    },
    "oneOf": [
        {"required": ["a11y_id"]},
        {"required": ["bbox"]},
        {"required": ["description"]},
    ],
}


ARGUMENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    ActionType.CLICK.value: {
        "type": "object",
        "required": ["target"],
        "additionalProperties": False,
        "properties": {
            "target": _TARGET_SELECTOR_SCHEMA,
            "button": {"enum": ["left", "right", "middle"]},
            "count": {"type": "integer", "minimum": 1, "maximum": 4},
            "hold_keys": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 5,
            },
        },
    },
    ActionType.DBLCLICK.value: {
        "type": "object",
        "required": ["target"],
        "additionalProperties": False,
        "properties": {
            "target": _TARGET_SELECTOR_SCHEMA,
            "button": {"enum": ["left", "right", "middle"]},
            "count": {"type": "integer", "minimum": 1, "maximum": 4},
            "hold_keys": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 5,
            },
        },
    },
    ActionType.TYPE.value: {
        "type": "object",
        "required": ["text"],
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string", "maxLength": 512},
            "target": _TARGET_SELECTOR_SCHEMA,
            "overwrite": {"type": "boolean"},
            "enter": {"type": "boolean"},
        },
    },
    ActionType.HOTKEY.value: {
        "type": "object",
        "required": ["keys"],
        "additionalProperties": False,
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 5,
            }
        },
    },
    ActionType.WAIT.value: {
        "type": "object",
        "required": ["seconds"],
        "additionalProperties": False,
        "properties": {
            "seconds": {"type": "number", "minimum": 0.0, "maximum": 30.0}
        },
    },
    ActionType.SCROLL.value: {
        "type": "object",
        "required": ["target", "clicks"],
        "additionalProperties": False,
        "properties": {
            "target": _TARGET_SELECTOR_SCHEMA,
            "clicks": {"type": "integer"},
            "shift": {"type": "boolean"},
        },
    },
    ActionType.SWITCH_APP.value: {
        "type": "object",
        "required": ["app_id"],
        "additionalProperties": False,
        "properties": {"app_id": {"type": "string", "minLength": 1}},
    },
    ActionType.FOCUS_APP.value: {
        "type": "object",
        "required": ["app_id"],
        "additionalProperties": False,
        "properties": {"app_id": {"type": "string", "minLength": 1}},
    },
    ActionType.SWITCH_APPLICATIONS.value: {
        "type": "object",
        "required": ["app_id"],
        "additionalProperties": False,
        "properties": {"app_id": {"type": "string", "minLength": 1}},
    },
    ActionType.OPEN_APP.value: {
        "type": "object",
        "required": ["app_name"],
        "additionalProperties": False,
        "properties": {"app_name": {"type": "string", "minLength": 1}},
    },
    ActionType.OPEN.value: {
        "type": "object",
        "required": ["app_name"],
        "additionalProperties": False,
        "properties": {"app_name": {"type": "string", "minLength": 1}},
    },
    ActionType.DRAG_AND_DROP.value: {
        "type": "object",
        "required": ["start", "end"],
        "additionalProperties": False,
        "properties": {
            "start": _TARGET_SELECTOR_SCHEMA,
            "end": _TARGET_SELECTOR_SCHEMA,
            "hold_keys": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 4,
            },
        },
    },
    ActionType.SAVE_TO_KNOWLEDGE.value: {
        "type": "object",
        "required": ["text"],
        "additionalProperties": False,
        "properties": {
            "text": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 20,
            }
        },
    },
    ActionType.HIGHLIGHT_TEXT_SPAN.value: {
        "type": "object",
        "required": ["start_phrase", "end_phrase"],
        "additionalProperties": False,
        "properties": {
            "start_phrase": {"type": "string", "minLength": 1},
            "end_phrase": {"type": "string", "minLength": 1},
            "button": {"enum": ["left", "right", "middle"]},
        },
    },
    ActionType.SET_CELL_VALUES.value: {
        "type": "object",
        "required": ["cell_values", "app_name", "sheet_name"],
        "additionalProperties": False,
        "properties": {
            "cell_values": {
                "type": "object",
                "additionalProperties": True,
                "minProperties": 1,
            },
            "app_name": {"type": "string", "minLength": 1},
            "sheet_name": {"type": "string", "minLength": 1},
        },
    },
    ActionType.HOLD_AND_PRESS.value: {
        "type": "object",
        "required": ["press_keys"],
        "additionalProperties": False,
        "properties": {
            "hold_keys": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "maxItems": 5,
            },
            "press_keys": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 5,
            },
        },
    },
    ActionType.DONE.value: {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "return_value": {}
        },
    },
    ActionType.FAIL.value: {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reason": {"type": "string"}
        },
    },
}


META_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["idempotency_key"],
    "additionalProperties": False,
    "properties": {
        "idempotency_key": {"type": "string", "minLength": 1},
        "roi_hash": {"type": "string", "minLength": 1},
        "explanation": {"type": "string", "maxLength": 280},
    },
}


def _action_all_of() -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for action, schema in ARGUMENT_SCHEMAS.items():
        rules.append(
            {
                "if": {"properties": {"type": {"const": action}}},
                "then": {"properties": {"args": schema}, "required": ["args"]},
            }
        )
    return rules


AGENT_ACTION_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentAction",
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "args", "meta"],
    "properties": {
        "type": {"enum": [member.value for member in ActionType]},
        "args": {"type": "object"},
        "meta": META_SCHEMA,
    },
    "allOf": _action_all_of(),
}


AGENT_ACTION_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_action",
        "schema": AGENT_ACTION_JSON_SCHEMA,
    },
}


SCHEMA_PROMPT_FRAGMENT = textwrap.dedent(
    f"""
    (Grounded Action)
    Emit exactly one JSON object describing the next UI action. The JSON must appear inside a \"```json\" fenced block and conform to the AgentAction schema below.
    ```json
    {json.dumps(AGENT_ACTION_JSON_SCHEMA, indent=2)}
    ```
    Do not add commentary before or after the JSON block. Never emit multiple JSON blocks.
    """
)


def agent_action_to_dict(action: AgentAction) -> Dict[str, Any]:
    """Return a plain dict representation suitable for logging/telemetry."""
    return action.model_dump(mode="python")

_JSON_BLOCK_PREFIX = "```json"
_JSON_BLOCK_SUFFIX = "```"


def extract_agent_action_json(payload: str) -> str:
    """Return the first JSON object inside a fenced block or raise."""
    if not payload:
        raise AgentActionParseError("Empty response from model")
    lowered = payload.lower()
    start = lowered.find(_JSON_BLOCK_PREFIX)
    if start != -1:
        start = payload.find("{", start)
        if start == -1:
            raise AgentActionParseError("JSON block started but no object found")
        end = payload.find(_JSON_BLOCK_SUFFIX, start)
        if end == -1:
            raise AgentActionParseError("JSON block not properly terminated")
        return payload[start:end].strip()
    stripped = payload.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    raise AgentActionParseError("No JSON object found in model response")


def parse_agent_action(payload: str) -> AgentAction:
    """Parse and validate an AgentAction from model output."""
    json_blob = extract_agent_action_json(payload)
    try:
        data = json.loads(json_blob)
    except json.JSONDecodeError as exc:
        raise AgentActionParseError(f"Invalid JSON action: {exc}") from exc
    try:
        return AgentAction.model_validate(data)
    except AgentActionParseError:
        raise
    except Exception as exc:
        raise AgentActionParseError(str(exc)) from exc
