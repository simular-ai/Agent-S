from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gui_agents.common.agent_action_schema import (
    ActionType,
    AgentAction,
    ClickArgs,
    DragAndDropArgs,
    HighlightTextSpanArgs,
    HoldAndPressArgs,
    HotkeyArgs,
    OpenAppArgs,
    SaveToKnowledgeArgs,
    ScrollArgs,
    SetCellValuesArgs,
    SwitchAppArgs,
    TargetSelector,
    TypeArgs,
    WaitArgs,
)

logger = logging.getLogger(__name__)


ACTION_METHOD_BY_TYPE: Dict[ActionType, str] = {
    ActionType.CLICK: "click",
    ActionType.DBLCLICK: "click",
    ActionType.TYPE: "type",
    ActionType.HOTKEY: "hotkey",
    ActionType.WAIT: "wait",
    ActionType.SCROLL: "scroll",
    ActionType.SWITCH_APP: "switch_applications",
    ActionType.FOCUS_APP: "switch_applications",
    ActionType.SWITCH_APPLICATIONS: "switch_applications",
    ActionType.OPEN_APP: "open",
    ActionType.OPEN: "open",
    ActionType.DRAG_AND_DROP: "drag_and_drop",
    ActionType.SAVE_TO_KNOWLEDGE: "save_to_knowledge",
    ActionType.HIGHLIGHT_TEXT_SPAN: "highlight_text_span",
    ActionType.SET_CELL_VALUES: "set_cell_values",
    ActionType.HOLD_AND_PRESS: "hold_and_press",
    ActionType.DONE: "done",
    ActionType.FAIL: "fail",
}


def execute_agent_action(agent: Any, action: AgentAction, obs: Dict[str, Any]) -> str:
    method_name = ACTION_METHOD_BY_TYPE.get(action.type)
    if method_name is None:
        raise RuntimeError(f"Unsupported action type: {action.type}")

    _assign_coordinates_if_supported(agent, action, obs)

    method = getattr(agent, method_name, None)
    if method is None:
        raise RuntimeError(f"Grounding agent is missing method '{method_name}' for action {action.type.value}")

    positional_args, keyword_args = _build_call_arguments(action)
    return method(*positional_args, **keyword_args)


def _assign_coordinates_if_supported(agent: Any, action: AgentAction, obs: Dict[str, Any]) -> None:
    assign_fn = getattr(agent, "assign_coordinates", None)
    if callable(assign_fn):
        try:
            assign_fn(action, obs)
        except TypeError:
            # Legacy signature (str, obs). Fall back to manual assignment below.
            logger.debug("assign_coordinates did not accept AgentAction payload", exc_info=True)
        except Exception:
            logger.debug("assign_coordinates raised during structured dispatch", exc_info=True)
        else:
            if not _coordinates_available(agent, action):
                logger.debug("assign_coordinates returned without coordinates; falling back")
            else:
                return
    _fallback_coordinate_assignment(agent, action, obs)



def _coordinates_available(agent: Any, action: AgentAction) -> bool:
    if action.type in {ActionType.CLICK, ActionType.DBLCLICK, ActionType.SCROLL}:
        return getattr(agent, "coords1", None) is not None
    if action.type == ActionType.TYPE:
        target = getattr(action.args, "target", None)
        if target is None:
            return True
        return getattr(agent, "coords1", None) is not None
    if action.type == ActionType.DRAG_AND_DROP:
        return getattr(agent, "coords1", None) is not None and getattr(agent, "coords2", None) is not None
    if action.type == ActionType.HIGHLIGHT_TEXT_SPAN:
        return getattr(agent, "coords1", None) is not None and getattr(agent, "coords2", None) is not None
    return True


def _fallback_coordinate_assignment(agent: Any, action: AgentAction, obs: Dict[str, Any]) -> None:
    # Provide basic bbox-driven coordinate assignment as a safety net when the agent does not support structured payloads yet.
    try:
        if action.type in {ActionType.CLICK, ActionType.DBLCLICK, ActionType.SCROLL}:
            target = getattr(action.args, "target", None)
            point = _resolve_target_to_point(agent, target, obs)
            if point is not None:
                setattr(agent, "coords1", point)
        elif action.type == ActionType.TYPE:
            target = getattr(action.args, "target", None)
            if target is not None:
                point = _resolve_target_to_point(agent, target, obs)
                if point is not None:
                    setattr(agent, "coords1", point)
        elif action.type == ActionType.DRAG_AND_DROP:
            start = getattr(action.args, "start", None)
            end = getattr(action.args, "end", None)
            point1 = _resolve_target_to_point(agent, start, obs)
            point2 = _resolve_target_to_point(agent, end, obs)
            if point1 is not None:
                setattr(agent, "coords1", point1)
            if point2 is not None:
                setattr(agent, "coords2", point2)
    except Exception:
        logger.debug("Fallback coordinate assignment failed", exc_info=True)


def _resolve_target_to_point(agent: Any, target: Optional[TargetSelector], obs: Dict[str, Any]) -> Optional[List[int]]:
    if target is None:
        return None
    if target.bbox is not None:
        return _center_from_bounds(target.bbox)
    if target.a11y_id is not None:
        bounds = _find_bounds_by_id(obs, target.a11y_id)
        if bounds is not None:
            return _center_from_bounds(bounds)
    if target.description is not None:
        resolver = getattr(agent, "generate_coords", None)
        if callable(resolver):
            try:
                coords = resolver(target.description, obs)
                if isinstance(coords, Iterable) and len(coords) == 2:
                    return [int(coords[0]), int(coords[1])]
            except Exception:
                logger.debug("generate_coords failed while resolving description", exc_info=True)
    return None


def _find_bounds_by_id(obs: Dict[str, Any], target_id: str) -> Optional[Tuple[int, int, int, int]]:
    for tree in _candidate_trees(obs):
        for node in _iter_nodes(tree):
            if not isinstance(node, dict):
                continue
            node_id = node.get("id") or node.get("nodeId") or node.get("node_id") or node.get("a11y_id")
            if node_id is None:
                continue
            if str(node_id) != str(target_id):
                continue
            bounds = (
                node.get("bounds")
                or node.get("bounding_box")
                or node.get("bbox")
                or node.get("frame")
            )
            if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
                try:
                    return tuple(int(v) for v in bounds)
                except Exception:
                    continue
    return None


def _candidate_trees(obs: Dict[str, Any]) -> Iterable[Any]:
    for key in ("a11y_tree", "serialized_a11y_tree", "tree", "som", "nodes"):
        tree = obs.get(key)
        if tree:
            yield tree


def _iter_nodes(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, dict):
            children = current.get("children") or current.get("nodes") or current.get("children_list")
            if isinstance(children, dict):
                stack.extend(children.values())
            elif isinstance(children, list):
                stack.extend(children)
        elif isinstance(current, list):
            stack.extend(current)


def _center_from_bounds(bounds: Tuple[int, int, int, int]) -> List[int]:
    x0, y0, x1, y1 = bounds
    if x1 >= x0 and y1 >= y0:
        return [int((x0 + x1) / 2), int((y0 + y1) / 2)]
    # Treat as (x, y, width, height)
    return [int(x0 + x1 / 2), int(y0 + y1 / 2)]


def _target_description(target: Optional[TargetSelector]) -> str:
    if target is None:
        return ""
    if target.description is not None:
        return target.description
    if target.a11y_id is not None:
        return f"a11y_id:{target.a11y_id}"
    if target.bbox is not None:
        return f"bbox:{','.join(str(v) for v in target.bbox)}"
    return ""


def _build_call_arguments(action: AgentAction) -> Tuple[List[Any], Dict[str, Any]]:
    args = action.args
    if action.type in {ActionType.CLICK, ActionType.DBLCLICK} and isinstance(args, ClickArgs):
        desc = _target_description(args.target)
        num_clicks = args.count if action.type == ActionType.CLICK else max(2, args.count)
        return [desc], {
            "num_clicks": num_clicks,
            "button_type": args.button,
            "hold_keys": list(args.hold_keys),
        }
    if action.type == ActionType.TYPE and isinstance(args, TypeArgs):
        kwargs: Dict[str, Any] = {
            "text": args.text,
            "overwrite": args.overwrite,
            "enter": args.enter,
        }
        if args.target is not None:
            kwargs["element_description"] = _target_description(args.target)
        return [], kwargs
    if action.type == ActionType.HOTKEY and isinstance(args, HotkeyArgs):
        return [list(args.keys)], {}
    if action.type == ActionType.WAIT and isinstance(args, WaitArgs):
        return [args.seconds], {}
    if action.type == ActionType.SCROLL and isinstance(args, ScrollArgs):
        return [
            _target_description(args.target)
        ], {
            "clicks": args.clicks,
            "shift": args.shift,
        }
    if action.type in {ActionType.SWITCH_APP, ActionType.FOCUS_APP, ActionType.SWITCH_APPLICATIONS} and isinstance(args, SwitchAppArgs):
        return [args.app_id], {}
    if action.type in {ActionType.OPEN_APP, ActionType.OPEN} and isinstance(args, OpenAppArgs):
        return [args.app_name], {}
    if action.type == ActionType.DRAG_AND_DROP and isinstance(args, DragAndDropArgs):
        return [
            _target_description(args.start),
            _target_description(args.end),
        ], {"hold_keys": list(args.hold_keys)}
    if action.type == ActionType.SAVE_TO_KNOWLEDGE and isinstance(args, SaveToKnowledgeArgs):
        return [list(args.text)], {}
    if action.type == ActionType.HIGHLIGHT_TEXT_SPAN and isinstance(args, HighlightTextSpanArgs):
        return [args.start_phrase, args.end_phrase], {"button": args.button}
    if action.type == ActionType.SET_CELL_VALUES and isinstance(args, SetCellValuesArgs):
        return [
            dict(args.cell_values),
            args.app_name,
            args.sheet_name,
        ], {}
    if action.type == ActionType.HOLD_AND_PRESS and isinstance(args, HoldAndPressArgs):
        return [], {
            "hold_keys": list(args.hold_keys),
            "press_keys": list(args.press_keys),
        }
    if action.type == ActionType.DONE:
        return [], {"return_value": getattr(args, "return_value", None)}
    if action.type == ActionType.FAIL:
        return [], {}
    raise RuntimeError(f"Unsupported or mismatched arguments for action type: {action.type}")

