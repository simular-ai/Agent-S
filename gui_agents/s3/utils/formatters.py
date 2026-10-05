"""This file contains various formatting checks used to reprompt an agent for correctly formatted responses."""

import ast
import inspect

from gui_agents.s3.utils.common_utils import (
    extract_agent_functions,
    parse_code_from_string,
    split_thinking_response,
)

single_action_check = (
    lambda response: len(extract_agent_functions(parse_code_from_string(response))) == 1
)
single_action_error_msg = (
    "Incorrect code: There must be a single agent action in the code response."
)
SINGLE_ACTION_FORMATTER = lambda response: (
    single_action_check(response),
    single_action_error_msg,
)


def _valid_action_signature(agent, code):
    """Check one agent action and its literal arguments without executing it."""
    try:
        action = ast.parse(code.strip(), mode="eval").body
        if not (
            isinstance(action, ast.Call)
            and isinstance(action.func, ast.Attribute)
            and isinstance(action.func.value, ast.Name)
            and action.func.value.id == "agent"
        ):
            return False

        method = getattr(agent, action.func.attr, None)
        if not callable(method) or not getattr(method, "is_agent_action", False):
            return False

        # Reject argument expressions that would themselves execute code.
        args = [ast.literal_eval(arg) for arg in action.args]
        names = [keyword.arg for keyword in action.keywords]
        if None in names or len(names) != len(set(names)):
            return False
        kwargs = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in action.keywords
        }
        inspect.signature(method).bind(*args, **kwargs)
        return True
    except (SyntaxError, ValueError, TypeError, AttributeError):
        return False


code_valid_check = lambda agent, obs, response: _valid_action_signature(
    agent, parse_code_from_string(response)
)
code_valid_error_msg = "Incorrect code: The agent action must be a valid function and use valid parameters from the docstring list."
CODE_VALID_FORMATTER = lambda agent, obs, response: (
    code_valid_check(agent, obs, response),
    code_valid_error_msg,
)

thoughts_answer_tag_check = lambda response: split_thinking_response(response)[1] != ""
thoughts_answer_tag_error_msg = "Incorrect response: The response must contain both <thoughts>...</thoughts> and <answer>...</answer> tags."
THOUGHTS_ANSWER_TAG_FORMATTER = lambda response: (
    thoughts_answer_tag_check(response),
    thoughts_answer_tag_error_msg,
)

integer_answer_check = (
    lambda response: split_thinking_response(response)[0].strip().isdigit()
)
integer_answer_error_msg = (
    "Incorrect response: The <answer>...</answer> tag must contain a single integer."
)
INTEGER_ANSWER_FORMATTER = lambda response: (
    integer_answer_check(response),
    integer_answer_error_msg,
)
