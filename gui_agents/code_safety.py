"""Code safety validation for model-generated Python code execution."""

import ast
import logging
import builtins as builtins_module
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class CodeSafetyError(Exception):
    pass


class _SafetyValidator(ast.NodeVisitor):
    def __init__(self, allow_agent_calls_only: bool = False):
        self.errors = []
        self.allow_agent_calls_only = allow_agent_calls_only
        self.found_agent_call = False

    def _is_dangerous_name(self, name: str) -> bool:
        dangerous_builtins = {
            "__import__", "eval", "exec", "compile", "globals", "locals",
            "vars", "getattr", "setattr", "delattr", "open", "input", "breakpoint"
        }
        return name in dangerous_builtins

    def _is_dangerous_module(self, name: str) -> bool:
        dangerous_modules = {
            "os", "sys", "subprocess", "shutil", "pathlib", "socket",
            "requests", "urllib", "http", "ftplib", "smtplib", "poplib",
            "imaplib", "telnetlib", "sqlite3", "pickle", "shelve", "marshal",
            "ctypes", "cffi", "multiprocessing", "threading", "concurrent", "asyncio"
        }
        return name.split(".")[0] in dangerous_modules

    def _is_dangerous_attr(self, attr: str) -> bool:
        dangerous_attrs = {
            "system", "popen", "spawn", "execv", "execve", "execvp", "execvpe",
            "fork", "kill", "terminate", "remove", "unlink", "rmdir", "removedirs",
            "rename", "renames", "chmod", "chown", "mkdir", "makedirs", "write",
            "writelines", "truncate"
        }
        return attr in dangerous_attrs

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if self._is_dangerous_module(module_name):
                self.errors.append(f"Blocked import of dangerous module: {alias.name}")
            elif self.allow_agent_calls_only and module_name != "":
                self.errors.append(f"Imports not allowed in agent action: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            module_name = node.module.split(".")[0]
            if self._is_dangerous_module(module_name):
                self.errors.append(f"Blocked import from dangerous module: {node.module}")
            elif self.allow_agent_calls_only:
                self.errors.append(f"Imports not allowed in agent action: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.allow_agent_calls_only:
            if not (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "agent"):
                self.errors.append("Only agent.method() calls are allowed in action code")
            else:
                self.found_agent_call = True

        if isinstance(node.func, ast.Name):
            if self._is_dangerous_name(node.func.id):
                self.errors.append(f"Blocked call to dangerous builtin: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if self._is_dangerous_attr(node.func.attr):
                self.errors.append(f"Blocked call to dangerous attribute: {node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._is_dangerous_attr(node.attr):
            parent = node.value
            if isinstance(parent, ast.Name) and self._is_dangerous_module(parent.id):
                self.errors.append(
                    f"Blocked access to dangerous attribute: {parent.id}.{node.attr}"
                )
        self.generic_visit(node)


def validate_code(code: str, allow_agent_calls_only: bool = False) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Syntax error in generated code: {e}")
        raise CodeSafetyError(f"Syntax error: {e}") from e

    validator = _SafetyValidator(allow_agent_calls_only=allow_agent_calls_only)
    validator.visit(tree)

    if validator.errors:
        for error in validator.errors:
            logger.warning(f"SAFETY BLOCK: {error}")
        raise CodeSafetyError(
            f"Code failed safety validation: {'; '.join(validator.errors)}"
        )

    return tree


def _extract_agent_call(tree: ast.Module) -> tuple[str, list[Any], dict[str, Any]]:
    if not tree.body or len(tree.body) != 1:
        raise CodeSafetyError("Action code must contain exactly one agent method call")

    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr):
        raise CodeSafetyError("Action code must be a single expression")

    call = stmt.value
    if not (isinstance(call, ast.Call) and
            isinstance(call.func, ast.Attribute) and
            isinstance(call.func.value, ast.Name) and
            call.func.value.id == "agent"):
        raise CodeSafetyError("Action code must be an agent.method() call")

    method_name = call.func.attr

    args = []
    for arg in call.args:
        try:
            args.append(ast.literal_eval(arg))
        except (ValueError, SyntaxError):
            raise CodeSafetyError(
                f"Arguments to agent.{method_name} must be literal values"
            )

    kwargs = {}
    for kw in call.keywords:
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            raise CodeSafetyError(
                f"Keyword arguments to agent.{method_name} must be literal values"
            )

    return method_name, args, kwargs


def safe_eval_agent_action(code: str, agent: Any) -> Any:
    tree = validate_code(code, allow_agent_calls_only=True)
    method_name, args, kwargs = _extract_agent_call(tree)

    if not hasattr(agent, method_name):
        raise CodeSafetyError(f"Agent has no method: {method_name}")

    method = getattr(agent, method_name)
    if not callable(method):
        raise CodeSafetyError(f"Attribute {method_name} is not callable on agent")

    logger.info(f"Safe agent call: agent.{method_name}({args}, {kwargs})")
    return method(*args, **kwargs)


def safe_exec(code: str, globals_dict: Optional[Dict] = None, locals_dict: Optional[Dict] = None) -> None:
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = {}

    validate_code(code)

    safe_builtins = {}
    for name in dir(builtins_module):
        if name not in {
            "__import__", "eval", "exec", "compile", "globals", "locals",
            "vars", "getattr", "setattr", "delattr", "open", "input", "breakpoint"
        }:
            safe_builtins[name] = getattr(builtins_module, name)

    safe_globals = {
        "__builtins__": safe_builtins,
    }
    safe_globals.update(globals_dict)

    try:
        exec(code, safe_globals, locals_dict)
    except Exception as e:
        logger.error(f"Error during safe code execution: {e}")
        raise
