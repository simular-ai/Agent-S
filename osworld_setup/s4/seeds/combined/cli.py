"""Shadow interface CLI.

Three subcommands:
  capabilities  — list every public helper with signature + first-line docstring
  doc <name>    — show the full docstring of one helper (e.g. `doc gui.click_at`)
  call <name>   — invoke a helper function from the shell

Helper modules are auto-discovered from `helpers/`. Add docstrings to your
helpers and they surface in `capabilities` automatically — you should not
need to edit this file as you add new helpers.
"""
import argparse
import importlib
import inspect
import os
import sys


def _discover_modules():
    """Return helper module names discovered under helpers/, alphabetical."""
    helpers_dir = os.path.join(os.path.dirname(__file__), "helpers")
    if not os.path.isdir(helpers_dir):
        return []
    out = []
    for entry in sorted(os.listdir(helpers_dir)):
        if entry.endswith(".py") and not entry.startswith("_"):
            out.append(entry[:-3])
    return out


def _public_fns(mod):
    """Yield (name, function) for public functions defined in `mod`."""
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        if obj.__module__ != mod.__name__:
            continue
        yield name, obj


def _signature(fn):
    try:
        return str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(?)"


def cmd_capabilities(args):
    """List every public helper with signature + first-line docstring."""
    print("shadow_interface helpers")
    print("=" * 60)
    for mod_name in _discover_modules():
        try:
            mod = importlib.import_module(
                f".helpers.{mod_name}", package=__package__
            )
        except Exception as e:
            print(f"\n{mod_name}  (failed to import: {e})")
            continue
        print(f"\n{mod_name}")
        print("-" * 60)
        listed = False
        for name, fn in _public_fns(mod):
            listed = True
            doc = inspect.getdoc(fn)
            first = (doc.split("\n", 1)[0] if doc else "").strip()
            line = f"  {name}{_signature(fn)}"
            if first:
                line += f"  — {first}"
            print(line)
        if not listed:
            print("  (no public functions)")
    print()
    print("Use:")
    print("  python3 -m shadow_interface doc <module>.<function>")
    print("  python3 -m shadow_interface call <module>.<function> [args ...]")


def cmd_doc(args):
    """Print the full docstring of one helper function."""
    mod_name, _, fn_name = args.func_path.rpartition(".")
    if not mod_name or not fn_name:
        sys.exit("Usage: doc <module>.<function>")
    try:
        mod = importlib.import_module(
            f".helpers.{mod_name}", package=__package__
        )
    except ImportError as e:
        sys.exit(f"Module 'helpers.{mod_name}' not found: {e}")
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        sys.exit(f"Function '{fn_name}' not found in helpers.{mod_name}")
    print(f"{mod_name}.{fn_name}{_signature(fn)}")
    print()
    doc = inspect.getdoc(fn)
    print(doc if doc else "(no docstring)")


def cmd_call(args):
    """Invoke a helper. Positional args become the function args (as strings)."""
    mod_name, _, fn_name = args.func_path.rpartition(".")
    if not mod_name or not fn_name:
        sys.exit("Usage: call <module>.<function> [args ...]")
    mod = importlib.import_module(f".helpers.{mod_name}", package=__package__)
    fn = getattr(mod, fn_name)
    result = fn(*args.extra)
    if result is not None:
        print(result)


def main():
    parser = argparse.ArgumentParser(prog="shadow_interface")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "capabilities",
        help="List all helpers with signatures + 1-line docs",
    )
    doc_p = sub.add_parser("doc", help="Show full docstring of one helper")
    doc_p.add_argument("func_path", help="module.function")
    call_p = sub.add_parser("call", help="Call a helper function")
    call_p.add_argument("func_path", help="module.function")
    call_p.add_argument("extra", nargs="*")
    args = parser.parse_args()
    if args.command == "capabilities":
        cmd_capabilities(args)
    elif args.command == "doc":
        cmd_doc(args)
    elif args.command == "call":
        cmd_call(args)
    else:
        parser.print_help()
