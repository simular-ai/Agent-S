"""Unprivileged Landlock filesystem sandbox for OSWorld V2 Builders."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

_CREATE_RULESET = 444
_ADD_RULE = 445
_RESTRICT_SELF = 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_EXECUTE = 1 << 0
_WRITE_FILE = 1 << 1
_READ_FILE = 1 << 2
_READ_DIR = 1 << 3
_REMOVE_DIR = 1 << 4
_REMOVE_FILE = 1 << 5
_MAKE_CHAR = 1 << 6
_MAKE_DIR = 1 << 7
_MAKE_REG = 1 << 8
_MAKE_SOCK = 1 << 9
_MAKE_FIFO = 1 << 10
_MAKE_BLOCK = 1 << 11
_MAKE_SYM = 1 << 12
_REFER = 1 << 13
_TRUNCATE = 1 << 14


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def _syscall(number: int, *args) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(number, *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def landlock_abi() -> int:
    """Return the host Landlock ABI, or zero when unsupported/disabled."""
    if sys.platform != "linux":
        return 0
    try:
        return _syscall(
            _CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint32(_CREATE_RULESET_VERSION),
        )
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            return 0
        raise


def _handled_access(abi: int) -> int:
    access = (1 << 13) - 1
    if abi >= 2:
        access |= _REFER
    if abi >= 3:
        access |= _TRUNCATE
    return access


def _add_path_rule(ruleset_fd: int, path: Path, access: int, handled: int) -> None:
    if not path.exists():
        return
    path = path.resolve()
    is_dir = path.is_dir()
    if is_dir:
        allowed = access
    else:
        # Directory mutation and READ_DIR rights cannot be granted to files.
        directory_only = (
            _READ_DIR
            | _REMOVE_DIR
            | _REMOVE_FILE
            | _MAKE_CHAR
            | _MAKE_DIR
            | _MAKE_REG
            | _MAKE_SOCK
            | _MAKE_FIFO
            | _MAKE_BLOCK
            | _MAKE_SYM
            | _REFER
        )
        allowed = access & ~directory_only
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _PathBeneathAttr(allowed & handled, fd)
        _syscall(
            _ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint32(0),
        )
    finally:
        os.close(fd)


def apply_builder_landlock(
    *,
    workspace: Path,
    executable_paths: list[Path],
    read_paths: list[Path] | None = None,
) -> None:
    """Allow only the Builder workspace plus minimal runtime/config paths.

    ``read_paths`` grants read+execute on additional roots an agent runtime
    needs but cannot reach from its own install dir (e.g. a pinned source
    tree with its own venv's site-packages).
    """
    abi = landlock_abi()
    if abi == 0:
        raise RuntimeError("Landlock is unavailable; refusing unsandboxed V2 Builder")
    handled = _handled_access(abi)
    attr = _RulesetAttr(handled)
    ruleset_fd = _syscall(
        _CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        ctypes.c_uint32(0),
    )
    read_only = _EXECUTE | _READ_FILE | _READ_DIR
    read_write = handled
    try:
        for path in (
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/etc"),
            Path("/proc"),
            Path("/sys"),
        ):
            _add_path_rule(ruleset_fd, path, read_only, handled)
        # /etc/resolv.conf commonly points into /run/systemd/resolve. Landlock
        # follows inode rules rather than treating /etc's symlink as authority
        # over its external target; without this file, Builders have network
        # access but every model request fails at DNS resolution.
        _add_path_rule(
            ruleset_fd,
            Path("/etc/resolv.conf").resolve(),
            read_only,
            handled,
        )
        # The run's locked interpreter/dependencies are read-only. The
        # standalone scratch bridge is refreshed inside ``workspace`` before
        # launch, so no parent-project source tree needs to be exposed.
        _add_path_rule(ruleset_fd, Path(sys.prefix), read_only, handled)
        for path in (Path(f"/run/user/{os.getuid()}"),):
            _add_path_rule(ruleset_fd, path, read_write, handled)
        for path in (
            Path("/dev/null"),
            Path("/dev/zero"),
            Path("/dev/random"),
            Path("/dev/urandom"),
            Path("/dev/tty"),
        ):
            _add_path_rule(ruleset_fd, path, _READ_FILE | _WRITE_FILE, handled)

        home = Path.home()
        for relative in (
            ".claude",
            ".claude.json",
            ".qwen",
            ".gitconfig",
            ".config/claude",
            ".config/qwen",
            ".cache/claude",
            ".cache/qwen",
            ".local/state/claude",
            ".local/state/qwen",
        ):
            _add_path_rule(ruleset_fd, home / relative, read_write, handled)
        for extra in read_paths or []:
            _add_path_rule(ruleset_fd, extra, read_only, handled)
        for executable in executable_paths:
            _add_path_rule(ruleset_fd, executable, read_only, handled)
            _add_path_rule(ruleset_fd, executable.resolve(), read_only, handled)
            install_dir = executable.resolve().parent
            # JS/native CLIs often load sibling runtime files. Grant only the
            # installation directory, never a parent that contains the V2
            # workspace or its gated sibling repositories.
            workspace_path = workspace.resolve()
            if install_dir not in {workspace_path, *workspace_path.parents}:
                _add_path_rule(ruleset_fd, install_dir, read_only, handled)
        workspace = workspace.resolve()
        # The agent may inspect the whole island, but only the evolved inner
        # repository, notebook, and private temp directory are mutable.
        _add_path_rule(ruleset_fd, workspace, read_only, handled)
        _add_path_rule(ruleset_fd, workspace / "shadow_interface", read_write, handled)
        _add_path_rule(ruleset_fd, workspace / "notebook.jsonl", read_write, handled)
        _add_path_rule(ruleset_fd, workspace / ".builder_tmp", read_write, handled)

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        _syscall(_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0))
    finally:
        os.close(ruleset_fd)


def sandboxed_builder_argv(
    argv: list[str], clone_dir: Path, read_paths: list[Path] | None = None
) -> list[str]:
    """Wrap one command with this module while preserving the original argv."""
    resolved = shutil.which(argv[0])
    if resolved is None and Path(argv[0]).is_file():
        resolved = str(Path(argv[0]).resolve())
    if resolved is None:
        # Preserve the harness's normal not-found result rather than failing
        # with a misleading sandbox diagnostic.
        return argv
    extra: list[str] = []
    for path in read_paths or []:
        extra += ["--read-path", str(Path(path).resolve())]
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--workspace",
        str(Path(clone_dir).resolve()),
        "--executable",
        resolved,
        *extra,
        "--",
        *argv,
    ]


# sockaddr_un.sun_path holds 108 bytes including the terminator, so a socket
# pathname may be at most 107 characters. Prime Agent appends its own names to
# TMPDIR, the longest observed being
# "/prime-agent-<uid>/worker-<12 hex>-<12 hex>.sock" (55 characters for a
# four-digit uid). Past this budget bind() truncates silently and Prime then
# lstat()s the untruncated name and dies with ENOENT, which surfaces as a
# Builder exit far from its cause.
_SOCKET_PATHNAME_MAX = 107
_PRIME_SOCKET_SUFFIX_BUDGET = 55
MAX_TMPDIR_LEN = _SOCKET_PATHNAME_MAX - _PRIME_SOCKET_SUFFIX_BUDGET


def _alias_parent_candidates() -> list[tuple[Path, bool]]:
    """Directories that could hold the alias, most preferred first.

    The second element says whether the directory is shared with other users
    and therefore needs a private, ownership-checked parent of our own before
    a symlink inside it can be trusted.
    """
    candidates: list[tuple[Path, bool]] = []
    for value in (os.environ.get("XDG_RUNTIME_DIR"), f"/run/user/{os.getuid()}"):
        if value:
            candidates.append((Path(value), False))
    # Hosts without a systemd user session have no /run/user/<uid> at all, and
    # the fallback has to be short or it defeats the purpose.
    candidates.append(
        (Path(tempfile.gettempdir()) / f"shadow-builder-{os.getuid()}", True)
    )
    return candidates


def _usable_alias_parent(directory: Path, needs_private_parent: bool) -> bool:
    """Whether we own *directory* outright, creating it when it is ours to make."""
    try:
        if needs_private_parent:
            directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
    except OSError:
        return False
    # A symlink or another user's directory could redirect the Builder's temp
    # files, so require a real directory owned by us.
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and not (needs_private_parent and info.st_mode & 0o077)
    )


def _short_tmp_alias(workspace: Path, private_tmp: Path) -> Path:
    """Return a short pathname for the private temp inode when possible.

    Prime Agent appends daemon/worker socket names to TMPDIR. Deep evolution
    checkout paths can exceed Linux's Unix-socket pathname limit even though
    the underlying directory is valid. A symlink in a short directory we own
    keeps the socket pathname inside that limit while resolving to the same
    Landlock-confined ``.builder_tmp`` directory. The alias is created before
    the ruleset is applied, and Landlock authorizes the resolved target rather
    than the symlink, so the alias needs no rule of its own.
    """
    digest = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:12]
    for parent, needs_private_parent in _alias_parent_candidates():
        if not _usable_alias_parent(parent, needs_private_parent):
            continue
        # A parent we created already carries the name, so the alias there is
        # just the digest: every character counts against the socket budget.
        alias = parent / (
            digest if needs_private_parent else f"shadow-builder-{digest}"
        )
        if len(str(alias)) > MAX_TMPDIR_LEN:
            continue
        try:
            if alias.is_symlink():
                if alias.resolve(strict=False) != private_tmp.resolve():
                    alias.unlink()
                    alias.symlink_to(private_tmp, target_is_directory=True)
            elif not alias.exists():
                alias.symlink_to(private_tmp, target_is_directory=True)
            else:
                continue
        except OSError:
            continue
        return alias
    return private_tmp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--executable", type=Path, action="append", default=[])
    parser.add_argument(
        "--read-path", type=Path, action="append", default=[], dest="read_paths"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("missing command after --")
    private_tmp = args.workspace.resolve() / ".builder_tmp"
    private_tmp.mkdir(mode=0o700, exist_ok=True)
    temp_path = _short_tmp_alias(args.workspace, private_tmp)
    os.environ["TMPDIR"] = str(temp_path)
    os.environ["TMP"] = str(temp_path)
    os.environ["TEMP"] = str(temp_path)
    apply_builder_landlock(
        workspace=args.workspace,
        executable_paths=args.executable,
        read_paths=args.read_paths,
    )
    # Prepend rather than replace: an agent runtime outside the workspace
    # puts its source on PYTHONPATH, and dropping it
    # breaks the import before the sandbox is ever exercised. This does not
    # widen access -- every entry still has to pass the ruleset above to be
    # readable.
    inherited = os.environ.get("PYTHONPATH", "")
    workspace_path = str(args.workspace.resolve())
    os.environ["PYTHONPATH"] = (
        workspace_path + os.pathsep + inherited if inherited else workspace_path
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.pop("PYTHONHOME", None)
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
