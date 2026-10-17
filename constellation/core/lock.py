"""One run per vault at a time.

Two concurrent runs (CLI + UI, or two watch loops) would both read the same
cache, both rewrite notes and then race to save the cache -- the loser's work
is lost and notes can end up with mixed link sections. ``vault_lock`` guards a
run with ``<vault>/.linker.lock`` holding the owner's pid and start time.

A lock is considered stale (and taken over) when its process is gone or it is
older than ``STALE_AFTER_SECONDS`` -- a crashed run must never block the vault
forever.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from constellation.core.log import log

LOCK_FILENAME = ".linker.lock"
STALE_AFTER_SECONDS = 6 * 60 * 60


class VaultLockedError(RuntimeError):
    """Another live Constellation run holds the vault lock."""


def lock_path(vault_path) -> Path:
    return Path(vault_path).expanduser() / LOCK_FILENAME


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _read_lock(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("pid", 0)), float(data.get("timestamp", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def is_stale(path: Path, now: float = None) -> bool:
    info = _read_lock(path)
    if info is None:
        return True  # unreadable / half-written lock file
    pid, timestamp = info
    now = time.time() if now is None else now
    return (now - timestamp) > STALE_AFTER_SECONDS or not _pid_alive(pid)


def _try_create(path: Path) -> bool:
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "timestamp": time.time()}, f)
    return True


@contextmanager
def vault_lock(vault_path):
    """Holds the vault lock for the duration of the block, or raises VaultLockedError."""
    path = lock_path(vault_path)
    if not _try_create(path):
        if not is_stale(path):
            info = _read_lock(path) or (0, 0)
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info[1]))
            raise VaultLockedError(
                f"Another Constellation run (pid {info[0]}, started {started}) is using this vault. "
                f"If that is wrong, delete {path}."
            )
        log(f"Removing stale lock {path}.")
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        if not _try_create(path):
            raise VaultLockedError(f"Another Constellation run grabbed the lock on {path}.")
    try:
        yield path
    finally:
        info = _read_lock(path)
        if info is not None and info[0] == os.getpid():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
