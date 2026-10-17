"""Snapshots of notes taken right before Constellation rewrites them.

Every real write is preceded by a copy of exactly the files about to change,
into ``<vault>/.constellation_backups/<YYYYmmdd-HHMMSS>/`` with their relative
paths preserved. That makes any run undoable with ``restore_backup`` without
asking users to keep their own vault backups. Only the newest ``MAX_BACKUPS``
snapshots are kept so the folder does not grow forever.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from constellation.core import config
from constellation.core.log import log

BACKUP_DIRNAME = ".constellation_backups"
_STAMP_FORMAT = "%Y%m%d-%H%M%S"


def backups_root(vault_path) -> Path:
    return Path(vault_path).expanduser() / BACKUP_DIRNAME


def _relative(vault: Path, note_path) -> Path:
    path = Path(note_path)
    if not path.is_absolute():
        path = vault / path
    resolved = path.resolve()
    try:
        return resolved.relative_to(vault.resolve())
    except ValueError:
        raise ValueError(f"'{note_path}' is outside the vault; refusing to back it up.")


def create_backup(vault_path, note_paths: Iterable) -> Path:
    """Copies note_paths (absolute, or relative to the vault) into a new
    timestamped backup folder and returns that folder."""
    vault = Path(vault_path).expanduser()
    root = backups_root(vault)
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime(_STAMP_FORMAT)
    # Two runs inside the same second (watch mode, tests) must not merge, and
    # the newer one must sort after every existing folder with that stamp --
    # even if pruning already removed the bare-stamp one.
    same_second = [_suffix(p.name) for p in root.iterdir() if p.name.startswith(stamp)]
    target = root / (f"{stamp}-{max(same_second) + 1}" if same_second else stamp)
    target.mkdir()

    for note_path in note_paths:
        rel = _relative(vault, note_path)
        source = vault / rel
        if not source.is_file():
            continue
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    _prune(vault)
    return target


def _backup_dirs(vault: Path) -> List[Path]:
    root = backups_root(vault)
    if not root.is_dir():
        return []
    # Names are timestamps (plus an optional -N suffix), so sorting by
    # (timestamp, suffix) is chronological.
    return sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: (p.name[:15], _suffix(p.name)),
        reverse=True,
    )


def _suffix(name: str) -> int:
    parts = name.split("-")
    return int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0


def _prune(vault: Path) -> None:
    keep = max(int(getattr(config, "MAX_BACKUPS", 10)), 1)
    for old in _backup_dirs(vault)[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def list_backups(vault_path) -> List[dict]:
    """Newest first: [{name, path, created (ISO), file_count}]."""
    result = []
    for d in _backup_dirs(Path(vault_path).expanduser()):
        try:
            created = datetime.strptime(d.name[:15], _STAMP_FORMAT).isoformat()
        except ValueError:
            created = datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")
        result.append({
            "name": d.name,
            "path": str(d),
            "created": created,
            "file_count": sum(1 for f in d.rglob("*") if f.is_file()),
        })
    return result


def restore_backup(vault_path, backup_name: str) -> int:
    """Copies every file of a backup back over the vault. Returns the count.

    Restored notes are also dropped from the cache's file hashes: the cache
    describes the post-run links, and without this the next run would believe
    the restored (older) notes are already up to date.
    """
    vault = Path(vault_path).expanduser()
    name = str(backup_name or "").strip()
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Invalid backup name: {backup_name!r}")
    source_root = backups_root(vault) / name
    if not source_root.is_dir():
        raise FileNotFoundError(f"No backup named '{name}' in {backups_root(vault)}")

    restored_rel = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(source_root)
        dest = vault / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        restored_rel.append(rel.as_posix())

    _forget_hashes(vault, restored_rel)
    log(f"Restored {len(restored_rel)} file(s) from backup '{name}'.")
    return len(restored_rel)


def _forget_hashes(vault: Path, rel_paths: List[str]) -> None:
    # Imported lazily: nodes.io imports core modules at import time.
    from constellation.nodes.io import CACHE_FILENAME

    cache_path = vault / CACHE_FILENAME
    if not cache_path.exists() or not rel_paths:
        return
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    files = cache.get("files") or {}
    names = {Path(p).name for p in rel_paths}
    for title in list(files):
        # Titles are file names, or "parent/name" for duplicate file names.
        if title in names or title in rel_paths or any(p.endswith("/" + title) for p in rel_paths):
            files.pop(title, None)
    cache["files"] = files
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
