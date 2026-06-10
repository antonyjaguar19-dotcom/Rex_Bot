"""
Claw Bot — file utilities

atomic_write_json: crash-safe JSON save. Writes to a temp file in the same
directory, then atomically renames over the target. A crash or power cut
mid-write leaves the OLD file intact instead of a half-written (corrupt)
one. Plain `path.write_text(json.dumps(...))` truncates the target first,
so a crash at the wrong moment destroys the only copy — this is the
incremental-save version of that.

Use for every JSON state file the bot must survive a restart with
(pending_state.json, runtime_settings.json, approved_prompts, stats, ...).
"""

import json
import os
from pathlib import Path

__all__ = ["atomic_write_json", "atomic_write_text"]


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to `path` atomically (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def atomic_write_json(path: Path, obj, indent: int = 2,
                      ensure_ascii: bool = False, default=None) -> None:
    """Serialize `obj` to JSON and write it to `path` atomically."""
    atomic_write_text(
        Path(path),
        json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii,
                   default=default),
    )
