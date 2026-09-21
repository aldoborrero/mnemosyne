"""Hermes' built-in memory files, the source of truth under approved writes.

Hermes writes MEMORY.md (target ``memory``) and USER.md (target ``user``)
under ``<hermes_home>/memories`` only after ``memory.write_approval`` let the
write through — inline, or later through ``/memory approve``, which updates
the files without notifying memory providers. Reading the files is therefore
the one view of approved memory that every approval path reaches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TARGETS = {"memory": "MEMORY.md", "user": "USER.md"}

try:
    from tools.memory_tool_store import ENTRY_DELIMITER
except Exception:  # standalone tests; the value Hermes writes with
    ENTRY_DELIMITER = "\n§\n"


@dataclass(frozen=True)
class TargetEntries:
    """Entries of one target. ``entries is None`` means the file could not be
    read or does not exist, which is not the same as an empty file: only a
    file that was read may drive deletions."""
    target: str
    entries: Optional[List[str]]
    missing: bool = False


def memories_dir(hermes_home: str) -> Path:
    return Path(hermes_home) / "memories"


def read_target(hermes_home: str, target: str) -> TargetEntries:
    path = memories_dir(hermes_home) / TARGETS[target]
    if not path.exists():
        return TargetEntries(target, None, missing=True)
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("mnemosyne: cannot read %s: %s", path, exc)
        return TargetEntries(target, None)
    entries = [e for e in (part.strip() for part in raw.split(ENTRY_DELIMITER)) if e]
    return TargetEntries(target, list(dict.fromkeys(entries)))


def read_all(hermes_home: str) -> Dict[str, TargetEntries]:
    return {target: read_target(hermes_home, target) for target in TARGETS}


def signature(hermes_home: str) -> Tuple:
    """Cheap change detector: (mtime_ns, size) per file, None when absent."""
    sig = []
    for name in TARGETS.values():
        try:
            st = (memories_dir(hermes_home) / name).stat()
            sig.append((st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append(None)
    return tuple(sig)
