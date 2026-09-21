"""Make a backend hold exactly the approved built-in memory entries.

Every entry has a content-derived id (``policy.entry_id``), so a reconcile is
a set difference per target: ids in the file but not in the backend are
stored, ids in the backend but not in the file are deleted. Hermes' own
edits (``replace`` / ``remove`` by ``old_text`` substring) need no special
handling: after the edit the file simply holds a different set.

Deletions are driven only by a file that was actually read. A missing or
unreadable file leaves that target's backend copies untouched, and a backend
whose listing fails is skipped for the round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Set

from .memory_files import TargetEntries
from .policy import entry_id

logger = logging.getLogger(__name__)

# More deletions than this in one round are logged at warning level, so a
# large purge driven by an edited file is visible in the logs.
_LOUD_DELETE_COUNT = 3


class EntryStore(Protocol):
    name: str

    def list_ids(self, target: str) -> Set[str]: ...
    def put(self, target: str, entry: str, content: str) -> None: ...
    def delete(self, target: str, entry: str) -> None: ...


@dataclass
class Outcome:
    created: int = 0
    deleted: int = 0
    kept_on_unreadable: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def reconcile(store: EntryStore, files: Dict[str, TargetEntries]) -> Outcome:
    out = Outcome()
    for target, te in files.items():
        try:
            current = store.list_ids(target)
        except Exception as exc:
            out.errors.append(f"{store.name}/{target}: listing failed: {exc}")
            continue
        if te.entries is None:
            if current:
                out.kept_on_unreadable.append(target)
            continue
        desired = {entry_id(text): text for text in te.entries}
        to_delete = sorted(current - set(desired))
        if len(to_delete) > _LOUD_DELETE_COUNT:
            logger.warning("mnemosyne: %s/%s: deleting %d entries no longer in the approved file",
                           store.name, target, len(to_delete))
        for eid, text in desired.items():
            if eid in current:
                continue
            try:
                store.put(target, eid, text)
                out.created += 1
            except Exception as exc:
                out.errors.append(f"{store.name}/{target}: store {eid} failed: {exc}")
        for eid in to_delete:
            try:
                store.delete(target, eid)
                out.deleted += 1
            except Exception as exc:
                out.errors.append(f"{store.name}/{target}: delete {eid} failed: {exc}")
    for err in out.errors:
        logger.warning("mnemosyne: %s", err)
    return out
