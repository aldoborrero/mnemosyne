"""Which provider runs, and what the approved-writes provider may do.

``ingest.mode=turns`` (the default) keeps the Honcho + Hindsight composite,
fed by every conversation turn. ``ingest.mode=approved_writes`` runs
``ApprovedMemoryProvider`` instead: its backends hold only the entries of
Hermes' built-in memory files, which ``memory.write_approval`` gates.

Isolation under approved writes is per Hermes profile. Built-in memory is per
profile and injected into every session of it, so a profile is the smallest
unit whose members can be kept apart; one profile per audience, each with its
own backend credentials, is how rooms with different members are separated.
Unknown values fall back to the stricter option.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from . import config

logger = logging.getLogger(__name__)

INGEST_TURNS = "turns"
INGEST_APPROVED_WRITES = "approved_writes"
APPROVED_BACKENDS = ("openviking", "hindsight")

# agent_context values whose sessions must not write (Hermes sends cron and
# subagent; Honcho also treats flush agents this way).
_NO_WRITE_CONTEXTS = ("cron", "subagent", "flush")


def ingest_mode(home: Optional[Path] = None) -> str:
    mode = str(config.get("ingest", "mode", default=INGEST_TURNS, home=home) or "").strip()
    if mode in (INGEST_TURNS, INGEST_APPROVED_WRITES):
        return mode
    logger.warning("mnemosyne: unknown ingest.mode %r — using %s", mode, INGEST_APPROVED_WRITES)
    return INGEST_APPROVED_WRITES


def approved_writes_only(home: Optional[Path] = None) -> bool:
    return ingest_mode(home) == INGEST_APPROVED_WRITES


@dataclass(frozen=True)
class ApprovedPolicy:
    """Approved-writes settings, read once per provider."""
    backends: Tuple[str, ...]
    rejected_backends: Tuple[str, ...]
    prefetch: bool
    reflect: bool


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def approved_policy(home: Optional[Path] = None) -> ApprovedPolicy:
    raw = config.get("approved", "backends", default=["openviking"], home=home)
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    names = [str(b).strip() for b in (raw or []) if str(b).strip()]
    return ApprovedPolicy(
        backends=tuple(dict.fromkeys(b for b in names if b in APPROVED_BACKENDS)),
        rejected_backends=tuple(b for b in names if b not in APPROVED_BACKENDS),
        prefetch=_as_bool(config.get("approved", "prefetch", default=False, home=home)),
        reflect=_as_bool(config.get("approved", "reflect", default=False, home=home)),
    )


def approved_problem(p: ApprovedPolicy) -> Optional[str]:
    """Why the approved-writes provider cannot run with these settings, or None."""
    if p.rejected_backends:
        return (f"approved.backends lists {', '.join(p.rejected_backends)}; only "
                f"{', '.join(APPROVED_BACKENDS)} can be limited to approved entries")
    if not p.backends:
        return "approved.backends is empty"
    return None


def writes_allowed(agent_context: str) -> bool:
    return (agent_context or "primary") not in _NO_WRITE_CONTEXTS


def namespace(agent_identity: str, hermes_home: str) -> str:
    """Stable, id-safe namespace for one Hermes profile."""
    key = f"{agent_identity or 'default'}:{Path(hermes_home).resolve()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def entry_id(text: str) -> str:
    """Content-derived id of one built-in memory entry."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def is_cloud_hindsight(url: str) -> bool:
    return "hindsight.vectorize.io" in (url or "").lower()


def describe() -> List[str]:
    """Human-readable summary of the approved policy, for `hermes mnemosyne status`."""
    p = approved_policy()
    return [f"ingest.mode:         {ingest_mode()}",
            f"approved.backends:   {', '.join(p.backends) or '-'}",
            f"approved.prefetch:   {p.prefetch}",
            f"approved.reflect:    {p.reflect}"]
