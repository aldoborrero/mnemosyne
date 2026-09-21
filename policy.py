"""Ingest, scope and backend policy.

Three independent settings decide what Mnemosyne may store and where:

* ``ingest.mode`` — ``turns`` feeds every conversation turn to the inner
  providers (their LLM extractors run on raw transcripts). ``approved_writes``
  feeds them only through ``on_memory_write``, which Hermes calls after a
  built-in ``memory`` tool write has committed — i.e. after
  ``memory.write_approval`` let it through.
* ``scope.mode`` — ``none`` keeps one memory per install. ``chat`` keys memory
  by the gateway chat (platform + chat id from Hermes' ``initialize`` kwargs),
  so what is stored in one room is never recalled in another.
* ``backends`` — which inner providers are loaded.

Unknown values fall back to the stricter option.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from . import config

logger = logging.getLogger(__name__)

INGEST_TURNS = "turns"
INGEST_APPROVED_WRITES = "approved_writes"
SCOPE_NONE = "none"
SCOPE_CHAT = "chat"
KNOWN_BACKENDS = ("honcho", "hindsight")

# Backends whose data can be partitioned per chat. Honcho keeps a user model
# across sessions, so chat scope refuses to run with it.
CHAT_SCOPABLE_BACKENDS = ("hindsight",)

# agent_context values for which Hermes itself tells providers to skip writes.
_NO_WRITE_CONTEXTS = ("cron", "subagent")


def ingest_mode() -> str:
    mode = str(config.get("ingest", "mode", default=INGEST_TURNS) or "").strip()
    if mode in (INGEST_TURNS, INGEST_APPROVED_WRITES):
        return mode
    logger.warning("mnemosyne: unknown ingest.mode %r — using %s", mode, INGEST_APPROVED_WRITES)
    return INGEST_APPROVED_WRITES


def approved_writes_only() -> bool:
    return ingest_mode() == INGEST_APPROVED_WRITES


def scope_mode() -> str:
    mode = str(config.get("scope", "mode", default=SCOPE_NONE) or "").strip()
    if mode in (SCOPE_NONE, SCOPE_CHAT):
        return mode
    logger.warning("mnemosyne: unknown scope.mode %r — using %s", mode, SCOPE_CHAT)
    return SCOPE_CHAT


def backends() -> List[str]:
    raw = config.get("backends", default=list(KNOWN_BACKENDS))
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    selected = [b for b in (raw or []) if b in KNOWN_BACKENDS]
    unknown = [b for b in (raw or []) if b and b not in KNOWN_BACKENDS]
    if unknown:
        logger.warning("mnemosyne: ignoring unknown backends %s", unknown)
    return list(dict.fromkeys(selected))


def prefetch_enabled() -> bool:
    return bool(config.get("prefetch", "enabled", default=True))


def writes_allowed(agent_context: str) -> bool:
    return (agent_context or "primary") not in _NO_WRITE_CONTEXTS


def chat_scope_key(platform: str, chat_id: str) -> Optional[str]:
    """Scope key for a gateway chat; None when a part is missing or for the CLI."""
    platform = (platform or "").strip()
    chat_id = (chat_id or "").strip()
    if not platform or not chat_id or platform == "cli":
        return None
    return f"{platform}:{chat_id}"


def scope_slug(scope_key: str) -> str:
    """Stable, filesystem- and bank-id-safe identifier for a scope key."""
    return hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:16]


def scope_dir(scope_key: str) -> Path:
    return config.plugin_dir() / "scopes" / scope_slug(scope_key)


def chat_scope_problem(selected_backends: List[str]) -> Optional[str]:
    """Why chat scope cannot run with these backends, or None if it can."""
    unscopable = [b for b in selected_backends if b not in CHAT_SCOPABLE_BACKENDS]
    if unscopable:
        return f"scope.mode=chat cannot partition {', '.join(unscopable)}; set backends to {list(CHAT_SCOPABLE_BACKENDS)}"
    if not selected_backends:
        return "scope.mode=chat needs at least one backend"
    return None
