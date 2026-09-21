"""Mnemosyne — composite memory provider for Hermes.

Wraps HonchoMemoryProvider (user model) and HindsightMemoryProvider
(facts) via composition. Adds:

  * Plan item 1 — date tags on every retained fact
  * Plan item 2 — repetition counter (FactStore SQLite)
  * Plan item 3 — pre-write dedup against semantic neighbours
  * Plan item 5 — anchor_card.md always-pinned facts
  * Plan item 6 — sectioned 3-block prefetch + conflict-aware labeling
  * Plan item 7 — recovery from session transcripts on startup
  * Plan item 8 — bulk import (via CLI)
  * Plan item 10 — bridge from built-in memory (USER.md/MEMORY.md)
                  to Hindsight with mention_count=10 forced strong signal
  * Plan item 11 — explicit forgetting via memory_forget tool & filter

See README.md for the full design and roadmap.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Parent-package + submodule bootstrap.
#
# The hermes plugin loader registers user-installed plugins under the synthetic
# package `_hermes_user_memory.<name>`, but it does NOT register the parent
# `_hermes_user_memory` itself. It also pre-registers each submodule in
# sys.modules and then exec_module()s them — so any submodule using
# `from . import …` blows up with ModuleNotFoundError on the parent. The
# loader logs that exception at DEBUG and continues with empty submodule stubs
# in sys.modules, which then poison every relative import we make from this
# file. The fix below: synthesise the parent packages and force-reload our
# submodules in dependency order so each `from . import …` here picks up
# fully-initialised module objects.
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util
import os as _os
import sys as _sys
import types as _types

_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PARENT_PKG = "_hermes_user_memory"
_FULL_PKG = f"{_PARENT_PKG}.mnemosyne"

for _pkg_name, _pkg_paths in (
    (_PARENT_PKG, [_os.path.dirname(_PLUGIN_DIR)]),
    (_FULL_PKG, [_PLUGIN_DIR]),
):
    if _pkg_name not in _sys.modules:
        _ns = _types.ModuleType(_pkg_name)
        _ns.__path__ = _pkg_paths
        _sys.modules[_pkg_name] = _ns


def _mnemosyne_force_reload(submodule_name: str):
    full = f"{_FULL_PKG}.{submodule_name}"
    fpath = _os.path.join(_PLUGIN_DIR, f"{submodule_name}.py")
    if not _os.path.exists(fpath):
        return None
    spec = _importlib_util.spec_from_file_location(full, fpath)
    if spec is None or spec.loader is None:
        return None
    mod = _importlib_util.module_from_spec(spec)
    _sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


for _sub in ("config", "policy", "openviking_store", "conflict", "fact_store", "forget", "recovery", "importer", "dedup"):
    try:
        _mnemosyne_force_reload(_sub)
    except Exception as _exc:  # pragma: no cover
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "mnemosyne: failed to load submodule %s: %s", _sub, _exc
        )

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from . import config, policy
from .conflict import is_contradiction, label_pair
from .fact_store import FactStore, today_iso
from .forget import (
    MEMORY_FORGET_SCHEMA,
    forget_by_query,
    is_forgotten as _is_forgotten,
    _write_tombstone,
)
from .openviking_store import OpenVikingStore
from .openviking_store import is_configured as _openviking_configured
from .recovery import initialize_cursor_if_missing, replay_missed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background tombstone writer.
#
# `forget.py` calls into us here via late-import so the heavy executor lives
# on the provider but the forget logic stays self-contained. One pool keeps
# tombstone backlog from blocking other writes (it has its own worker, not
# stolen from the prefetch/recall pool).
# ---------------------------------------------------------------------------

_tombstone_executor: Optional[ThreadPoolExecutor] = None
_tombstone_lock = threading.Lock()


def _get_tombstone_executor() -> ThreadPoolExecutor:
    global _tombstone_executor
    with _tombstone_lock:
        if _tombstone_executor is None:
            _tombstone_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mnemosyne-tombstone"
            )
        return _tombstone_executor


def _spawn_tombstone_writer(hindsight_provider, texts, when: str) -> None:
    """Submit one job that writes all tombstones sequentially. Sequential
    (not parallel) on purpose: each retain hits the same Hindsight
    embedding pipeline, parallelising would just queue inside Hindsight."""
    if not hindsight_provider or not texts:
        return
    executor = _get_tombstone_executor()

    def _runner():
        ok = 0
        for text in texts:
            if _write_tombstone(hindsight_provider, text, when):
                ok += 1
        logger.info(
            "mnemosyne: tombstones written %d/%d (when=%s)",
            ok, len(texts), when,
        )

    try:
        executor.submit(_runner)
    except Exception as exc:
        logger.debug("mnemosyne: tombstone executor submit failed: %s", exc)


# ---------------------------------------------------------------------------
# Curated tool schemas (plan item 6 — tightened role descriptions)
# ---------------------------------------------------------------------------

_PROFILE_SCHEMA = {
    "name": "memory_profile",
    "description": (
        "ONLY for the user's profile card: name, role, communication style, "
        "stable preferences. Read or update. Do NOT use for general facts or "
        "past-conversation history — for those use memory_recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "card": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New card as a list of fact strings. Omit to read.",
            },
        },
        "required": [],
    },
}

_REASONING_SCHEMA = {
    "name": "memory_reasoning",
    "description": (
        "ONLY questions about the user as a person: their style, habits, "
        "behavioral patterns, what approach works best with them. NOT for "
        "general knowledge and NOT for facts from past conversations — for "
        "those use memory_recall or memory_reflect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language question about the user as a person.",
            },
            "reasoning_level": {
                "type": "string",
                "enum": ["minimal", "low", "medium", "high", "max"],
                "description": "Depth control. Omit for default (low).",
            },
        },
        "required": ["query"],
    },
}

_CONCLUDE_SCHEMA = {
    "name": "memory_conclude",
    "description": (
        "Record a stable user-related conclusion (preference, habit, style). "
        "NOT for technical facts or events — those go through memory_recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The conclusion to persist."},
        },
        "required": ["conclusion"],
    },
}

_RECALL_SCHEMA = {
    "name": "memory_recall",
    "description": (
        "FIRST CHOICE for 'do you remember when we did X?', 'we discussed this', "
        "'how did we fix that before'. Multi-strategy search (semantic + entity "
        "graph) over all past conversations. Returns relevant facts and "
        "fragments. THIS IS THE MAIN LONG-TERM MEMORY TOOL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "max_tokens": {
                "type": "integer",
                "description": "Token budget (default 800, max 4096).",
            },
        },
        "required": ["query"],
    },
}

_READ_SCHEMA = {
    "name": "memory_read",
    "description": (
        "Read the full text of one memory file by the viking:// URI that "
        "memory_recall returned. Only URIs from this conversation's memory "
        "are readable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {"type": "string", "description": "viking:// URI from memory_recall."},
        },
        "required": ["uri"],
    },
}

_REFLECT_SCHEMA = {
    "name": "memory_reflect",
    "description": (
        "LLM synthesis across past-conversation facts. Use when you need a "
        "summary spanning multiple sources ('what did we conclude about X?', "
        "'what facts do we have on topic Y?'). NOT for questions about the "
        "user as a person."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language question."},
        },
        "required": ["query"],
    },
}


# Maps curated tool names → (inner_provider_attr, inner_tool_name).
_TOOL_DISPATCH = {
    "memory_profile":   ("honcho",    "honcho_profile"),
    "memory_reasoning": ("honcho",    "honcho_reasoning"),
    "memory_conclude":  ("honcho",    "honcho_conclude"),
    "memory_recall":    ("hindsight", "hindsight_recall"),
    "memory_reflect":   ("hindsight", "hindsight_reflect"),
    "memory_read":      ("openviking", None),
}


def _truncate_to_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.5:
        cut = cut[:last_nl]
    return cut + "\n…[truncated]"


# Hindsight enforces "Query too long: N tokens exceeds maximum of 500".
# 1500 chars ≈ 350-450 tokens (RU runs higher chars/token than EN); leaves
# headroom for any query expansion Hindsight does internally.
_RECALL_QUERY_MAX_CHARS = 1500


def _truncate_recall_query(query: str) -> str:
    """Trim query so it never trips Hindsight's 500-token recall limit.
    Prefers to cut at a word boundary near the end."""
    if not query:
        return query
    if len(query) <= _RECALL_QUERY_MAX_CHARS:
        return query
    cut = query[:_RECALL_QUERY_MAX_CHARS]
    last_space = cut.rfind(" ")
    if last_space > _RECALL_QUERY_MAX_CHARS * 0.7:
        cut = cut[:last_space]
    return cut


class MnemosyneMemoryProvider(MemoryProvider):
    """Composite provider — Honcho for user model, Hindsight for facts."""

    def __init__(self) -> None:
        self._honcho: Optional[MemoryProvider] = None
        self._hindsight: Optional[MemoryProvider] = None
        # Connected in initialize(), once the scope is known.
        self._openviking: Optional[OpenVikingStore] = None
        # 4 workers: 2 for write fan-out (sync_turn), 2 spare for parallel
        # tool calls so agent-driven recall isn't queued behind background
        # retain jobs.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mnemosyne")
        self._fact_store: Optional[FactStore] = None
        self._init_lock = threading.Lock()
        self._initialized = False
        # Last prefetch payload — stripped from assistant_content before
        # we hand the turn to Hindsight's LLM extractor. Without this the
        # extractor re-extracts whatever we just put into the prompt,
        # producing endless paraphrases of the same fact (the "Barsik
        # loop"). Updated atomically; no lock needed for last-write-wins
        # semantics.
        self._last_prefetch: str = ""
        # Per-turn caches to avoid re-reading anchor_card from disk and
        # re-fetching the Honcho peer card every prefetch. Anchor is keyed by
        # file mtime so manual edits are picked up immediately; peer card
        # uses a short TTL and is invalidated on session-switch / write.
        self._anchor_cache: Optional[tuple] = None  # (mtime, rendered_text)
        self._peer_cache: Optional[tuple] = None    # (expires_at_ts, text)
        self._peer_cache_ttl_s: float = float(
            config.get("prefetch", "peer_card_ttl_s", default=60.0)
        )
        self._cache_lock = threading.Lock()
        # Set in initialize(). While _disabled_reason is not None the provider
        # neither recalls nor writes (fail closed on a scope it cannot apply).
        self._scope_key: Optional[str] = None
        self._disabled_reason: Optional[str] = None
        self._writes_allowed = True
        self._policy = policy.snapshot()
        self._backends = list(self._policy.backends)
        # One worker: approved writes reach the backends in the order Hermes made them.
        self._write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mnemosyne-write")
        self._load_inner_providers()

    def _inject_hindsight_routing_env(self) -> None:
        """Set Hindsight embedding/reranker routing in os.environ so the
        embedded daemon (which inherits parent environ) picks it up.

        Routes:
          * Embeddings → local omlx Jina v5 multilingual via OpenAI-compatible
            API on :8000 (fast, runs on this Mac).
          * Reranker → cloud rerank via litellm on :4000 (alias model name
            `rerank`; the actual upstream is configured in litellm).

        Values are read from mnemosyne config.json under "hindsight_env"
        with hard-coded defaults so a fresh install Just Works.
        """
        import os as _o

        defaults = {
            "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "openai",
            "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": "sk-local-litellm",
            "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL": "http://localhost:8000/v1",
            "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL":
                "jina-embeddings-v5-text-small-retrieval-mlx",
            "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
            "HINDSIGHT_API_RERANKER_COHERE_API_KEY": "sk-local-litellm",
            "HINDSIGHT_API_RERANKER_COHERE_BASE_URL": "http://localhost:4000/v1/rerank",
            "HINDSIGHT_API_RERANKER_COHERE_MODEL": "rerank",
        }
        overrides = config.get("hindsight_env", default={}) or {}
        for k, v in defaults.items():
            # Don't clobber if the user has set something themselves at the
            # shell level — they win over our defaults.
            if k in _o.environ and _o.environ[k]:
                continue
            value = overrides.get(k, v)
            if value:
                _o.environ[k] = str(value)

    def _load_inner_providers(self) -> None:
        if "honcho" in self._backends:
            try:
                from plugins.memory.honcho import HonchoMemoryProvider
                self._honcho = HonchoMemoryProvider()
            except Exception as exc:
                logger.warning("mnemosyne: failed to load Honcho inner provider: %s", exc)

        if "hindsight" in self._backends:
            try:
                from plugins.memory.hindsight import HindsightMemoryProvider
                self._hindsight = HindsightMemoryProvider()
            except Exception as exc:
                logger.warning("mnemosyne: failed to load Hindsight inner provider: %s", exc)

    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        if not self._backends:
            return False
        inner = {"honcho": self._honcho, "hindsight": self._hindsight}
        for backend in self._backends:
            if backend == "openviking":
                if not _openviking_configured():
                    return False
            elif inner[backend] is None or not inner[backend].is_available():
                return False
        return True

    def _disable(self, reason: str) -> None:
        self._disabled_reason = reason
        logger.warning("mnemosyne: memory disabled for this session — %s", reason)

    @property
    def _blocked(self) -> bool:
        return self._disabled_reason is not None

    def _may_ingest_turns(self) -> bool:
        return not self._blocked and self._writes_allowed and not self._policy.approved_writes_only

    def _apply_chat_scope(self, kwargs: Dict[str, Any]) -> bool:
        """Resolve the chat scope and point Hindsight at its bank.

        The scope comes from Hermes' gateway identity kwargs, never from a tool
        argument. Returns False (and disables the provider) when it cannot be
        applied."""
        problem = policy.chat_scope_problem(self._backends)
        if problem:
            self._disable(problem)
            return False
        scope_key = policy.chat_scope_key(str(kwargs.get("platform") or ""),
                                          str(kwargs.get("chat_id") or ""))
        if scope_key is None:
            self._disable("scope.mode=chat but this session has no gateway chat id")
            return False
        self._scope_key = scope_key
        return True

    def _scope_hindsight_bank(self) -> bool:
        """Suffix Hindsight's resolved bank with the scope slug.

        Hindsight's bank_id_template has no chat placeholder, so the bank is
        re-pointed after its initialize() resolved the configured one."""
        if self._scope_key is None or self._hindsight is None:
            return True
        base = getattr(self._hindsight, "_bank_id", None)
        if not isinstance(base, str) or not base:
            self._disable("cannot read Hindsight's bank id to scope it")
            return False
        scoped = f"{base}-{policy.scope_slug(self._scope_key)}"
        self._hindsight._bank_id = scoped
        if getattr(self._hindsight, "_bank_id", None) != scoped:
            self._disable("cannot set Hindsight's scoped bank id")
            return False
        logger.info("mnemosyne: scoped to %s (bank %s)", self._scope_key, scoped)
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        with self._init_lock:
            self._writes_allowed = policy.writes_allowed(str(kwargs.get("agent_context") or ""))
            problem = (policy.ingest_problem(self._backends)
                       if self._policy.approved_writes_only else None)
            if problem:
                self._disable(problem)
                self._initialized = True
                return
            if self._policy.scope_mode == policy.SCOPE_CHAT and not self._apply_chat_scope(kwargs):
                self._initialized = True
                return

            # The Hindsight embedded daemon inherits the parent process's
            # os.environ at start time (daemon_embed_manager.py:331). Inject
            # our Jina/Qwen3 routing here so the daemon picks them up via
            # env without needing to patch the hermes-shipped hindsight
            # plugin or rely on the auto-managed profile.env (which gets
            # re-materialised on every start with only LLM keys).
            self._inject_hindsight_routing_env()

            if self._honcho:
                try:
                    self._honcho.initialize(session_id, **kwargs)
                except Exception as exc:
                    logger.warning("mnemosyne: Honcho initialize failed: %s", exc)
            if self._hindsight:
                try:
                    self._hindsight.initialize(session_id, **kwargs)
                except Exception as exc:
                    logger.warning("mnemosyne: Hindsight initialize failed: %s", exc)
            if not self._scope_hindsight_bank():
                self._initialized = True
                return
            if "openviking" in self._backends:
                slug = policy.scope_slug(self._scope_key) if self._scope_key else None
                self._openviking = OpenVikingStore.connect(slug)
                if self._openviking is None:
                    logger.warning("mnemosyne: OpenViking backend unavailable this session")

            try:
                db_path = (policy.scope_dir(self._scope_key) / "fact_store.db"
                           if self._scope_key else None)
                self._fact_store = FactStore(db_path=db_path)
                # One-shot vacuum: bound the signatures table so it
                # never silently grows past the configured ceiling.
                try:
                    max_sigs = int(config.get("forget", "max_signatures",
                                              default=1000))
                    stale_days = config.get("forget", "signature_stale_days",
                                            default=365)
                    stale = int(stale_days) if stale_days else None
                    removed = self._fact_store.vacuum_signatures(
                        max_count=max_sigs, stale_days=stale,
                    )
                    if removed:
                        logger.info("mnemosyne: vacuumed %d forget signature(s)",
                                    removed)
                except Exception as exc:
                    logger.debug("mnemosyne: signature vacuum skipped: %s", exc)
            except Exception as exc:
                logger.warning("mnemosyne: FactStore init failed: %s", exc)
                self._fact_store = None

            # Plan item 7 — recovery from session transcripts.
            #
            # Stamping the cursor is instant, so it stays inline. The replay
            # itself is not: every pair is an embedding + reranker round trip,
            # so a backlog could hold `_init_lock` — and with it the whole
            # agent startup — for minutes. It runs on its own daemon thread
            # instead, where it also can't starve the prefetch pool.
            # Recovery replays every session transcript on disk: raw turns, from
            # every chat. Never under approved writes or chat scope.
            if (self._policy.approved_writes_only or self._scope_key is not None
                    or not self._writes_allowed):
                self._initialized = True
                return
            try:
                if initialize_cursor_if_missing():
                    logger.info("mnemosyne: recovery cursor stamped at current state")
                else:
                    self._spawn_recovery()
            except Exception as exc:
                logger.debug("mnemosyne: recovery skipped: %s", exc)

            self._initialized = True

    def _spawn_recovery(self) -> None:
        """Replay missed transcript turns in the background."""
        hindsight = self._hindsight
        if hindsight is None:
            return

        def _runner() -> None:
            try:
                summary = replay_missed(hindsight, max_pairs=50)
                if summary.get("replayed", 0):
                    logger.info("mnemosyne: recovery replayed %d turn pair(s)",
                                summary["replayed"])
                for reason in ("stopped_at_failure", "stopped_at_deadline",
                               "stopped_at_limit"):
                    if summary.get(reason):
                        logger.info("mnemosyne: recovery %s — resumes next startup",
                                    reason)
                        break
            except Exception as exc:
                logger.debug("mnemosyne: recovery failed: %s", exc)

        threading.Thread(
            target=_runner, name="mnemosyne-recovery", daemon=True
        ).start()

    def shutdown(self) -> None:
        # Queued writes go to the inner providers, so drain them first.
        self.flush_writes()
        self._write_executor.shutdown(wait=False)
        if self._honcho:
            try:
                self._honcho.shutdown()
            except Exception as exc:
                logger.debug("mnemosyne: Honcho shutdown failed: %s", exc)
        if self._hindsight:
            try:
                self._hindsight.shutdown()
            except Exception as exc:
                logger.debug("mnemosyne: Hindsight shutdown failed: %s", exc)
        if self._openviking is not None:
            self._openviking.close()
        self._executor.shutdown(wait=False)

    _TOOL_HINTS = {
        "memory_recall": "`memory_recall(query)`        — FIRST CHOICE for 'do you "
                         "remember…', 'we discussed…', 'how did we fix…'.",
        "memory_reflect": "`memory_reflect(query)`       — synthesised summary across "
                          "multiple past facts ('what did we conclude about X?').",
        "memory_profile": "`memory_profile(card?)`       — read or update the user's "
                          "profile card (stable preferences, role, communication style).",
        "memory_reasoning": "`memory_reasoning(query)`     — questions about the user "
                            "**as a person** (style, habits). Slow — use sparingly.",
        "memory_conclude": "`memory_conclude(conclusion)` — record a stable fact about "
                           "the user.",
        "memory_read": "`memory_read(uri)`            — full text of a memory file "
                       "whose viking:// URI memory_recall returned.",
        "memory_forget": "`memory_forget(query)`        — TWO STEPS. First call with "
                         "just the query returns a preview list of candidates. Show that "
                         "list to the user verbatim, get explicit confirmation, then "
                         "re-invoke with `confirmed=true` (or `indices=[1,3]` to pick a "
                         "subset). NEVER call with `confirmed=true` on the first try.",
    }

    def system_prompt_block(self) -> str:
        """Tell the agent about Mnemosyne's curated tool surface.

        Crucially we DO NOT delegate to ``self._honcho.system_prompt_block()``
        or ``self._hindsight.system_prompt_block()`` — those describe their
        native tool names (``honcho_*`` / ``hindsight_*``) which we
        deliberately hide behind our curated tools. Letting them through
        would tell the LLM that ``honcho_search`` etc. exist when in fact
        only the curated set is callable, leading to phantom tool calls and
        confused tool selection."""
        names = [schema["name"] for schema in self.get_tool_schemas()]
        if not names:
            return ""
        lines = [
            "# Memory (Mnemosyne)",
            "You have a long-term memory system. It is accessed only through "
            "the tools below — DO NOT call any tool name that starts with "
            "`honcho_` or `hindsight_`; those are not exposed.",
            "",
            "When to use which:",
        ]
        lines += [f"- {self._TOOL_HINTS[n]}" for n in names if n in self._TOOL_HINTS]
        if self._policy.approved_writes_only:
            lines += ["", "To store something new, use the built-in `memory` tool; "
                          "its writes are the only ones kept."]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Prefetch (plan item 6/7 fusion: anchor → peer card → Hindsight recall)
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._blocked or not self._policy.prefetch_enabled:
            return ""
        max_total = int(config.get("prefetch", "max_total_tokens", default=4500))
        anchor_budget = int(config.get("prefetch", "anchor_token_budget", default=200))
        peer_card_budget = int(config.get("prefetch", "honcho_card_token_budget", default=200))
        hindsight_budget = int(config.get("prefetch", "hindsight_token_budget", default=4096))
        per_branch_timeout = float(
            config.get("prefetch", "parallel_timeout_s", default=12.0)
        )

        # Fan out the three independent fetches in parallel via the existing
        # executor. They are independent reads (anchor=disk, peer=Honcho,
        # hindsight=HTTP) — the prior serial layout was dominated by
        # _fetch_hindsight_recall (~6-8s). Concurrent execution caps total
        # time at the slowest branch plus a few ms of overhead.
        anchor_fut = self._executor.submit(self._read_anchor_card)
        peer_fut = self._executor.submit(self._fetch_honcho_peer_card)
        hindsight_fut = self._executor.submit(
            self._fetch_hindsight_recall, query, hindsight_budget,
        )
        viking_fut = self._executor.submit(self._openviking_recall_text, query)

        def _wait(fut, default=""):
            try:
                return fut.result(timeout=per_branch_timeout) or default
            except Exception as exc:
                logger.debug("mnemosyne: prefetch branch failed/timeout: %s", exc)
                return default

        anchor_text = _wait(anchor_fut)
        peer_card_text = _wait(peer_fut)
        hindsight_text = "\n".join(t for t in (_wait(viking_fut), _wait(hindsight_fut)) if t)

        sections: List[str] = []

        if anchor_text:
            sections.append("# Pinned (anchor card)\n" +
                            _truncate_to_chars(anchor_text, anchor_budget * 4))

        if peer_card_text:
            sections.append("# User profile\n" +
                            _truncate_to_chars(peer_card_text, peer_card_budget * 4))

        if hindsight_text:
            hindsight_text = self._filter_forgotten(hindsight_text)
            if hindsight_text:
                sections.append("# Facts (relevant)\n" +
                                _truncate_to_chars(hindsight_text, hindsight_budget * 4))

        if len(sections) >= 3:
            sections = self._apply_conflict_resolver(sections)

        result = "\n\n".join(sections)
        capped = _truncate_to_chars(result, max_total * 4)
        self._last_prefetch = capped
        return capped

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._blocked or not self._policy.prefetch_enabled:
            return
        if self._honcho:
            try:
                self._honcho.queue_prefetch(query, session_id=session_id)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.queue_prefetch(query, session_id=session_id)
            except Exception:
                pass

    def _read_anchor_card(self) -> str:
        fn = config.get("anchor_card", "filename", default="anchor_card.md")
        path = config.plugin_dir() / fn
        if not path.exists():
            return ""
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = None
        with self._cache_lock:
            cache = self._anchor_cache
        if cache is not None and mtime is not None and cache[0] == mtime:
            return cache[1]
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        keep = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            keep.append(stripped)
        rendered = "\n".join(f"- {line}" for line in keep)
        if mtime is not None:
            with self._cache_lock:
                self._anchor_cache = (mtime, rendered)
        return rendered

    def _fetch_honcho_peer_card(self) -> str:
        """Static peer card via Honcho — no LLM, no representation summary.
        Short-TTL cached because the card rarely changes between turns."""
        if self._honcho is None:
            return ""
        now = time.monotonic()
        with self._cache_lock:
            cache = self._peer_cache
        if cache is not None and cache[0] > now:
            return cache[1]
        text = ""
        try:
            raw = self._honcho.handle_tool_call("honcho_profile", {})
            data = json.loads(raw) if isinstance(raw, str) else raw
            card = data.get("card") if isinstance(data, dict) else None
            if isinstance(card, list) and card:
                text = "\n".join(f"- {item}" for item in card)
            elif isinstance(data, dict) and data.get("hint"):
                text = f"_{data['hint']}_"
        except Exception as exc:
            logger.debug("mnemosyne: honcho profile fetch failed: %s", exc)
        with self._cache_lock:
            self._peer_cache = (now + self._peer_cache_ttl_s, text)
        return text

    def _fetch_hindsight_recall(self, query: str, max_tokens: int) -> str:
        if self._hindsight is None or not query:
            return ""
        try:
            raw = self._hindsight.handle_tool_call(
                "hindsight_recall",
                {"query": _truncate_recall_query(query),
                 "max_tokens": min(max_tokens, 4096)},
            )
            if isinstance(raw, str):
                # Try JSON, fall back to plain text
                try:
                    data = json.loads(raw)
                    return self._format_hindsight_results(data)
                except Exception:
                    return raw
            return self._format_hindsight_results(raw)
        except Exception as exc:
            logger.debug("mnemosyne: hindsight recall failed: %s", exc)
            return ""

    def _format_hindsight_results(self, data: Any) -> str:
        if isinstance(data, str):
            return self._dedupe_recall_text(data)
        if isinstance(data, list):
            joined = "\n".join(self._extract_text(item) for item in data
                               if self._extract_text(item))
            return self._dedupe_recall_text(joined)
        if isinstance(data, dict):
            # 'result' is the key the hermes hindsight plugin uses for
            # numbered-list recall output. Check it first.
            for key in ("result", "memories", "results", "items", "matches", "data", "text"):
                v = data.get(key)
                if v:
                    return self._format_hindsight_results(v)
        return str(data)

    @staticmethod
    def _dedupe_recall_text(text: str) -> str:
        """Hindsight returns top-N candidates ranked by similarity to the
        QUERY, not pairwise-distinct. With 40+ paraphrases of one event in
        the bank, all of them survive the existing exact-canonical-key
        filter and bloat context.

        Hybrid clustering (see ``dedup.cluster_lines``):
          - Jaccard token overlap groups paraphrases.
          - Embedding cosine confirms the merge — pairs that pass
            Jaccard but fail cosine are kept apart, protecting against
            "Barsik got sick" vs "Barsik died" type collapses.
        Configurable via ``prefetch.dedup_*`` keys."""
        if not text:
            return text
        from .dedup import cluster_lines

        cleaned: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            content = stripped
            for prefix in range(1, 100):
                pfx = f"{prefix}. "
                if content.startswith(pfx):
                    content = content[len(pfx):]
                    break
            head, sep, _ = content.partition(" | Involving:")
            content = head if sep else content
            if content:
                cleaned.append(content)

        kept = cluster_lines(cleaned)
        return "\n".join(f"- {item}" for item in kept)

    @staticmethod
    def _extract_text(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return (item.get("text") or item.get("content") or
                    item.get("body") or "")
        return ""

    def _filter_forgotten(self, text: str) -> str:
        """Drop lines whose canonical key was point-forgotten OR whose
        token set hits any stored semantic forget signature.

        Matching uses **candidate-containment**: ``|cand ∩ sig| / |cand|``.
        Asks "is most of this candidate's vocabulary covered by the
        forget signature?" — the right question for asymmetric sizes (a
        signature accumulates many tokens across an op; a candidate is
        one line). Symmetric Jaccard fails here: a 6-token candidate vs
        an 11-token signature with 4 shared words scores 0.31 — below
        any safe threshold — even though every content word in the
        candidate IS in the signature. Containment scores 0.66 and
        catches the paraphrase as intended.
        """
        if not self._fact_store:
            return text

        # Refresh the in-memory signature cache on demand. The table is
        # tiny (designed to stay <1k rows), so we just re-read it on
        # each filter pass for correctness; cost is microseconds.
        try:
            sigs = self._fact_store.list_signatures()
        except Exception:
            sigs = []
        try:
            cont_min = float(config.get("forget", "signature_jaccard_min",
                                        default=0.5))
        except Exception:
            cont_min = 0.5

        # Pre-compute signature token sets once per call.
        sig_tokens: List[tuple] = []
        for sig in sigs:
            tokens = set(sig.get("tokens") or [])
            if tokens:
                sig_tokens.append((sig.get("id"), tokens))

        from .dedup import _content_tokens
        kept: List[str] = []
        sig_hits: set = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            # Already-tombstoned lines from Hindsight
            if "[FORGOTTEN" in line:
                continue
            # Exact-key forget marks
            if _is_forgotten(self._fact_store, line):
                continue
            # Semantic signature filter (candidate-containment)
            if sig_tokens:
                content = stripped
                # strip "1. " number prefix and trailing "| Involving:"
                head, sep, _ = content.partition(" | Involving:")
                content = head if sep else content
                line_tokens = _content_tokens(content)
                if line_tokens:
                    matched = False
                    for sid, tokens in sig_tokens:
                        inter = line_tokens & tokens
                        cont = len(inter) / len(line_tokens)
                        if cont >= cont_min:
                            matched = True
                            if sid:
                                sig_hits.add(sid)
                            break
                    if matched:
                        continue
            kept.append(line)

        # Bump last_match_ts on signatures that did real work — keeps
        # vacuum from dropping useful sigs.
        for sid in sig_hits:
            try:
                self._fact_store.touch_signature(int(sid))
            except Exception:
                pass

        return "\n".join(kept)

    def _apply_conflict_resolver(self, sections: List[str]) -> List[str]:
        """If we detect a contradiction between the user profile and a fact,
        annotate both inline. Best-effort; rule-based detector."""
        if len(sections) < 3:
            return sections
        anchor, profile, facts = sections[0], sections[1], sections[2]
        profile_lines = [ln for ln in profile.splitlines() if ln.startswith("- ")]
        annotated_facts: List[str] = []
        today = today_iso()
        for fact_line in facts.splitlines():
            if not fact_line.strip() or fact_line.startswith("#"):
                annotated_facts.append(fact_line)
                continue
            conflict = False
            for profile_line in profile_lines:
                if is_contradiction(fact_line, profile_line):
                    a, b = label_pair(
                        fact_line, {"label": "Hindsight", "when": today},
                        profile_line, {"label": "Honcho profile"},
                    )
                    annotated_facts.append(a)
                    annotated_facts.append(b)
                    conflict = True
                    break
            if not conflict:
                annotated_facts.append(fact_line)
        return [anchor, profile, "\n".join(annotated_facts)]

    # ------------------------------------------------------------------
    # Write path (plan items 1, 2, 3, 10)
    # ------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Under approved writes a raw turn is not a write anyone approved.
        if not self._may_ingest_turns():
            return
        # Bump fact_store on user turn (cheap, exact-key dedup only).
        if self._fact_store and user_content.strip():
            try:
                self._fact_store.bump(user_content, source="conversation")
            except Exception as exc:
                logger.debug("mnemosyne: fact_store bump failed: %s", exc)

        # Break the prefetch→extract→retain feedback loop: strip from the
        # assistant turn anything we already injected as recalled context,
        # so Hindsight's extractor can't re-mint paraphrases of facts it
        # just gave us. user_content is left untouched — that's the only
        # actual new information in the turn.
        cleaned_assistant = self._strip_prefetched(assistant_content)

        futures = []
        if self._honcho:
            futures.append(self._executor.submit(
                self._honcho.sync_turn, user_content, cleaned_assistant,
                session_id=session_id,
            ))
        if self._hindsight:
            futures.append(self._executor.submit(
                self._hindsight.sync_turn, user_content, cleaned_assistant,
                session_id=session_id,
            ))
        for f in futures:
            try:
                f.result(timeout=5)
            except Exception as exc:
                logger.warning("mnemosyne: sync_turn fan-out failure: %s", exc)

    _SHINGLE_SIZE = 8        # words per shingle
    _SHINGLE_HIT_RATIO = 0.5 # fraction of a paragraph's shingles that must
                             # be in the prefetch to qualify for stripping

    def _strip_prefetched(self, assistant_content: str) -> str:
        """Drop paragraphs from `assistant_content` that mostly repeat
        text we just put into the prompt via prefetch.

        Why: Hindsight.sync_turn LLM-extracts facts from the JSON of the
        turn. If the assistant cited or paraphrased the prefetched
        memories, the extractor mints fresh records of them — every
        session. Recall picks them up next time, and the bank grows
        without bound (the Barsik loop).

        Approach: word-shingles, paragraph granularity. If a paragraph's
        shingle hit rate against the prefetch is high, drop it. Conservative
        thresholds: short paragraphs (<8 words) untouched, partial
        paraphrases survive."""
        if not assistant_content or not self._last_prefetch:
            return assistant_content
        if not bool(config.get("prefetch", "strip_from_extraction", default=True)):
            return assistant_content

        prefetch_shingles = self._build_shingles(self._last_prefetch)
        if not prefetch_shingles:
            return assistant_content

        kept_paragraphs: List[str] = []
        for para in assistant_content.split("\n\n"):
            if not para.strip():
                kept_paragraphs.append(para)
                continue
            para_shingles = self._build_shingles(para)
            if len(para_shingles) < 2:
                kept_paragraphs.append(para)
                continue
            hits = sum(1 for sh in para_shingles if sh in prefetch_shingles)
            ratio = hits / len(para_shingles)
            if ratio >= self._SHINGLE_HIT_RATIO:
                logger.debug("mnemosyne: stripped paraphrased paragraph "
                             "(%.0f%% shingle overlap with prefetch)",
                             ratio * 100)
                continue
            kept_paragraphs.append(para)
        return "\n\n".join(kept_paragraphs)

    @classmethod
    def _build_shingles(cls, text: str) -> set:
        if not text:
            return set()
        # Lowercase + strip non-word characters; same logic as
        # fact_store._canonical_key but token-level so word order matters.
        import re as _re
        words = _re.findall(r"\w+", text.lower(), flags=_re.UNICODE)
        if len(words) < cls._SHINGLE_SIZE:
            return set()
        return {
            tuple(words[i:i + cls._SHINGLE_SIZE])
            for i in range(len(words) - cls._SHINGLE_SIZE + 1)
        }

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Plan item 10 — built-in memory bridge.

        Mirror every memory(...) call from the user-facing tool into Hindsight
        with `source:user_explicit` and a max-strength fact_store mark.

        Hermes calls this only for writes that committed, so it is the one
        ingest path left under ``ingest.mode=approved_writes``."""
        if self._blocked or not self._writes_allowed or not action:
            return
        metadata = metadata or {}
        # Hermes names the entry to replace or remove by a unique substring in
        # metadata["old_text"]; a remove carries no content of its own.
        if action == "remove":
            content = content or str(metadata.get("old_text") or "")
        if not content:
            return
        # A write may invalidate the cached peer card (e.g. profile update).
        with self._cache_lock:
            self._peer_cache = None
        try:
            self._write_executor.submit(self._apply_memory_write, action, target, content, metadata)
        except RuntimeError as exc:  # executor already shut down
            logger.warning("mnemosyne: memory write after shutdown dropped: %s", exc)

    def flush_writes(self, timeout: float = 30.0) -> bool:
        """Wait for queued backend writes; True if they all finished in time."""
        try:
            done = self._write_executor.submit(lambda: None)
            done.result(timeout=timeout)
            return True
        except Exception:
            return False

    def _apply_memory_write(self, action: str, target: str, content: str,
                            metadata: Dict[str, Any]) -> None:
        # Pass-through to inner providers so they can do their own bookkeeping.
        if self._honcho:
            try:
                self._honcho.on_memory_write(action, target, content, metadata)
            except Exception as exc:
                logger.warning("mnemosyne: Honcho on_memory_write failed: %s", exc)
        if self._hindsight:
            try:
                self._hindsight.on_memory_write(action, target, content, metadata)
            except Exception as exc:
                logger.warning("mnemosyne: Hindsight on_memory_write failed: %s", exc)

        self._mirror_to_openviking(action, target, content, metadata)

        if action == "remove":
            if self._fact_store:
                try:
                    self._fact_store.mark_forgotten(content)
                except Exception as exc:
                    logger.warning("mnemosyne: fact_store mark_forgotten failed: %s", exc)
            if self._hindsight:
                try:
                    self._hindsight.handle_tool_call(
                        "hindsight_retain",
                        {
                            "content": f"[FORGOTTEN {today_iso()}] {content}",
                            "tags": [f"forgotten:{today_iso()}", "source:built_in_remove"],
                        },
                    )
                except Exception as exc:
                    logger.warning("mnemosyne: hindsight tombstone for removed memory failed: %s", exc)
            return

        if action in ("add", "replace"):
            if self._fact_store:
                try:
                    self._fact_store.force_strong(content, source="user_explicit")
                except Exception as exc:
                    logger.warning("mnemosyne: fact_store force_strong failed: %s", exc)
            if self._hindsight:
                try:
                    tags = [f"ts:{today_iso()}", "source:user_explicit",
                            f"target:{target or 'memory'}"]
                    if action == "replace":
                        tags.append("supersedes:built_in")
                    self._hindsight.handle_tool_call(
                        "hindsight_retain",
                        {"content": content, "tags": tags},
                    )
                except Exception as exc:
                    logger.warning("mnemosyne: hindsight retain mirror failed: %s", exc)

    def _mirror_to_openviking(self, action: str, target: str, content: str,
                              metadata: Dict[str, Any]) -> None:
        store = self._openviking
        if store is None:
            return
        try:
            if action == "add":
                store.write(target, content)
            elif action == "replace":
                new_uri = store.write(target, content)
                if metadata.get("old_text"):
                    store.delete_containing(target, str(metadata["old_text"]), keep=new_uri)
            elif action == "remove":
                store.delete_containing(target, content)
        except Exception as exc:
            logger.warning("mnemosyne: OpenViking %s mirror failed: %s", action, exc)

    def _openviking_recall_text(self, query: str, limit: int = 10) -> str:
        store = self._openviking
        if store is None or not query:
            return ""
        try:
            uris = [item["uri"] for item in store.find(query, limit=limit)]
            # The files themselves, not the server's abstracts: the files are the approved text.
            texts = store.read_many(uris)
            return "\n".join(f"- {text[:500]} ({uri})" for uri, text in zip(uris, texts) if text)
        except Exception as exc:
            logger.warning("mnemosyne: OpenViking recall failed: %s", exc)
            return ""

    def _handle_read(self, args: Dict[str, Any]) -> str:
        uri = str(args.get("uri") or "").strip()
        store = self._openviking
        if store is None:
            return json.dumps({"error": "openviking not available"})
        if not store.owns(uri):
            return json.dumps({"error": "that URI is outside this conversation's memory"})
        try:
            return json.dumps({"uri": uri, "content": self._filter_forgotten(store.read(uri))},
                              ensure_ascii=False)
        except Exception as exc:
            logger.warning("mnemosyne: OpenViking read failed: %s", exc)
            return json.dumps({"error": f"memory_read failed: {exc}"}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Tools (plan item 6 — curated 6 with tightened descriptions)
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        exposed = config.get("tools", "expose", default=[
            "memory_profile", "memory_reasoning", "memory_conclude",
            "memory_recall", "memory_reflect", "memory_forget", "memory_read",
        ])
        catalogue = {
            "memory_profile":   _PROFILE_SCHEMA,
            "memory_reasoning": _REASONING_SCHEMA,
            "memory_conclude":  _CONCLUDE_SCHEMA,
            "memory_recall":    _RECALL_SCHEMA,
            "memory_reflect":   _REFLECT_SCHEMA,
            "memory_forget":    MEMORY_FORGET_SCHEMA,
            "memory_read":      _READ_SCHEMA,
        }
        return [catalogue[n] for n in exposed if n in catalogue and self._tool_usable(n)]

    # Tools that write to a backend directly, outside Hermes' write approval.
    _DIRECT_WRITE_TOOLS = ("memory_conclude",)

    def _tool_usable(self, tool_name: str) -> bool:
        if self._blocked:
            return False
        if tool_name == "memory_forget":
            return self._hindsight is not None and self._writes_allowed
        if tool_name == "memory_recall":
            return self._hindsight is not None or self._openviking is not None
        backend = _TOOL_DISPATCH.get(tool_name, (None,))[0]
        if backend is None or getattr(self, f"_{backend}", None) is None:
            return False
        if tool_name in self._DIRECT_WRITE_TOOLS:
            return self._writes_allowed and not self._policy.approved_writes_only
        return True

    # Per-tool timeout (seconds), env-overridable — see config.py _ENV_MAP.
    # Defaults are generous (3-5 min for reasoning paths) so genuine deep
    # synthesis isn't truncated; tighten via MNEMOSYNE_TIMEOUT_* env vars
    # if a specific call misbehaves.
    @staticmethod
    def _timeout_for(tool_name: str) -> Optional[float]:
        key_map = {
            "memory_recall":    "recall",
            "memory_reasoning": "reasoning",
            "memory_reflect":   "reflect",
            "memory_profile":   "profile",
            "memory_conclude":  "conclude",
            "memory_forget":    "forget",
            "memory_read":      "recall",
        }
        key = key_map.get(tool_name)
        if key is None:
            return None
        try:
            return float(config.get("timeouts", key,
                                    default=config.get("timeouts", "default",
                                                       default=120)))
        except Exception:
            return None

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._blocked:
            return json.dumps({"error": f"memory unavailable: {self._disabled_reason}"})
        if not self._tool_usable(tool_name):
            return json.dumps({"error": f"{tool_name} is not available in this configuration"})
        if tool_name == "memory_profile" and args.get("card") is not None and (
                self._policy.approved_writes_only or not self._writes_allowed):
            return json.dumps({"error": "profile updates go through the built-in memory tool "
                                        "in this configuration; memory_profile is read-only"})
        if tool_name == "memory_forget":
            return self._handle_forget(args)
        if tool_name == "memory_read":
            return self._handle_read(args)
        if tool_name == "memory_recall" and self._hindsight is None:
            return self._recall_result(self._openviking_recall_text(str(args.get("query") or "")))

        mapping = _TOOL_DISPATCH.get(tool_name)
        if not mapping:
            raise NotImplementedError(f"mnemosyne does not handle tool {tool_name}")
        provider_attr, inner_name = mapping
        provider = getattr(self, f"_{provider_attr}", None)
        if provider is None:
            return json.dumps({"error": f"{provider_attr} not available"})

        # Truncate the query argument for tools that hit Hindsight's 500-token
        # recall limit. Defensive — without this, an LLM-formed long query
        # comes back as 400 Bad Request from Hindsight.
        if tool_name in ("memory_recall", "memory_reflect"):
            q = args.get("query")
            if isinstance(q, str) and len(q) > _RECALL_QUERY_MAX_CHARS:
                args = dict(args)
                args["query"] = _truncate_recall_query(q)
                logger.debug("mnemosyne: truncated %s query from %d to %d chars",
                             tool_name, len(q), len(args["query"]))

        timeout = self._timeout_for(tool_name)
        if timeout and timeout > 0:
            future = self._executor.submit(
                provider.handle_tool_call, inner_name, args, **kwargs
            )
            try:
                raw = future.result(timeout=timeout)
            except FuturesTimeout:
                logger.warning(
                    "mnemosyne: %s (→ %s) timed out after %.0fs",
                    tool_name, inner_name, timeout,
                )
                return json.dumps({
                    "error": f"{tool_name} timed out after {timeout:.0f}s "
                             f"(inner: {inner_name}). The underlying memory "
                             f"backend is slow or stuck — try a more focused "
                             f"query or a different memory tool.",
                }, ensure_ascii=False)
            except Exception as exc:
                logger.warning("mnemosyne: %s (→ %s) raised: %s",
                               tool_name, inner_name, exc)
                return json.dumps({"error": f"{tool_name} failed: {exc}"},
                                  ensure_ascii=False)
        else:
            raw = provider.handle_tool_call(inner_name, args, **kwargs)

        # Post-process recall: dedup duplicate surface forms and drop forgotten lines.
        if tool_name == "memory_recall":
            try:
                cleaned = self._format_hindsight_results(json.loads(raw)
                                                        if isinstance(raw, str) else raw)
                viking = self._openviking_recall_text(str(args.get("query") or ""))
                return self._recall_result("\n".join(t for t in (viking, cleaned) if t))
            except Exception as exc:
                logger.debug("mnemosyne: recall post-process failed: %s", exc)
                return raw
        return raw

    def _recall_result(self, text: str) -> str:
        cleaned = self._filter_forgotten(text) if text else ""
        if cleaned.strip():
            return json.dumps({"result": cleaned}, ensure_ascii=False)
        return json.dumps({"result": "No relevant memories found."}, ensure_ascii=False)

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "query required"}, ensure_ascii=False)
        if not self._fact_store:
            return json.dumps({"error": "fact_store unavailable"}, ensure_ascii=False)

        confirmed = bool(args.get("confirmed", False))
        indices = args.get("indices")
        if indices is not None:
            try:
                indices = [int(i) for i in indices]
            except Exception:
                indices = None
        max_items = int(args.get("max_items") or 30)

        result = forget_by_query(
            self._fact_store, self._hindsight, query,
            confirmed=confirmed,
            indices=indices,
            max_items=max_items,
        )
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Other ABC hooks (fan-out)
    # ------------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if not self._may_ingest_turns():
            return
        if self._honcho:
            try:
                self._honcho.on_turn_start(turn_number, message, **kwargs)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)
        if self._hindsight:
            try:
                self._hindsight.on_turn_start(turn_number, message, **kwargs)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._may_ingest_turns():
            return
        if self._honcho:
            try:
                self._honcho.on_session_end(messages)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)
        if self._hindsight:
            try:
                self._hindsight.on_session_end(messages)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        # New session — drop both per-turn caches so peer card and anchor
        # are re-evaluated for the new context.
        with self._cache_lock:
            self._peer_cache = None
            self._anchor_cache = None
        if self._blocked:
            return
        if self._honcho:
            try:
                self._honcho.on_session_switch(
                    new_session_id, parent_session_id=parent_session_id,
                    reset=reset, **kwargs,
                )
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)
        if self._hindsight:
            try:
                self._hindsight.on_session_switch(
                    new_session_id, parent_session_id=parent_session_id,
                    reset=reset, **kwargs,
                )
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._may_ingest_turns():
            return ""
        parts: List[str] = []
        if self._honcho:
            try:
                s = self._honcho.on_pre_compress(messages) or ""
                if s:
                    parts.append(s)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)
        if self._hindsight:
            try:
                s = self._hindsight.on_pre_compress(messages) or ""
                if s:
                    parts.append(s)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)
        return "\n\n".join(parts)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        if not self._may_ingest_turns():
            return
        if self._honcho:
            try:
                self._honcho.on_delegation(task, result, child_session_id=child_session_id, **kwargs)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)
        if self._hindsight:
            try:
                self._hindsight.on_delegation(task, result, child_session_id=child_session_id, **kwargs)
            except Exception as exc:
                logger.warning("mnemosyne: inner provider hook failed: %s", exc)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass


def register(ctx) -> None:
    """Hermes plugin entry point — register Mnemosyne as a memory provider."""
    ctx.register_memory_provider(MnemosyneMemoryProvider())
