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

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from . import config, recall_processing
from .fact_store import FactStore, today_iso
from .forget import (
    forget_by_query,
    _write_tombstone,
)
from .prefetch import Prefetcher
from .recovery import initialize_cursor_if_missing, replay_missed
from .tool_schemas import (
    DEFAULT_TOOL_NAMES,
    RECALL_QUERY_MAX_CHARS,
    TOOL_DISPATCH,
    TOOL_SCHEMAS,
    TOOL_TIMEOUT_KEYS,
    WRITE_TOOLS,
)

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
            ok,
            len(texts),
            when,
        )

    try:
        executor.submit(_runner)
    except Exception as exc:
        logger.debug("mnemosyne: tombstone executor submit failed: %s", exc)


# agent_context values whose sessions must not write memory. Hermes passes
# "cron" for scheduled jobs and "subagent" for delegate_task children, and
# documents that providers skip writes for them; Honcho also skips "flush".
_NO_WRITE_CONTEXTS = ("cron", "subagent", "flush")


class MnemosyneMemoryProvider(MemoryProvider):
    """Composite provider — Honcho for user model, Hindsight for facts."""

    def __init__(self) -> None:
        self._honcho: Optional[MemoryProvider] = None
        self._hindsight: Optional[MemoryProvider] = None
        # 4 workers: 2 for write fan-out (sync_turn), 2 spare for parallel
        # tool calls so agent-driven recall isn't queued behind background
        # retain jobs.
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="mnemosyne"
        )
        self._writes_allowed = True
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
        self._prefetcher = Prefetcher(self._executor)
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
            "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": "jina-embeddings-v5-text-small-retrieval-mlx",
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
        try:
            from plugins.memory.honcho import HonchoMemoryProvider

            self._honcho = HonchoMemoryProvider()
        except Exception as exc:
            logger.warning("mnemosyne: failed to load Honcho inner provider: %s", exc)

        try:
            from plugins.memory.hindsight import HindsightMemoryProvider

            self._hindsight = HindsightMemoryProvider()
        except Exception as exc:
            logger.warning(
                "mnemosyne: failed to load Hindsight inner provider: %s", exc
            )

    @property
    def name(self) -> str:
        return "mnemosyne"

    def _provider_availability(self) -> tuple[bool, bool]:
        h_ok = bool(self._honcho and self._honcho.is_available())
        i_ok = bool(self._hindsight and self._hindsight.is_available())
        return h_ok, i_ok

    def is_available(self) -> bool:
        # OR, not AND: a Hindsight outage must not disable the working Honcho
        # half (that would leave the agent with no memory at all). Tool calls to
        # an absent provider already fail loud and prefetch tolerates one missing.
        h_ok, i_ok = self._provider_availability()
        return h_ok or i_ok

    def unavailable_reason(self) -> str:
        h_ok, i_ok = self._provider_availability()
        if h_ok or i_ok:
            return ""
        parts = [
            "Honcho " + ("not loaded" if not self._honcho else "unavailable"),
            "Hindsight " + ("not loaded" if not self._hindsight else "unavailable"),
        ]
        return "both inner memory providers down (" + "; ".join(parts) + ")"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        with self._init_lock:
            self._writes_allowed = (
                str(kwargs.get("agent_context") or "") not in _NO_WRITE_CONTEXTS
            )
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

            # Name the down half so single-provider operation isn't invisible.
            h_ok, i_ok = self._provider_availability()
            if h_ok != i_ok:
                logger.warning(
                    "mnemosyne: running DEGRADED — %s available, %s down; "
                    "continuing with the available provider only",
                    "Honcho" if h_ok else "Hindsight",
                    "Hindsight" if h_ok else "Honcho",
                )
            elif not h_ok and not i_ok:
                logger.warning("mnemosyne: both inner providers unavailable at init")

            try:
                self._fact_store = FactStore()
                # One-shot vacuum: bound the signatures table so it
                # never silently grows past the configured ceiling.
                try:
                    max_sigs = int(config.get("forget", "max_signatures", default=1000))
                    stale_days = config.get(
                        "forget", "signature_stale_days", default=365
                    )
                    stale = int(stale_days) if stale_days else None
                    removed = self._fact_store.vacuum_signatures(
                        max_count=max_sigs,
                        stale_days=stale,
                    )
                    if removed:
                        logger.info(
                            "mnemosyne: vacuumed %d forget signature(s)", removed
                        )
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
            try:
                if not self._writes_allowed:
                    logger.debug("mnemosyne: recovery skipped in a non-writing session")
                elif initialize_cursor_if_missing():
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
                    logger.info(
                        "mnemosyne: recovery replayed %d turn pair(s)",
                        summary["replayed"],
                    )
                for reason in (
                    "stopped_at_failure",
                    "stopped_at_deadline",
                    "stopped_at_limit",
                ):
                    if summary.get(reason):
                        logger.info(
                            "mnemosyne: recovery %s — resumes next startup", reason
                        )
                        break
            except Exception as exc:
                logger.debug("mnemosyne: recovery failed: %s", exc)

        threading.Thread(target=_runner, name="mnemosyne-recovery", daemon=True).start()

    def shutdown(self) -> None:
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
        self._executor.shutdown(wait=False)

    def system_prompt_block(self) -> str:
        """Tell the agent about Mnemosyne's curated tool surface.

        Crucially we DO NOT delegate to ``self._honcho.system_prompt_block()``
        or ``self._hindsight.system_prompt_block()`` — those describe their
        native tool names (``honcho_*`` / ``hindsight_*``) which we
        deliberately hide behind our 6 curated tools. Letting them through
        would tell the LLM that ``honcho_search`` etc. exist when in fact
        only the curated set is callable, leading to phantom tool calls and
        confused tool selection."""
        return (
            "# Memory (Mnemosyne)\n"
            "You have a long-term memory system backed by two layers: a user "
            "model (style, preferences, behavioral patterns) and a fact store "
            "(prior conversations, entities, decisions). Both are accessed "
            "through these six tools — DO NOT call any tool name that starts "
            "with `honcho_` or `hindsight_`; those are not exposed.\n"
            "\n"
            "When to use which:\n"
            "- `memory_recall(query)`        — FIRST CHOICE for 'do you "
            "remember…', 'we discussed…', 'how did we fix…'.\n"
            "- `memory_reflect(query)`       — synthesised summary across "
            "multiple past facts ('what did we conclude about X?').\n"
            "- `memory_profile(card?)`       — read or update the user's "
            "profile card (stable preferences, role, communication style).\n"
            "- `memory_reasoning(query)`     — questions about the user "
            "**as a person** (style, habits). Slow — use sparingly.\n"
            "- `memory_conclude(conclusion)` — record a stable fact about "
            "the user.\n"
            "- `memory_forget(query)`        — TWO STEPS. First call with "
            "just the query returns a preview list of candidates. Show that "
            "list to the user verbatim, get explicit confirmation, then "
            "re-invoke with `confirmed=true` (or `indices=[1,3]` to pick a "
            "subset). NEVER call with `confirmed=true` on the first try.\n"
        )

    # ------------------------------------------------------------------
    # Prefetch (plan item 6/7 fusion: anchor → peer card → Hindsight recall)
    # ------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        result = self._prefetcher.prefetch(
            query,
            honcho=self._honcho,
            hindsight=self._hindsight,
            fact_store=self._fact_store,
        )
        self._last_prefetch = result
        return result

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self._prefetcher.queue_prefetch(
            query,
            session_id=session_id,
            honcho=self._honcho,
            hindsight=self._hindsight,
        )

    # ------------------------------------------------------------------
    # Write path (plan items 1, 2, 3, 10)
    # ------------------------------------------------------------------

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        if not self._writes_allowed:
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
            futures.append(
                self._executor.submit(
                    self._honcho.sync_turn,
                    user_content,
                    cleaned_assistant,
                    session_id=session_id,
                )
            )
        if self._hindsight:
            futures.append(
                self._executor.submit(
                    self._hindsight.sync_turn,
                    user_content,
                    cleaned_assistant,
                    session_id=session_id,
                )
            )
        for f in futures:
            try:
                f.result(timeout=5)
            except Exception as exc:
                logger.debug("mnemosyne: sync_turn fan-out failure: %s", exc)

    _SHINGLE_SIZE = 8  # words per shingle
    _SHINGLE_HIT_RATIO = 0.5  # fraction of a paragraph's shingles that must
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
                logger.debug(
                    "mnemosyne: stripped paraphrased paragraph "
                    "(%.0f%% shingle overlap with prefetch)",
                    ratio * 100,
                )
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
            tuple(words[i : i + cls._SHINGLE_SIZE])
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
        with `source:user_explicit` and a max-strength fact_store mark."""
        if not self._writes_allowed:
            return
        # A write may invalidate the cached peer card (e.g. profile update).
        self._prefetcher.invalidate_peer_card()
        # Pass-through to inner providers so they can do their own bookkeeping.
        if self._honcho:
            try:
                self._honcho.on_memory_write(action, target, content, metadata)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_memory_write(action, target, content, metadata)
            except Exception:
                pass

        if not content or not action:
            return

        if action == "remove":
            if self._fact_store:
                try:
                    self._fact_store.mark_forgotten(content)
                except Exception:
                    pass
            if self._hindsight:
                try:
                    self._hindsight.handle_tool_call(
                        "hindsight_retain",
                        {
                            "content": f"[FORGOTTEN {today_iso()}] {content}",
                            "tags": [
                                f"forgotten:{today_iso()}",
                                "source:built_in_remove",
                            ],
                        },
                    )
                except Exception:
                    pass
            return

        if action in ("add", "replace"):
            if self._fact_store:
                try:
                    self._fact_store.force_strong(content, source="user_explicit")
                except Exception:
                    pass
            if self._hindsight:
                try:
                    tags = [
                        f"ts:{today_iso()}",
                        "source:user_explicit",
                        f"target:{target or 'memory'}",
                    ]
                    if action == "replace":
                        tags.append("supersedes:built_in")
                    self._hindsight.handle_tool_call(
                        "hindsight_retain",
                        {"content": content, "tags": tags},
                    )
                except Exception as exc:
                    logger.debug("mnemosyne: hindsight retain mirror failed: %s", exc)

    # ------------------------------------------------------------------
    # Tools (plan item 6 — curated 6 with tightened descriptions)
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        exposed = config.get(
            "tools",
            "expose",
            default=list(DEFAULT_TOOL_NAMES),
        )
        return [
            TOOL_SCHEMAS[n]
            for n in exposed
            if n in TOOL_SCHEMAS and (self._writes_allowed or n not in WRITE_TOOLS)
        ]

    # Per-tool timeout (seconds), env-overridable — see config.py _ENV_MAP.
    # Defaults are generous (3-5 min for reasoning paths) so genuine deep
    # synthesis isn't truncated; tighten via MNEMOSYNE_TIMEOUT_* env vars
    # if a specific call misbehaves.
    @staticmethod
    def _timeout_for(tool_name: str) -> Optional[float]:
        key = TOOL_TIMEOUT_KEYS.get(tool_name)
        if key is None:
            return None
        try:
            return float(
                config.get(
                    "timeouts",
                    key,
                    default=config.get("timeouts", "default", default=120),
                )
            )
        except Exception:
            return None

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._writes_allowed and (
            tool_name in WRITE_TOOLS
            or (tool_name == "memory_profile" and args.get("card"))
        ):
            return json.dumps(
                {
                    "error": f"{tool_name} cannot write from this session "
                    "(cron, subagent or flush context)"
                }
            )
        if tool_name == "memory_forget":
            return self._handle_forget(args)

        mapping = TOOL_DISPATCH.get(tool_name)
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
            if isinstance(q, str) and len(q) > RECALL_QUERY_MAX_CHARS:
                args = dict(args)
                args["query"] = recall_processing.truncate_recall_query(q)
                logger.debug(
                    "mnemosyne: truncated %s query from %d to %d chars",
                    tool_name,
                    len(q),
                    len(args["query"]),
                )

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
                    tool_name,
                    inner_name,
                    timeout,
                )
                return json.dumps(
                    {
                        "error": f"{tool_name} timed out after {timeout:.0f}s "
                        f"(inner: {inner_name}). The underlying memory "
                        f"backend is slow or stuck — try a more focused "
                        f"query or a different memory tool.",
                    },
                    ensure_ascii=False,
                )
            except Exception as exc:
                logger.warning(
                    "mnemosyne: %s (→ %s) raised: %s", tool_name, inner_name, exc
                )
                return json.dumps(
                    {"error": f"{tool_name} failed: {exc}"}, ensure_ascii=False
                )
        else:
            raw = provider.handle_tool_call(inner_name, args, **kwargs)

        # Post-process recall: dedup duplicate surface forms and drop forgotten lines.
        if tool_name == "memory_recall":
            try:
                cleaned = recall_processing.format_hindsight_results(
                    json.loads(raw) if isinstance(raw, str) else raw
                )
                cleaned = recall_processing.filter_forgotten(
                    cleaned, fact_store=self._fact_store
                )
                if cleaned.strip():
                    return json.dumps({"result": cleaned}, ensure_ascii=False)
                return json.dumps(
                    {"result": "No relevant memories found."}, ensure_ascii=False
                )
            except Exception as exc:
                logger.debug("mnemosyne: recall post-process failed: %s", exc)
                return raw
        return raw

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "query required"}, ensure_ascii=False)
        if not self._fact_store:
            return json.dumps({"error": "fact_store unavailable"}, ensure_ascii=False)

        confirmed = bool(args.get("confirmed", False))
        preview_token = args.get("preview_token") or None
        indices = args.get("indices")
        if indices is not None:
            try:
                indices = [int(i) for i in indices]
            except Exception:
                indices = None
        max_items = int(args.get("max_items") or 30)

        result = forget_by_query(
            self._fact_store,
            self._hindsight,
            query,
            confirmed=confirmed,
            preview_token=preview_token,
            indices=indices,
            max_items=max_items,
        )
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Other ABC hooks (fan-out)
    # ------------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        if self._honcho:
            try:
                self._honcho.on_turn_start(turn_number, message, **kwargs)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_turn_start(turn_number, message, **kwargs)
            except Exception:
                pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._writes_allowed:
            return
        if self._honcho:
            try:
                self._honcho.on_session_end(messages)
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_session_end(messages)
            except Exception:
                pass

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
        self._prefetcher.invalidate_all()
        if self._honcho:
            try:
                self._honcho.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception:
                pass

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if not self._writes_allowed:
            return ""
        parts: List[str] = []
        if self._honcho:
            try:
                s = self._honcho.on_pre_compress(messages) or ""
                if s:
                    parts.append(s)
            except Exception:
                pass
        if self._hindsight:
            try:
                s = self._hindsight.on_pre_compress(messages) or ""
                if s:
                    parts.append(s)
            except Exception:
                pass
        return "\n\n".join(parts)

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs
    ) -> None:
        if not self._writes_allowed:
            return
        if self._honcho:
            try:
                self._honcho.on_delegation(
                    task, result, child_session_id=child_session_id, **kwargs
                )
            except Exception:
                pass
        if self._hindsight:
            try:
                self._hindsight.on_delegation(
                    task, result, child_session_id=child_session_id, **kwargs
                )
            except Exception:
                pass

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass


def register(ctx) -> None:
    """Hermes plugin entry point — register Mnemosyne as a memory provider.

    ``ingest.mode=approved_writes`` selects ``ApprovedMemoryProvider``, whose
    backends mirror only Hermes' approved built-in memory; the default keeps
    the Honcho + Hindsight composite above."""
    from . import policy

    if policy.approved_writes_only():
        from .approved import ApprovedMemoryProvider

        ctx.register_memory_provider(ApprovedMemoryProvider())
        return
    ctx.register_memory_provider(MnemosyneMemoryProvider())
