"""Parallel prefetch and per-provider anchor/profile caches.

The executor is borrowed from the memory provider, which owns its lifetime.
Backends and the fact store are supplied on each call, including after initialize.
Cache locks protect state only; disk and backend reads run outside the lock.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Executor
from typing import TYPE_CHECKING, List, Optional

from . import config, recall_processing
from .fact_store import FactStore

if TYPE_CHECKING:
    from agent.memory_provider import MemoryProvider

__all__ = ["Prefetcher"]

logger = logging.getLogger(__name__)


class Prefetcher:
    """Assemble pinned facts, profile and recall with independent read caches."""

    def __init__(self, executor: Executor) -> None:
        self._executor = executor
        # Per-turn caches to avoid re-reading anchor_card from disk and
        # re-fetching the Honcho peer card every prefetch. Anchor is keyed by
        # file mtime so manual edits are picked up immediately; peer card
        # uses a short TTL and is invalidated on session-switch / write.
        self._anchor_cache: Optional[tuple] = None  # (mtime, rendered_text)
        self._peer_cache: Optional[tuple] = None  # (expires_at_ts, text)
        self._peer_cache_ttl_s: float = float(
            config.get("prefetch", "peer_card_ttl_s", default=60.0)
        )
        self._cache_lock = threading.Lock()

    def invalidate_peer_card(self) -> None:
        """A permitted memory write may change the Honcho profile."""
        with self._cache_lock:
            self._peer_cache = None

    def invalidate_all(self) -> None:
        """A session switch must re-read both the anchor card and profile."""
        with self._cache_lock:
            self._peer_cache = None
            self._anchor_cache = None

    def prefetch(
        self,
        query: str,
        *,
        honcho: Optional[MemoryProvider],
        hindsight: Optional[MemoryProvider],
        fact_store: Optional[FactStore],
    ) -> str:
        max_total = int(config.get("prefetch", "max_total_tokens", default=4500))
        anchor_budget = int(config.get("prefetch", "anchor_token_budget", default=200))
        peer_card_budget = int(
            config.get("prefetch", "honcho_card_token_budget", default=200)
        )
        hindsight_budget = int(
            config.get("prefetch", "hindsight_token_budget", default=4096)
        )
        per_branch_timeout = float(
            config.get("prefetch", "parallel_timeout_s", default=12.0)
        )

        # Fan out the three independent fetches in parallel via the existing
        # executor. They are independent reads (anchor=disk, peer=Honcho,
        # hindsight=HTTP) — the prior serial layout was dominated by
        # _fetch_hindsight_recall (~6-8s). Concurrent execution caps total
        # time at the slowest branch plus a few ms of overhead.
        anchor_fut = self._executor.submit(self._read_anchor_card)
        peer_fut = self._executor.submit(self._fetch_honcho_peer_card, honcho)
        hindsight_fut = self._executor.submit(
            self._fetch_hindsight_recall,
            hindsight,
            query,
            hindsight_budget,
        )

        def _wait(fut, default=""):
            try:
                return fut.result(timeout=per_branch_timeout) or default
            except Exception as exc:
                logger.debug("mnemosyne: prefetch branch failed/timeout: %s", exc)
                return default

        anchor_text = _wait(anchor_fut)
        peer_card_text = _wait(peer_fut)
        hindsight_text = _wait(hindsight_fut)

        sections: List[str] = []

        if anchor_text:
            sections.append(
                "# Pinned (anchor card)\n"
                + recall_processing.truncate_to_chars(anchor_text, anchor_budget * 4)
            )

        if peer_card_text:
            sections.append(
                "# User profile\n"
                + recall_processing.truncate_to_chars(
                    peer_card_text, peer_card_budget * 4
                )
            )

        if hindsight_text:
            hindsight_text = recall_processing.filter_forgotten(
                hindsight_text, fact_store=fact_store
            )
            if hindsight_text:
                sections.append(
                    "# Facts (relevant)\n"
                    + recall_processing.truncate_to_chars(
                        hindsight_text, hindsight_budget * 4
                    )
                )

        if len(sections) >= 3:
            sections = recall_processing.apply_conflict_resolver(sections)

        result = "\n\n".join(sections)
        capped = recall_processing.truncate_to_chars(result, max_total * 4)
        return capped

    @staticmethod
    def queue_prefetch(
        query: str,
        *,
        session_id: str = "",
        honcho: Optional[MemoryProvider],
        hindsight: Optional[MemoryProvider],
    ) -> None:
        if honcho:
            try:
                honcho.queue_prefetch(query, session_id=session_id)
            except Exception:
                pass
        if hindsight:
            try:
                hindsight.queue_prefetch(query, session_id=session_id)
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

    def _fetch_honcho_peer_card(self, honcho: Optional[MemoryProvider]) -> str:
        """Static peer card via Honcho — no LLM, no representation summary.
        Short-TTL cached because the card rarely changes between turns."""
        if honcho is None:
            return ""
        now = time.monotonic()
        with self._cache_lock:
            cache = self._peer_cache
        if cache is not None and cache[0] > now:
            return cache[1]
        text = ""
        try:
            raw = honcho.handle_tool_call("honcho_profile", {})
            data = json.loads(raw) if isinstance(raw, str) else raw
            # A honcho_profile read returns the card as a list under "result";
            # "card" only appears on a write, so reading it dropped every profile.
            result = data.get("result") if isinstance(data, dict) else None
            if isinstance(result, list) and result:
                text = "\n".join(f"- {item}" for item in result)
            elif isinstance(data, dict) and data.get("hint"):
                text = f"_{data['hint']}_"
        except Exception as exc:
            logger.debug("mnemosyne: honcho profile fetch failed: %s", exc)
        with self._cache_lock:
            self._peer_cache = (now + self._peer_cache_ttl_s, text)
        return text

    @staticmethod
    def _fetch_hindsight_recall(
        hindsight: Optional[MemoryProvider],
        query: str,
        max_tokens: int,
    ) -> str:
        if hindsight is None or not query:
            return ""
        try:
            raw = hindsight.handle_tool_call(
                "hindsight_recall",
                {
                    "query": recall_processing.truncate_recall_query(query),
                    "max_tokens": min(max_tokens, 4096),
                },
            )
            if isinstance(raw, str):
                # Try JSON, fall back to plain text
                try:
                    data = json.loads(raw)
                    return recall_processing.format_hindsight_results(data)
                except Exception:
                    return raw
            return recall_processing.format_hindsight_results(raw)
        except Exception as exc:
            logger.debug("mnemosyne: hindsight recall failed: %s", exc)
            return ""
