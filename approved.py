"""Approved-writes memory provider.

Under ``ingest.mode=approved_writes`` Mnemosyne's backends mirror Hermes'
built-in memory files (MEMORY.md, USER.md) and nothing else. Hermes writes
those files only after ``memory.write_approval`` let a write through, either
inline or later through ``/memory approve``; the second path updates the
files without calling providers, so this provider reconciles against the
files instead of trusting ``on_memory_write`` payloads. Turns, session ends,
compressions and delegations never reach a backend, and no backend-side LLM
writes anything (see ``openviking_store`` and ``hindsight_store``).

The provider starts disabled and is enabled only at the end of an
``initialize`` that resolved the profile's home and namespace and connected
at least one backend; anything else leaves it without memory for the session.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from . import memory_files, policy
from .hindsight_store import HindsightStore
from .hindsight_store import configured_url as _hindsight_url
from .openviking_store import OpenVikingStore
from .openviking_store import is_configured as _openviking_configured
from .reconcile import reconcile

logger = logging.getLogger(__name__)

_FLUSH_TIMEOUT_S = 5.0
_RECALL_LIMIT = 10
_MAX_CONTEXT_CHARS = 6000

_RECALL_SCHEMA = {
    "name": "memory_recall",
    "description": (
        "Search long-term memory: the entries saved with the built-in `memory` tool, "
        "including ones from past sessions. Returns matching entries."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look for."}},
        "required": ["query"],
    },
}

_READ_SCHEMA = {
    "name": "memory_read",
    "description": "Read one memory entry in full by the viking:// URI memory_recall returned.",
    "parameters": {
        "type": "object",
        "properties": {"uri": {"type": "string", "description": "viking:// URI from memory_recall."}},
        "required": ["uri"],
    },
}

_REFLECT_SCHEMA = {
    "name": "memory_reflect",
    "description": "Answer a question by reasoning over the saved memory entries.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Natural-language question."}},
        "required": ["query"],
    },
}


class ApprovedMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._policy = policy.approved_policy()
        self._disabled_reason: Optional[str] = "not initialized"
        self._home: Optional[str] = None
        self._namespace: Optional[str] = None
        self._writes_allowed = False
        self._openviking: Optional[OpenVikingStore] = None
        self._hindsight: Optional[HindsightStore] = None
        self._signature: Optional[tuple] = None
        self._seen_files: Dict[str, bool] = {}
        self._reconcile_lock = threading.Lock()
        self._reconcile_queued = False
        # One worker: reconciles run one at a time, off the agent's turn.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mnemosyne-reconcile")

    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        if policy.approved_problem(self._policy):
            return False
        checks = {"openviking": _openviking_configured, "hindsight": lambda: bool(_hindsight_url())}
        return all(checks[b]() for b in self._policy.backends)

    # -- lifecycle ---------------------------------------------------------------

    def _disable(self, reason: str) -> None:
        self._disabled_reason = reason
        logger.warning("mnemosyne: memory disabled for this session — %s", reason)

    @property
    def _blocked(self) -> bool:
        return self._disabled_reason is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            self._initialize(kwargs)
        except Exception as exc:  # Hermes logs and keeps the provider; stay disabled
            self._disable(f"initialize failed: {exc}")

    def _initialize(self, kwargs: Dict[str, Any]) -> None:
        home = str(kwargs.get("hermes_home") or "").strip()
        if not home:
            return self._disable("Hermes passed no hermes_home")
        self._policy = policy.approved_policy(Path(home))
        problem = policy.approved_problem(self._policy)
        if problem:
            return self._disable(problem)
        self._home = home
        self._namespace = policy.namespace(str(kwargs.get("agent_identity") or ""), home)
        self._writes_allowed = policy.writes_allowed(str(kwargs.get("agent_context") or ""))
        if "openviking" in self._policy.backends:
            self._openviking = OpenVikingStore.connect(self._namespace)
        if "hindsight" in self._policy.backends:
            self._hindsight = HindsightStore.connect(self._namespace, Path(home))
        if self._openviking is None and self._hindsight is None:
            return self._disable("no configured backend is reachable")
        self._disabled_reason = None
        self._schedule_reconcile()

    def shutdown(self) -> None:
        self.flush(_FLUSH_TIMEOUT_S)
        self._worker.shutdown(wait=False, cancel_futures=True)
        for store in (self._openviking, self._hindsight):
            if store is not None:
                store.close()

    # -- reconciliation ------------------------------------------------------------

    def _schedule_reconcile(self) -> None:
        if self._blocked or not self._writes_allowed:
            return
        with self._reconcile_lock:
            if self._reconcile_queued:
                return
            self._reconcile_queued = True
        try:
            self._worker.submit(self._run_reconcile)
        except RuntimeError:  # shutting down
            with self._reconcile_lock:
                self._reconcile_queued = False

    def _run_reconcile(self) -> None:
        with self._reconcile_lock:
            self._reconcile_queued = False
        self.reconcile_now()

    def reconcile_now(self) -> Dict[str, Any]:
        """Reconcile every connected backend with the approved files; returns a summary."""
        if self._blocked or not self._home:
            return {"error": self._disabled_reason or "not initialized"}
        self._signature = memory_files.signature(self._home)
        files = memory_files.read_all(self._home)
        for target, te in files.items():
            if te.missing and self._seen_files.get(target):
                logger.warning("mnemosyne: %s disappeared; its mirrored copies are kept. "
                               "Run `hermes mnemosyne purge --yes` to remove them.",
                               memory_files.TARGETS[target])
            self._seen_files[target] = not te.missing
        summary: Dict[str, Any] = {}
        for store in (self._openviking, self._hindsight):
            if store is None:
                continue
            out = reconcile(store, files)
            summary[store.name] = {"created": out.created, "deleted": out.deleted,
                                   "kept_on_unreadable": out.kept_on_unreadable, "errors": out.errors}
        return summary

    def purge(self) -> Dict[str, Any]:
        """Delete everything this profile's namespace holds in every connected backend."""
        if self._blocked:
            return {"error": self._disabled_reason}
        done: Dict[str, Any] = {}
        for store in (self._openviking, self._hindsight):
            if store is None:
                continue
            try:
                store.purge()
                done[store.name] = "purged"
            except Exception as exc:
                done[store.name] = f"failed: {exc}"
        return done

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait for queued reconciles; True if they finished in time."""
        try:
            self._worker.submit(lambda: None).result(timeout=timeout)
            return True
        except Exception:
            return False

    # -- hooks -------------------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        # The file is already written when Hermes calls this; reconcile against it.
        self._schedule_reconcile()

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        # Catches /memory approve, which changes the files without calling providers.
        if self._home and memory_files.signature(self._home) != self._signature:
            self._schedule_reconcile()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        return None

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        return None

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        return None

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    # -- recall ------------------------------------------------------------------

    def _recall_lines(self, query: str) -> List[str]:
        lines: List[str] = []
        if self._openviking is not None:
            try:
                for item in self._openviking.find(query, limit=_RECALL_LIMIT):
                    if item["content"]:
                        lines.append(f"- {item['content'][:500]} ({item['uri']})")
            except Exception as exc:
                logger.warning("mnemosyne: OpenViking recall failed: %s", exc)
        if self._hindsight is not None:
            try:
                lines += [f"- {text[:500]}" for text in self._hindsight.recall(query)]
            except Exception as exc:
                logger.warning("mnemosyne: Hindsight recall failed: %s", exc)
        return list(dict.fromkeys(lines))

    def system_prompt_block(self) -> str:
        names = [s["name"] for s in self.get_tool_schemas()]
        if not names:
            return ""
        lines = ["# Memory (Mnemosyne)",
                 "Long-term memory holds the entries saved with the built-in `memory` tool. "
                 "To store something, use that tool; it is the only way anything is kept."]
        if "memory_recall" in names:
            lines.append("- `memory_recall(query)` — search saved entries, including earlier sessions'.")
        if "memory_read" in names:
            lines.append("- `memory_read(uri)` — full text of an entry by the URI memory_recall returned.")
        if "memory_reflect" in names:
            lines.append("- `memory_reflect(query)` — an answer reasoned over the saved entries.")
        return "\n".join(lines) + "\n"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._blocked or not self._policy.prefetch or not query:
            return ""
        lines = self._recall_lines(query[:1500])
        if not lines:
            return ""
        return ("# Memory (relevant entries)\n" + "\n".join(lines))[:_MAX_CONTEXT_CHARS]

    # -- tools -------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._blocked:
            return []
        tools = [_RECALL_SCHEMA]
        if self._openviking is not None:
            tools.append(_READ_SCHEMA)
        if self._hindsight is not None and self._policy.reflect:
            tools.append(_REFLECT_SCHEMA)
        return tools

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._blocked:
            return json.dumps({"error": f"memory unavailable: {self._disabled_reason}"})
        names = {s["name"] for s in self.get_tool_schemas()}
        if tool_name not in names:
            return json.dumps({"error": f"{tool_name} is not available"})
        if tool_name == "memory_recall":
            lines = self._recall_lines(str(args.get("query") or "")[:1500])
            return json.dumps({"result": "\n".join(lines) or "No matching memory entries."},
                              ensure_ascii=False)
        if tool_name == "memory_read":
            uri = str(args.get("uri") or "").strip()
            if self._openviking is None or not self._openviking.owns(uri):
                return json.dumps({"error": "that URI is outside this profile's memory"})
            try:
                return json.dumps({"uri": uri, "content": self._openviking.read(uri)}, ensure_ascii=False)
            except Exception as exc:
                return json.dumps({"error": f"memory_read failed: {exc}"}, ensure_ascii=False)
        if self._hindsight is None:
            return json.dumps({"error": "hindsight not available"})
        try:
            return json.dumps({"result": self._hindsight.reflect(str(args.get("query") or "")[:1500])},
                              ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"memory_reflect failed: {exc}"}, ensure_ascii=False)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        pass
