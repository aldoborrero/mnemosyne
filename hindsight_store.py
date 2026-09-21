"""Hindsight backend: one verbatim document per approved entry.

Mnemosyne talks to the Hindsight API with ``hindsight_client`` instead of
wrapping Hermes' ``HindsightMemoryProvider``, because that wrapper can do
none of what approved-only storage needs: it retains with the bank's default
LLM extraction, recalls LLM-consolidated observations only, has no document
ids, no delete and no bank configuration.

Each profile gets a bank ``mnemosyne-<ns>`` configured at connect time with
``retain_extraction_mode="chunks"`` (the entry is stored as-is, no LLM call),
observations and auto-consolidation off (no LLM-written memories) and
``store_document_text``. The connection fails closed unless the server
reports that configuration back. Each entry is document
``mn-<target>-<id>`` tagged ``mnemosyne`` and ``target:<target>``; removal
deletes the document with its memory units.

Recall is not an LLM call. Reflect is, and the server keeps LLM traces by
default (``HINDSIGHT_API_LLM_TRACE_ENABLED``, one day), so it is opt-in.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Set

from . import config
from .policy import is_cloud_hindsight

logger = logging.getLogger(__name__)

_TAG = "mnemosyne"
_PAGE = 100
_REQUIRED_BANK_CONFIG = {
    "retain_extraction_mode": "chunks",
    "enable_observations": False,
    "enable_auto_consolidation": False,
}


def _secret(name: str) -> str:
    try:
        from agent.secret_scope import get_secret  # profile-scoped under a multiplexed gateway
        return get_secret(name, "") or ""
    except Exception:
        import os
        return os.environ.get(name, "")


def _thread_loop() -> None:
    # hindsight_client's sync wrappers run on the thread's event loop; one
    # thread and one loop keep its HTTP sessions on the loop that made them.
    asyncio.set_event_loop(asyncio.new_event_loop())


def configured_url(home=None) -> str:
    return str(config.get("hindsight_direct", "api_url", default="", home=home) or "").strip() \
        or _secret("HINDSIGHT_API_URL").strip()


class HindsightStore:
    name = "hindsight"

    def __init__(self, client: Any, bank_id: str) -> None:
        self._client = client
        self.bank_id = bank_id
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mnemosyne-hindsight",
                                          initializer=_thread_loop)

    @classmethod
    def connect(cls, namespace: str, home=None) -> Optional["HindsightStore"]:
        url = configured_url(home)
        if not url:
            logger.warning("mnemosyne: hindsight_direct.api_url is not set; Hindsight backend off")
            return None
        if is_cloud_hindsight(url) and not config.get("hindsight_direct", "allow_cloud",
                                                      default=False, home=home):
            logger.warning("mnemosyne: refusing Hindsight Cloud (%s) without hindsight_direct.allow_cloud", url)
            return None
        try:
            from hindsight_client import Hindsight
            timeout = float(config.get("hindsight_direct", "timeout", default=30.0, home=home))
            client = Hindsight(base_url=url, api_key=_secret("HINDSIGHT_API_KEY") or None,
                               timeout=timeout, user_agent="mnemosyne")
        except Exception as exc:
            logger.warning("mnemosyne: Hindsight client unavailable: %s", exc)
            return None
        store = cls(client, f"mnemosyne-{namespace}")
        try:
            store._configure_bank()
        except Exception as exc:
            logger.warning("mnemosyne: Hindsight bank %s not usable: %s", store.bank_id, exc)
            store.close()
            return None
        return store

    def _call(self, fn: Callable[[], Any]) -> Any:
        return self._worker.submit(fn).result()

    def _configure_bank(self) -> None:
        c, bank = self._client, self.bank_id
        self._call(lambda: c.create_bank(bank, retain_extraction_mode="chunks",
                                         enable_observations=False))
        self._call(lambda: c.update_bank_config(bank, retain_extraction_mode="chunks",
                                                enable_observations=False,
                                                enable_auto_consolidation=False,
                                                store_document_text=True))
        resolved = (self._call(lambda: c.get_bank_config(bank)) or {}).get("config") or {}
        wrong = {k: resolved.get(k) for k, v in _REQUIRED_BANK_CONFIG.items() if resolved.get(k) != v}
        if wrong:
            raise RuntimeError(f"bank config not applied: {wrong}")

    @staticmethod
    def _doc(target: str, entry: str) -> str:
        return f"mn-{target}-{entry}"

    def _tags(self, target: str) -> List[str]:
        return [_TAG, f"target:{target}"]

    def list_ids(self, target: str) -> Set[str]:
        prefix = f"mn-{target}-"
        c, bank, tags = self._client, self.bank_id, self._tags(target)
        ids: Set[str] = set()
        offset = 0
        while True:
            resp = self._call(lambda o=offset: asyncio.get_event_loop().run_until_complete(
                c.documents.list_documents(bank, tags=tags, tags_match="all_strict", limit=_PAGE, offset=o)))
            items = list(getattr(resp, "items", None) or [])
            for item in items:
                doc_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", "")
                if isinstance(doc_id, str) and doc_id.startswith(prefix):
                    ids.add(doc_id[len(prefix):])
            if len(items) < _PAGE:
                return ids
            offset += _PAGE

    def put(self, target: str, entry: str, content: str) -> None:
        c, bank = self._client, self.bank_id
        item = {"content": content, "document_id": self._doc(target, entry), "tags": self._tags(target)}
        self._call(lambda: c.retain_batch(bank, items=[item]))

    def delete(self, target: str, entry: str) -> None:
        c, bank, doc = self._client, self.bank_id, self._doc(target, entry)
        try:
            self._call(lambda: asyncio.get_event_loop().run_until_complete(
                c.documents.delete_document(bank, doc)))
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise

    def purge(self) -> None:
        for target in ("memory", "user"):
            for entry in self.list_ids(target):
                self.delete(target, entry)

    def recall(self, query: str, *, max_tokens: int = 2048) -> List[str]:
        c, bank = self._client, self.bank_id
        resp = self._call(lambda: c.recall(bank, query, types=["world"], max_tokens=max_tokens,
                                           tags=[_TAG], tags_match="all_strict"))
        return [r.text.strip() for r in (getattr(resp, "results", None) or []) if getattr(r, "text", "")]

    def reflect(self, query: str) -> str:
        c, bank = self._client, self.bank_id
        resp = self._call(lambda: c.reflect(bank, query, fact_types=["world"], exclude_mental_models=True,
                                            tags=[_TAG], tags_match="all_strict"))
        return str(getattr(resp, "text", "") or "")

    def close(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)
