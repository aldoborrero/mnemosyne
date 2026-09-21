"""OpenViking backend: plain memory files under a per-scope root.

Mnemosyne talks to OpenViking's REST API through the Hermes plugin's HTTP
client instead of instantiating ``OpenVikingMemoryProvider``. That provider
uploads every turn and commits sessions — on session end, at startup for
sessions left pending, and from an ``atexit`` hook — and a commit makes the
OpenViking server extract memories with its own LLM. None of that respects
Hermes' write approval, so this store never touches ``/sessions``: it only
writes, finds, reads and deletes files below its root.

Layout: ``viking://user/<space>/memories/mnemosyne/<scope>/<target>/mem_<hash>.md``
where ``<scope>`` is the chat scope slug (or ``global``), ``<target>`` is the
built-in memory target (``memory`` / ``user``) and ``<hash>`` is derived from
the content, so writing the same entry twice yields one file. The server
stores the file as written; embeddings and directory overviews it derives are
indexes only.

Hermes identifies the entry to replace or remove by a unique substring
(``old_text``), not by its full text, so edits list the target directory and
match on containment, as Hermes' own store does.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = "global"
_TARGETS = ("memory", "user")
_READ_WORKERS = 4


def _ov_module():
    return importlib.import_module("plugins.memory.openviking")


def is_configured() -> bool:
    """Cheap, network-free check that an OpenViking endpoint is configured."""
    try:
        ov = _ov_module()
        settings = ov._resolve_connection_settings(ov._load_hermes_openviking_config())
    except Exception as exc:
        logger.debug("mnemosyne: OpenViking not configured: %s", exc)
        return False
    return bool(settings.get("endpoint"))


class OpenVikingStore:
    def __init__(self, client: Any, user_space: str, scope: str) -> None:
        self._client = client
        self.root = f"viking://user/{user_space}/memories/mnemosyne/{scope}/"
        # Own pool: recall may itself run on the provider's executor.
        self._readers = ThreadPoolExecutor(max_workers=_READ_WORKERS,
                                           thread_name_prefix="mnemosyne-viking")

    @classmethod
    def connect(cls, scope: Optional[str]) -> Optional["OpenVikingStore"]:
        """Build a store for ``scope`` (a slug, or None for the global root).

        Returns None when the endpoint is unset, unhealthy or the user space
        cannot be resolved; callers treat that as the backend being down."""
        try:
            ov = _ov_module()
            settings = ov._resolve_connection_settings(ov._load_hermes_openviking_config())
            if not settings.get("endpoint"):
                return None
            client = ov._VikingClient(
                settings["endpoint"], settings.get("api_key", ""),
                account=settings.get("account") or None,
                user=settings.get("user") or None,
                agent=settings.get("agent"),
            )
            if not client.health():
                logger.warning("mnemosyne: OpenViking at %s is not healthy", settings["endpoint"])
                return None
            # The server-asserted user, so URIs never land in another tenant's space.
            user_space = ov._resolve_user_space(client)
        except Exception as exc:
            logger.warning("mnemosyne: OpenViking connection failed: %s", exc)
            return None
        if not user_space:
            logger.warning("mnemosyne: OpenViking did not report a user space")
            return None
        return cls(client, user_space, scope or GLOBAL_SCOPE)

    def owns(self, uri: str) -> bool:
        if not isinstance(uri, str) or not uri.startswith(self.root):
            return False
        rest = uri[len(self.root):]
        return bool(rest) and all(part not in ("", ".", "..") for part in rest.split("/"))

    def uri_for(self, target: str, content: str) -> str:
        target = target if target in _TARGETS else "memory"
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]
        return f"{self.root}{target}/mem_{digest}.md"

    def write(self, target: str, content: str) -> str:
        uri = self.uri_for(target, content)
        try:
            self._client.post("/api/v1/content/write",
                              {"uri": uri, "content": content.strip(), "mode": "create"})
        except Exception:
            # mode=create refuses an existing file; the same content is already stored.
            if self._read_or_none(uri) == content.strip():
                return uri
            raise
        return uri

    def find(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Search below the root; results outside it are dropped even if the server returns them."""
        resp = self._client.post("/api/v1/search/find",
                                 {"query": query, "target_uri": self.root, "limit": limit})
        result = (resp or {}).get("result") or {}
        items = []
        for kind in ("memories", "resources"):
            for item in result.get(kind) or []:
                uri = item.get("uri", "")
                if self.owns(uri):
                    items.append({"uri": uri, "score": item.get("score") or 0.0,
                                  "abstract": item.get("abstract", "")})
        items.sort(key=lambda it: it["score"], reverse=True)
        return items

    def read(self, uri: str) -> str:
        if not self.owns(uri):
            raise ValueError(f"{uri} is outside this memory scope")
        resp = self._client.get("/api/v1/content/read", params={"uri": uri})
        result = (resp or {}).get("result", resp)
        if isinstance(result, dict):
            return result.get("content") or result.get("text") or ""
        return result if isinstance(result, str) else ""

    def read_many(self, uris: Iterable[str]) -> List[str]:
        """Read several files in parallel; a failed read yields an empty string."""
        return list(self._readers.map(lambda u: self._read_or_none(u) or "", list(uris)))

    def _read_or_none(self, uri: str) -> Optional[str]:
        try:
            return self.read(uri).strip()
        except Exception as exc:
            logger.debug("mnemosyne: OpenViking read of %s failed: %s", uri, exc)
            return None

    def list_files(self, target: str) -> List[str]:
        target = target if target in _TARGETS else "memory"
        try:
            resp = self._client.get("/api/v1/fs/ls", params={"uri": f"{self.root}{target}/"})
        except Exception as exc:
            # A target directory that was never written does not exist yet.
            logger.debug("mnemosyne: OpenViking ls of %s failed: %s", target, exc)
            return []
        result = (resp or {}).get("result", resp)
        if isinstance(result, dict):
            result = result.get("entries") or result.get("items") or result.get("children") or []
        uris = []
        for entry in result if isinstance(result, list) else []:
            uri = entry.get("uri", "") if isinstance(entry, dict) else ""
            name = uri.rsplit("/", 1)[-1]
            # Skip the server's generated .abstract.md / .overview.md sidecars.
            if self.owns(uri) and name.endswith(".md") and not name.startswith("."):
                uris.append(uri)
        return uris

    def delete(self, uri: str) -> None:
        if not self.owns(uri):
            raise ValueError(f"{uri} is outside this memory scope")
        self._client.delete("/api/v1/fs", params={"uri": uri, "recursive": False})

    def delete_containing(self, target: str, old_text: str, *, keep: Optional[str] = None) -> int:
        """Delete the one file of ``target`` whose content contains ``old_text``.

        Returns 1 when a file was deleted, 0 when none or several matched
        (several is logged: Hermes refuses ambiguous edits, so it means this
        store and the built-in one disagree)."""
        needle = old_text.strip()
        if not needle:
            return 0
        uris = [u for u in self.list_files(target) if u != keep]
        matches = [u for u, text in zip(uris, self.read_many(uris)) if needle in text]
        if len(matches) > 1:
            logger.warning("mnemosyne: %d OpenViking memories contain %r; none deleted",
                           len(matches), needle[:80])
            return 0
        if matches:
            self.delete(matches[0])
        return len(matches)

    def close(self) -> None:
        self._readers.shutdown(wait=False)
