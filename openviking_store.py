"""OpenViking backend: one plain memory file per approved entry.

Mnemosyne talks to OpenViking's REST API through the Hermes plugin's HTTP
client instead of instantiating ``OpenVikingMemoryProvider``. That provider
uploads every turn and commits sessions — on session end, at startup for
sessions left pending, and from an ``atexit`` hook — and a commit makes the
OpenViking server extract memories with its own LLM. This store never touches
``/sessions``: it only writes, finds, reads, lists and deletes files below
its root.

Layout: ``viking://user/<space>/memories/mnemosyne/<ns>/<target>/mem_<id>.md``
where ``<ns>`` is the profile namespace, ``<target>`` is ``memory`` or
``user`` and ``<id>`` is the entry's content hash. Writes under ``memories/``
run no LLM on the server: the file is embedded, and a directory overview is
generated only for registered memory types, which ``mnemosyne`` is not. The
server does linkify bare ``viking://`` URIs inside the text and keeps link
metadata in a trailer that normal reads strip.

The root is a client-side boundary only: the server lets any caller with the
same user's key read the whole user space. Separate profiles that must not
see each other need separate OpenViking users (a key per profile ``.env``).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_TARGETS = ("memory", "user")
_PAGE = 500
_WRITE_TIMEOUT_S = 30.0
_SCORE_THRESHOLD = 0.15


def _ov_module():
    return importlib.import_module("plugins.memory.openviking")


def _status(exc: Exception) -> Optional[int]:
    return getattr(exc, "status_code", None)


def is_configured() -> bool:
    """Cheap, network-free check that the OpenViking plugin and settings load."""
    try:
        ov = _ov_module()
        settings = ov._resolve_connection_settings(ov._load_hermes_openviking_config())
    except Exception as exc:
        logger.debug("mnemosyne: OpenViking not configured: %s", exc)
        return False
    return bool(settings.get("endpoint"))


class OpenVikingStore:
    name = "openviking"

    def __init__(self, client: Any, user_space: str, namespace: str) -> None:
        self._client = client
        self.root = f"viking://user/{user_space}/memories/mnemosyne/{namespace}/"

    @classmethod
    def connect(cls, namespace: str) -> Optional["OpenVikingStore"]:
        """Store for one profile namespace, or None when the server is unusable."""
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
        return cls(client, user_space, namespace)

    # -- ids -------------------------------------------------------------------

    def _dir(self, target: str) -> str:
        if target not in _TARGETS:
            raise ValueError(f"unknown memory target {target!r}")
        return f"{self.root}{target}/"

    def uri(self, target: str, entry: str) -> str:
        return f"{self._dir(target)}mem_{entry}.md"

    def owns(self, uri: str) -> bool:
        if not isinstance(uri, str) or not uri.startswith(self.root):
            return False
        rest = uri[len(self.root):]
        return bool(rest) and all(part not in ("", ".", "..") for part in rest.split("/"))

    def list_ids(self, target: str) -> Set[str]:
        ids: Set[str] = set()
        offset = 0
        while True:
            try:
                resp = self._client.get("/api/v1/fs/ls", params={
                    "uri": self._dir(target), "node_limit": _PAGE, "offset": offset})
            except Exception as exc:
                if _status(exc) == 404 and offset == 0:
                    return ids  # never written
                raise
            result = (resp or {}).get("result", resp)
            if isinstance(result, dict):
                result = result.get("entries") or result.get("items") or result.get("children") or []
            entries = result if isinstance(result, list) else []
            for entry in entries:
                uri = entry.get("uri", "") if isinstance(entry, dict) else ""
                name = uri.rsplit("/", 1)[-1]
                if self.owns(uri) and name.startswith("mem_") and name.endswith(".md"):
                    ids.add(name[len("mem_"):-len(".md")])
            if len(entries) < _PAGE:
                return ids
            offset += _PAGE

    # -- writes ----------------------------------------------------------------

    def put(self, target: str, entry: str, content: str) -> None:
        try:
            self._client.post("/api/v1/content/write", {
                "uri": self.uri(target, entry), "content": content, "mode": "create",
                "processing_mode": "vectors_only", "wait": True, "timeout": _WRITE_TIMEOUT_S})
        except Exception as exc:
            if _status(exc) != 409:  # already stored under its content hash
                raise

    def delete(self, target: str, entry: str) -> None:
        try:
            self._client.delete("/api/v1/fs", params={"uri": self.uri(target, entry), "recursive": False})
        except Exception as exc:
            if _status(exc) != 404:
                raise

    def purge(self) -> None:
        """Delete this namespace's whole subtree."""
        try:
            self._client.delete("/api/v1/fs", params={"uri": self.root, "recursive": True})
        except Exception as exc:
            if _status(exc) != 404:
                raise

    # -- reads -----------------------------------------------------------------

    def find(self, query: str, *, limit: int = 10) -> List[Dict[str, Any]]:
        """Search below the root; results outside it are dropped even if the server returns them."""
        resp = self._client.post("/api/v1/search/find", {
            "query": query, "target_uri": self.root, "limit": limit,
            "score_threshold": _SCORE_THRESHOLD, "context_type": "memory", "read_content": True})
        result = (resp or {}).get("result") or {}
        items = []
        for item in result.get("memories") or []:
            uri = item.get("uri", "")
            if self.owns(uri):
                items.append({"uri": uri, "score": item.get("score") or 0.0,
                              "content": (item.get("content") or "").strip()})
        items.sort(key=lambda it: it["score"], reverse=True)
        return items

    def read(self, uri: str) -> str:
        if not self.owns(uri):
            raise ValueError(f"{uri} is outside this memory namespace")
        resp = self._client.get("/api/v1/content/read", params={"uri": uri})
        result = (resp or {}).get("result", resp)
        if isinstance(result, dict):
            return result.get("content") or result.get("text") or ""
        return result if isinstance(result, str) else ""

    def close(self) -> None:
        pass
