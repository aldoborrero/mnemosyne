"""OpenViking backend: plain memory files under a per-scope root.

Mnemosyne talks to OpenViking's REST API through the Hermes plugin's HTTP
client instead of instantiating ``OpenVikingMemoryProvider``. That provider
uploads every turn and commits sessions — on session end, at startup for
sessions left pending, and from an ``atexit`` hook — and a commit makes the
OpenViking server extract memories with its own LLM. None of that respects
Hermes' write approval, so this store never touches ``/sessions``: it only
writes, finds, reads and deletes files below its root.

Layout: ``viking://user/<space>/memories/mnemosyne/<scope>/<target>/mem_<id>.md``
where ``<scope>`` is the chat scope slug (or ``global``) and ``<target>`` is
the built-in memory target (``memory`` / ``user``). The server stores the file
as written; embeddings and directory overviews it derives are indexes only.
"""

from __future__ import annotations

import importlib
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = "global"
_TARGETS = ("memory", "user")


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
        return isinstance(uri, str) and uri.startswith(self.root) and ".." not in uri

    def write(self, target: str, content: str) -> str:
        target = target if target in _TARGETS else "memory"
        uri = f"{self.root}{target}/mem_{uuid.uuid4().hex[:12]}.md"
        self._client.post("/api/v1/content/write", {"uri": uri, "content": content, "mode": "create"})
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

    def delete(self, uri: str) -> None:
        if not self.owns(uri):
            raise ValueError(f"{uri} is outside this memory scope")
        self._client.delete("/api/v1/fs", params={"uri": uri, "recursive": False})

    def delete_matching(self, content: str) -> int:
        """Delete the files whose content is exactly ``content``; returns how many."""
        wanted = content.strip()
        removed = 0
        for item in self.find(wanted, limit=20):
            if item["uri"].endswith(".md") and self.read(item["uri"]).strip() == wanted:
                self.delete(item["uri"])
                removed += 1
        return removed
