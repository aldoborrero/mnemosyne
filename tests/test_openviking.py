"""OpenViking backend: approved writes as files under a per-scope root.

The store must never open or commit an OpenViking session (a commit makes the
server extract memories with its own LLM), and must never read, list or
delete outside its root.
"""
from __future__ import annotations

import json
import sys
import types

import pytest
from conftest import make_provider

from _hermes_user_memory.mnemosyne import openviking_store, policy
from _hermes_user_memory.mnemosyne.openviking_store import OpenVikingStore

ROOT = "viking://user/alice/memories/mnemosyne/abc123/"
FOREIGN = "viking://user/alice/memories/mnemosyne/other/memory/mem_1.md"


class _FakeClient:
    def __init__(self, files=None, extra_results=()):
        self.files = dict(files or {})
        self.extra_results = list(extra_results)
        self.calls: list[tuple] = []

    def post(self, path, payload=None, **kwargs):
        self.calls.append(("POST", path, payload))
        if path == "/api/v1/content/write":
            self.files[payload["uri"]] = payload["content"]
            return {"result": {"uri": payload["uri"]}}
        if path == "/api/v1/search/find":
            hits = [{"uri": u, "score": 0.9, "abstract": ""}
                    for u, c in self.files.items()
                    if u.startswith(payload["target_uri"]) and payload["query"].split()[0].lower() in c.lower()]
            return {"result": {"memories": hits + self.extra_results}}
        raise AssertionError(f"unexpected POST {path}")

    def get(self, path, **kwargs):
        self.calls.append(("GET", path, kwargs.get("params")))
        assert path == "/api/v1/content/read"
        return {"result": {"content": self.files.get(kwargs["params"]["uri"], "")}}

    def delete(self, path, **kwargs):
        self.calls.append(("DELETE", path, kwargs.get("params")))
        self.files.pop(kwargs["params"]["uri"], None)
        return {"result": {}}

    def paths(self):
        return [c[1] for c in self.calls]


def _store(client):
    return OpenVikingStore(client, "alice", "abc123")


def _provider(client, monkeypatch, backends=("openviking",)):
    monkeypatch.setenv("MNEMOSYNE_BACKENDS", ",".join(backends))
    p = make_provider()
    p._backends = list(backends)
    p._honcho = None
    p._hindsight = None
    p._openviking = _store(client)
    return p


@pytest.fixture(autouse=True)
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "plugins" / "mnemosyne").mkdir(parents=True)
    for var in ("MNEMOSYNE_INGEST_MODE", "MNEMOSYNE_SCOPE_MODE", "MNEMOSYNE_BACKENDS",
                "MNEMOSYNE_PREFETCH_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_write_goes_under_scope_root():
    client = _FakeClient()
    uri = _store(client).write("user", "prefers terse replies")
    assert uri.startswith(ROOT + "user/mem_") and uri.endswith(".md")
    assert client.calls[0][2]["mode"] == "create"


def test_find_is_scoped_and_drops_foreign_results():
    client = _FakeClient(files={ROOT + "memory/mem_a.md": "deploy key is in vault"},
                         extra_results=[{"uri": FOREIGN, "score": 1.0, "abstract": "leak"}])
    items = _store(client).find("deploy")
    assert [i["uri"] for i in items] == [ROOT + "memory/mem_a.md"]
    assert client.calls[0][2]["target_uri"] == ROOT


@pytest.mark.parametrize("uri", [FOREIGN, ROOT + "../other/memory/mem_1.md"])
def test_read_and_delete_refuse_outside_root(uri):
    store = _store(_FakeClient())
    with pytest.raises(ValueError):
        store.read(uri)
    with pytest.raises(ValueError):
        store.delete(uri)


def test_delete_matching_is_exact():
    client = _FakeClient(files={ROOT + "memory/mem_a.md": "deploy key is in vault",
                                ROOT + "memory/mem_b.md": "deploy key is in vault, rotated monthly"})
    assert _store(client).delete_matching("deploy key is in vault") == 1
    assert list(client.files) == [ROOT + "memory/mem_b.md"]


def test_connect_needs_health_and_server_user(monkeypatch):
    fake = types.ModuleType("plugins.memory.openviking")
    fake._load_hermes_openviking_config = lambda: {}
    fake._resolve_connection_settings = lambda cfg: {
        "endpoint": "http://127.0.0.1:1933", "api_key": "", "account": "", "user": "", "agent": "hermes"}

    class _Client:
        healthy = True

        def __init__(self, *a, **k):
            pass

        def health(self):
            return _Client.healthy

    fake._VikingClient = _Client
    fake._resolve_user_space = lambda client: "alice"
    monkeypatch.setitem(sys.modules, "plugins.memory.openviking", fake)

    store = OpenVikingStore.connect("abc123")
    assert store is not None and store.root == ROOT
    assert OpenVikingStore.connect(None).root.endswith("/mnemosyne/global/")
    fake._resolve_user_space = lambda client: None
    assert OpenVikingStore.connect("abc123") is None
    fake._resolve_user_space = lambda client: "alice"
    _Client.healthy = False
    assert OpenVikingStore.connect("abc123") is None


# ---------------------------------------------------------------------------
# provider integration
# ---------------------------------------------------------------------------

def test_memory_write_bridge_add_replace_remove(monkeypatch):
    client = _FakeClient()
    p = _provider(client, monkeypatch)
    p.on_memory_write("add", "memory", "deploy key is in vault")
    assert list(client.files.values()) == ["deploy key is in vault"]
    p.on_memory_write("replace", "memory", "deploy key is in the new vault",
                      {"old_text": "deploy key is in vault"})
    assert list(client.files.values()) == ["deploy key is in the new vault"]
    p.on_memory_write("remove", "memory", "deploy key is in the new vault")
    assert client.files == {}


def test_turns_never_reach_openviking_and_no_sessions(monkeypatch):
    client = _FakeClient()
    p = _provider(client, monkeypatch)
    p.sync_turn("user secret", "assistant")
    p.on_session_end([{"role": "user", "content": "x"}])
    p.on_pre_compress([{"role": "user", "content": "x"}])
    p.on_memory_write("add", "memory", "fact")
    assert not [path for path in client.paths() if "session" in path]
    assert [c[1] for c in client.calls] == ["/api/v1/content/write"]


def test_recall_and_read_openviking_only(monkeypatch):
    client = _FakeClient(files={ROOT + "memory/mem_a.md": "deploy key is in vault"},
                         extra_results=[{"uri": FOREIGN, "score": 1.0, "abstract": "leak"}])
    p = _provider(client, monkeypatch)
    names = [s["name"] for s in p.get_tool_schemas()]
    assert set(names) == {"memory_recall", "memory_read"}
    out = json.loads(p.handle_tool_call("memory_recall", {"query": "deploy"}))["result"]
    assert "deploy key is in vault" in out and ROOT in out
    assert "leak" not in out
    assert "error" in json.loads(p.handle_tool_call("memory_read", {"uri": FOREIGN}))
    read = json.loads(p.handle_tool_call("memory_read", {"uri": ROOT + "memory/mem_a.md"}))
    assert read["content"] == "deploy key is in vault"


def test_chat_scope_accepts_openviking(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SCOPE_MODE", "chat")
    monkeypatch.setenv("MNEMOSYNE_BACKENDS", "openviking")
    seen = []
    monkeypatch.setattr(OpenVikingStore, "connect",
                        classmethod(lambda cls, scope: seen.append(scope) or _store(_FakeClient())))
    p = make_provider()
    p._backends = ["openviking"]
    p._honcho = p._hindsight = None
    p._openviking = None
    p.initialize("s1", platform="matrix", chat_id="!a:example.org")
    assert not p._blocked
    assert seen == [policy.scope_slug("matrix:!a:example.org")]


def test_is_available_uses_configuration(monkeypatch):
    p = _provider(_FakeClient(), monkeypatch)
    monkeypatch.setattr(sys.modules[p.__class__.__module__], "_openviking_configured", lambda: False)
    assert not p.is_available()
    monkeypatch.setattr(sys.modules[p.__class__.__module__], "_openviking_configured", lambda: True)
    assert p.is_available()
    assert openviking_store.GLOBAL_SCOPE == "global"
