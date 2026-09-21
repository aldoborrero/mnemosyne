"""ingest.mode=approved_writes: backends mirror only Hermes' approved memory files.

The files (MEMORY.md, USER.md) are the source of truth because Hermes'
/memory approve updates them without notifying providers. These tests drive
the reconciliation, the provider's fail-closed lifecycle and both stores
against recording doubles of their HTTP clients.
"""
from __future__ import annotations

import json
import types

import pytest
from conftest import mnemosyne

from _hermes_user_memory.mnemosyne import approved as approved_mod
from _hermes_user_memory.mnemosyne import cli, memory_files, policy
from _hermes_user_memory.mnemosyne.approved import ApprovedMemoryProvider
from _hermes_user_memory.mnemosyne.hindsight_store import HindsightStore
from _hermes_user_memory.mnemosyne.openviking_store import OpenVikingStore
from _hermes_user_memory.mnemosyne.reconcile import reconcile

DELIM = "\n§\n"
ROOT = "viking://user/alice/memories/mnemosyne/ns1/"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "memories").mkdir()
    (tmp_path / "plugins" / "mnemosyne").mkdir(parents=True)
    for var in ("MNEMOSYNE_INGEST_MODE", "MNEMOSYNE_APPROVED_BACKENDS", "MNEMOSYNE_APPROVED_PREFETCH",
                "MNEMOSYNE_APPROVED_REFLECT", "MNEMOSYNE_HINDSIGHT_URL", "HINDSIGHT_API_URL"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _write(home, target, entries):
    name = memory_files.TARGETS[target]
    (home / "memories" / name).write_text(DELIM.join(entries), encoding="utf-8")


class _MemStore:
    """In-memory EntryStore with call recording."""

    def __init__(self, name="fake", fail_list=False):
        self.name = name
        self.data = {"memory": {}, "user": {}}
        self.fail_list = fail_list
        self.calls = []

    def list_ids(self, target):
        if self.fail_list:
            raise RuntimeError("503")
        return set(self.data[target])

    def put(self, target, entry, content):
        self.calls.append(("put", target, entry))
        self.data[target][entry] = content

    def delete(self, target, entry):
        self.calls.append(("delete", target, entry))
        self.data[target].pop(entry, None)

    def texts(self, target):
        return sorted(self.data[target].values())


# ---------------------------------------------------------------------------
# memory files and reconcile
# ---------------------------------------------------------------------------

def test_read_distinguishes_missing_from_empty(home):
    assert memory_files.read_target(str(home), "user").entries is None
    assert memory_files.read_target(str(home), "user").missing
    _write(home, "user", [])
    assert memory_files.read_target(str(home), "user").entries == []
    _write(home, "memory", ["a", " b ", "a", ""])
    assert memory_files.read_target(str(home), "memory").entries == ["a", "b"]


def test_reconcile_creates_and_deletes_by_content_id(home):
    store = _MemStore()
    _write(home, "memory", ["deploy key is in vault", "prod is eu-west"])
    reconcile(store, memory_files.read_all(str(home)))
    assert store.texts("memory") == ["deploy key is in vault", "prod is eu-west"]
    # Hermes replaced by substring: the file now holds a different set.
    _write(home, "memory", ["deploy key is in the new vault", "prod is eu-west"])
    out = reconcile(store, memory_files.read_all(str(home)))
    assert store.texts("memory") == ["deploy key is in the new vault", "prod is eu-west"]
    assert (out.created, out.deleted) == (1, 1)


def test_reconcile_empty_file_deletes_missing_file_keeps(home):
    store = _MemStore()
    store.put("user", policy.entry_id("likes tea"), "likes tea")
    store.put("memory", policy.entry_id("old"), "old")
    _write(home, "memory", [])
    out = reconcile(store, memory_files.read_all(str(home)))
    assert store.texts("memory") == []
    assert store.texts("user") == ["likes tea"]
    assert out.kept_on_unreadable == ["user"]


def test_reconcile_skips_a_backend_whose_listing_fails(home):
    store = _MemStore(fail_list=True)
    _write(home, "memory", ["x"])
    out = reconcile(store, memory_files.read_all(str(home)))
    assert store.calls == [] and out.errors


# ---------------------------------------------------------------------------
# provider lifecycle
# ---------------------------------------------------------------------------

def _provider(home, monkeypatch, *, backends="openviking", store=None, **env):
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    monkeypatch.setenv("MNEMOSYNE_APPROVED_BACKENDS", backends)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    store = store if store is not None else _MemStore("openviking")
    store.owns = lambda uri: isinstance(uri, str) and uri.startswith(ROOT)
    store.find = lambda q, limit=10: [{"uri": ROOT + "memory/mem_" + k + ".md", "score": 1.0, "content": v}
                                      for k, v in store.data["memory"].items() if q.split()[0] in v]
    store.read = lambda uri: "full text"
    store.close = lambda: None
    monkeypatch.setattr(OpenVikingStore, "connect", classmethod(lambda cls, ns: store))
    return ApprovedMemoryProvider(), store


def _init(p, home, **kw):
    p.initialize("s1", hermes_home=str(home), agent_identity="team-a", platform="matrix",
                 agent_context=kw.pop("agent_context", "primary"), **kw)
    assert p.flush()


def test_starts_disabled_until_initialize(home, monkeypatch):
    p, _ = _provider(home, monkeypatch)
    assert p._blocked and p.get_tool_schemas() == [] and p.system_prompt_block() == ""
    assert "error" in json.loads(p.handle_tool_call("memory_recall", {"query": "x"}))


def test_initialize_mirrors_the_files(home, monkeypatch):
    _write(home, "memory", ["deploy key is in vault"])
    _write(home, "user", ["prefers terse replies"])
    p, store = _provider(home, monkeypatch)
    _init(p, home)
    assert not p._blocked
    assert store.texts("memory") == ["deploy key is in vault"]
    assert store.texts("user") == ["prefers terse replies"]


def test_memory_approve_is_picked_up_at_turn_start(home, monkeypatch):
    _write(home, "memory", ["a"])
    p, store = _provider(home, monkeypatch)
    _init(p, home)
    # /memory approve rewrites the file and never calls the provider.
    _write(home, "memory", ["a", "approved later"])
    p.on_turn_start(2, "hi")
    assert p.flush()
    assert store.texts("memory") == ["a", "approved later"]


def test_on_memory_write_triggers_reconcile(home, monkeypatch):
    p, store = _provider(home, monkeypatch)
    _init(p, home)
    _write(home, "memory", ["new fact"])
    p.on_memory_write("add", "memory", "IGNORED payload text")
    assert p.flush()
    assert store.texts("memory") == ["new fact"]


def test_turns_and_hooks_never_write(home, monkeypatch):
    p, store = _provider(home, monkeypatch)
    _init(p, home)
    store.calls.clear()
    p.sync_turn("secret user turn", "assistant")
    p.on_session_end([{"role": "user", "content": "x"}])
    assert p.on_pre_compress([{"role": "user", "content": "x"}]) == ""
    p.on_delegation("task", "result")
    p.queue_prefetch("q")
    assert p.flush() and store.calls == []


def test_cron_context_reads_but_never_reconciles(home, monkeypatch):
    _write(home, "memory", ["a"])
    p, store = _provider(home, monkeypatch)
    _init(p, home, agent_context="cron")
    assert store.calls == []
    assert [s["name"] for s in p.get_tool_schemas()] == ["memory_recall", "memory_read"]


@pytest.mark.parametrize("backends", ["honcho", "openviking,honcho"])
def test_unsupported_backends_fail_closed(home, monkeypatch, backends):
    p, _ = _provider(home, monkeypatch, backends=backends)
    _init(p, home)
    assert p._blocked


def test_empty_backends_fail_closed(home, monkeypatch):
    p, _ = _provider(home, monkeypatch)
    monkeypatch.delenv("MNEMOSYNE_APPROVED_BACKENDS")
    (home / "plugins" / "mnemosyne" / "config.json").write_text(json.dumps({"approved": {"backends": []}}))
    _init(p, home)
    assert p._blocked and "empty" in p._disabled_reason


def test_no_hermes_home_fails_closed(home, monkeypatch):
    p, _ = _provider(home, monkeypatch)
    p.initialize("s1", agent_identity="team-a")
    assert p._blocked


def test_initialize_exception_fails_closed(home, monkeypatch):
    p, _ = _provider(home, monkeypatch)
    monkeypatch.setattr(OpenVikingStore, "connect", classmethod(lambda cls, ns: 1 / 0))
    _init(p, home)
    assert p._blocked and "initialize failed" in p._disabled_reason


def test_unreachable_backend_fails_closed(home, monkeypatch):
    p, _ = _provider(home, monkeypatch)
    monkeypatch.setattr(OpenVikingStore, "connect", classmethod(lambda cls, ns: None))
    _init(p, home)
    assert p._blocked


def test_namespace_is_per_profile(home):
    a = policy.namespace("team-a", str(home))
    assert a == policy.namespace("team-a", str(home))
    assert a != policy.namespace("team-b", str(home))
    assert a != policy.namespace("team-a", str(home / "other"))


def test_tools_recall_read_and_reflect_opt_in(home, monkeypatch):
    _write(home, "memory", ["deploy key is in vault"])
    p, store = _provider(home, monkeypatch)
    _init(p, home)
    out = json.loads(p.handle_tool_call("memory_recall", {"query": "deploy"}))["result"]
    assert "deploy key is in vault" in out and ROOT in out
    assert "error" in json.loads(p.handle_tool_call("memory_read", {"uri": "viking://user/alice/x.md"}))
    assert json.loads(p.handle_tool_call("memory_read", {"uri": ROOT + "memory/mem_x.md"}))["content"] == "full text"
    assert "error" in json.loads(p.handle_tool_call("memory_reflect", {"query": "x"}))
    assert "memory_forget" not in [s["name"] for s in p.get_tool_schemas()]
    assert "built-in `memory` tool" in p.system_prompt_block()


def test_prefetch_is_off_by_default(home, monkeypatch):
    _write(home, "memory", ["deploy key is in vault"])
    p, _ = _provider(home, monkeypatch)
    _init(p, home)
    assert p.prefetch("deploy") == ""
    p2, _ = _provider(home, monkeypatch, MNEMOSYNE_APPROVED_PREFETCH="true")
    _init(p2, home)
    assert "deploy key is in vault" in p2.prefetch("deploy")


def test_purge_and_disappeared_file_warning(home, monkeypatch, caplog):
    _write(home, "user", ["likes tea"])
    p, store = _provider(home, monkeypatch)
    purged = []
    store.purge = lambda: purged.append(True)
    _init(p, home)
    (home / "memories" / "USER.md").unlink()
    p.reconcile_now()
    assert store.texts("user") == ["likes tea"]
    assert "disappeared" in caplog.text
    assert p.purge() == {"openviking": "purged"} and purged


def test_register_selects_provider_by_mode(home, monkeypatch):
    got = []
    ctx = types.SimpleNamespace(register_memory_provider=got.append)
    mnemosyne.register(ctx)
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    mnemosyne.register(ctx)
    assert type(got[0]).__name__ == "MnemosyneMemoryProvider"
    assert isinstance(got[1], ApprovedMemoryProvider)


# ---------------------------------------------------------------------------
# OpenViking store
# ---------------------------------------------------------------------------

class _HTTPError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class _VikingClient:
    def __init__(self, files=None, extra=()):
        self.files = dict(files or {})
        self.extra = list(extra)
        self.calls = []

    def post(self, path, payload=None, **kw):
        self.calls.append(("POST", path, payload))
        if path == "/api/v1/content/write":
            if payload["uri"] in self.files:
                raise _HTTPError(409)
            self.files[payload["uri"]] = payload["content"]
            return {"result": {}}
        if path == "/api/v1/search/find":
            hits = [{"uri": u, "score": 0.9, "content": c} for u, c in self.files.items()
                    if u.startswith(payload["target_uri"]) and payload["query"] in c]
            return {"result": {"memories": hits + self.extra}}
        raise AssertionError(path)

    def get(self, path, params=None, **kw):
        self.calls.append(("GET", path, params))
        if path == "/api/v1/fs/ls":
            under = [u for u in self.files if u.startswith(params["uri"])]
            if not under:
                raise _HTTPError(404)
            page = under[params["offset"]:params["offset"] + params["node_limit"]]
            return {"result": [{"uri": u} for u in page] + [{"uri": params["uri"] + ".overview.md"}]}
        if path == "/api/v1/content/read":
            return {"result": self.files[params["uri"]]}
        raise AssertionError(path)

    def delete(self, path, params=None, **kw):
        self.calls.append(("DELETE", path, params))
        if params["uri"] not in self.files and not params.get("recursive"):
            raise _HTTPError(404)
        for u in [u for u in self.files if u == params["uri"] or (params.get("recursive") and u.startswith(params["uri"]))]:
            del self.files[u]
        return {"result": {}}


def test_openviking_store_ids_writes_and_scope():
    client = _VikingClient(extra=[{"uri": "viking://user/alice/memories/other/x.md", "score": 1, "content": "leak"}])
    store = OpenVikingStore(client, "alice", "ns1")
    assert store.list_ids("memory") == set()          # 404: never written
    store.put("memory", "abc", "deploy key")
    store.put("memory", "abc", "deploy key")          # 409 is success
    write = [c for c in client.calls if c[1] == "/api/v1/content/write"][0][2]
    assert write["wait"] is True and write["processing_mode"] == "vectors_only" and write["mode"] == "create"
    assert store.list_ids("memory") == {"abc"}
    found = store.find("deploy")
    assert [f["uri"] for f in found] == [ROOT + "memory/mem_abc.md"] and found[0]["content"] == "deploy key"
    find = [c for c in client.calls if c[1] == "/api/v1/search/find"][0][2]
    assert find["read_content"] is True and find["target_uri"] == ROOT and find["score_threshold"]
    store.delete("memory", "abc")
    store.delete("memory", "abc")                     # 404 is success
    assert store.list_ids("memory") == set()
    assert not any("session" in c[1] for c in client.calls)


def test_openviking_list_paginates(monkeypatch):
    from _hermes_user_memory.mnemosyne import openviking_store
    monkeypatch.setattr(openviking_store, "_PAGE", 2)
    client = _VikingClient(files={ROOT + f"user/mem_{i}.md": "x" for i in range(5)})
    assert OpenVikingStore(client, "alice", "ns1").list_ids("user") == {str(i) for i in range(5)}


@pytest.mark.parametrize("uri", ["viking://user/alice/x.md", ROOT, ROOT + "../x.md", ROOT + "a//b.md", ROOT + "./a.md"])
def test_openviking_owns_rejects(uri):
    assert not OpenVikingStore(_VikingClient(), "alice", "ns1").owns(uri)


def test_openviking_purge_is_recursive_on_the_root():
    client = _VikingClient(files={ROOT + "memory/mem_a.md": "a", "viking://user/alice/other.md": "keep"})
    OpenVikingStore(client, "alice", "ns1").purge()
    assert list(client.files) == ["viking://user/alice/other.md"]


# ---------------------------------------------------------------------------
# Hindsight store
# ---------------------------------------------------------------------------

class _Docs:
    def __init__(self, owner):
        self.o = owner

    async def list_documents(self, bank, tags=None, tags_match=None, limit=100, offset=0):
        self.o.calls.append(("list", bank, tuple(tags), tags_match))
        ids = sorted(d for d, t in self.o.docs.items() if set(tags) <= set(t["tags"]))
        return types.SimpleNamespace(items=[types.SimpleNamespace(id=i) for i in ids[offset:offset + limit]])

    async def delete_document(self, bank, doc):
        self.o.calls.append(("delete", bank, doc))
        if doc not in self.o.docs:
            err = RuntimeError("404")
            err.status = 404
            raise err
        del self.o.docs[doc]


class _HindsightClient:
    def __init__(self, config=None):
        self.calls = []
        self.docs = {}
        self.config = config if config is not None else {
            "retain_extraction_mode": "chunks", "enable_observations": False, "enable_auto_consolidation": False}
        self.documents = _Docs(self)

    def create_bank(self, bank, **kw):
        self.calls.append(("create_bank", bank, kw))

    def update_bank_config(self, bank, **kw):
        self.calls.append(("update_bank_config", bank, kw))

    def get_bank_config(self, bank):
        return {"bank_id": bank, "config": self.config}

    def retain_batch(self, bank, items):
        self.calls.append(("retain", bank, items))
        for it in items:
            self.docs[it["document_id"]] = {"tags": it["tags"], "content": it["content"]}

    def recall(self, bank, query, **kw):
        self.calls.append(("recall", bank, kw))
        return types.SimpleNamespace(results=[types.SimpleNamespace(text=d["content"]) for d in self.docs.values()])

    def reflect(self, bank, query, **kw):
        self.calls.append(("reflect", bank, kw))
        return types.SimpleNamespace(text="answer")


def _hindsight(monkeypatch, client, url="http://127.0.0.1:8888", **cfg_env):
    fake = types.ModuleType("hindsight_client")
    fake.Hindsight = lambda **kw: client
    monkeypatch.setitem(__import__("sys").modules, "hindsight_client", fake)
    monkeypatch.setenv("MNEMOSYNE_HINDSIGHT_URL", url)
    return HindsightStore.connect("ns1")


def test_hindsight_bank_is_configured_verbatim_without_observations(home, monkeypatch):
    client = _HindsightClient()
    store = _hindsight(monkeypatch, client)
    assert store is not None and store.bank_id == "mnemosyne-ns1"
    create = [c for c in client.calls if c[0] == "create_bank"][0][2]
    update = [c for c in client.calls if c[0] == "update_bank_config"][0][2]
    assert create["retain_extraction_mode"] == "chunks" and create["enable_observations"] is False
    assert update["enable_auto_consolidation"] is False and update["store_document_text"] is True
    store.close()


def test_hindsight_refuses_unverified_config_and_cloud(home, monkeypatch):
    assert _hindsight(monkeypatch, _HindsightClient(config={"retain_extraction_mode": "concise"})) is None
    assert _hindsight(monkeypatch, _HindsightClient(), url="https://api.hindsight.vectorize.io") is None
    monkeypatch.setenv("MNEMOSYNE_HINDSIGHT_URL", "")
    assert HindsightStore.connect("ns1") is None


def test_hindsight_documents_by_content_id(home, monkeypatch):
    client = _HindsightClient()
    store = _hindsight(monkeypatch, client)
    store.put("user", "abc", "likes tea")
    retain = [c for c in client.calls if c[0] == "retain"][0][2][0]
    assert retain["document_id"] == "mn-user-abc" and retain["tags"] == ["mnemosyne", "target:user"]
    assert store.list_ids("user") == {"abc"} and store.list_ids("memory") == set()
    store.delete("user", "abc")
    store.delete("user", "abc")                       # 404 is success
    assert client.docs == {}
    store.recall("tea")
    store.reflect("tea")
    recall = [c for c in client.calls if c[0] == "recall"][0][2]
    reflect = [c for c in client.calls if c[0] == "reflect"][0][2]
    assert recall["types"] == ["world"] and recall["tags"] == ["mnemosyne"]
    assert reflect["fact_types"] == ["world"] and reflect["exclude_mental_models"] is True
    store.close()


def test_hindsight_reflect_needs_opt_in(home, monkeypatch):
    client = _HindsightClient()
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    monkeypatch.setenv("MNEMOSYNE_APPROVED_BACKENDS", "hindsight")
    store = _hindsight(monkeypatch, client)
    monkeypatch.setattr(HindsightStore, "connect", classmethod(lambda cls, ns, home=None: store))
    p = ApprovedMemoryProvider()
    _init(p, home)
    assert [s["name"] for s in p.get_tool_schemas()] == ["memory_recall"]
    monkeypatch.setenv("MNEMOSYNE_APPROVED_REFLECT", "true")
    p2 = ApprovedMemoryProvider()
    _init(p2, home)
    assert "memory_reflect" in [s["name"] for s in p2.get_tool_schemas()]
    assert json.loads(p2.handle_tool_call("memory_reflect", {"query": "x"}))["result"] == "answer"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_gates(home, monkeypatch, capsys):
    ns = types.SimpleNamespace
    assert cli.mnemosyne_command(ns(action="reconcile")) == 2        # not in approved mode
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    assert cli.mnemosyne_command(ns(action="purge", yes=False)) == 2
    assert cli._cmd_import(ns(days=None, min_turns=None)) == 2
    assert cli._cmd_forget(ns(query="x", yes=True, max_items=5)) == 2
    assert "Refusing" in capsys.readouterr().err
    assert approved_mod is not None
