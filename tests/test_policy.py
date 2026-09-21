"""Ingest, scope and backend policy.

Under ``ingest.mode=approved_writes`` nothing but committed built-in memory
writes may reach the backends; under ``scope.mode=chat`` memory is keyed by
the gateway chat and the provider fails closed when it cannot be.
"""
from __future__ import annotations

import json

import pytest
from conftest import make_provider, mnemosyne

from _hermes_user_memory.mnemosyne import cli, policy


class _RecordingProvider:
    """Inner provider double that records every hook and tool call."""

    def __init__(self, bank_id: str = "hermes"):
        self._bank_id = bank_id
        self.events: list[tuple] = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.events.append(("initialize", session_id))

    def sync_turn(self, user, assistant, *, session_id=""):
        self.events.append(("sync_turn", user))

    def on_memory_write(self, action, target, content, metadata=None):
        self.events.append(("on_memory_write", action, content))

    def on_session_end(self, messages):
        self.events.append(("on_session_end",))

    def on_pre_compress(self, messages):
        self.events.append(("on_pre_compress",))
        return ""

    def on_delegation(self, task, result, **kwargs):
        self.events.append(("on_delegation",))

    def on_turn_start(self, turn_number, message, **kwargs):
        self.events.append(("on_turn_start",))

    def queue_prefetch(self, query, *, session_id=""):
        self.events.append(("queue_prefetch",))

    def handle_tool_call(self, name, args, **kwargs):
        self.events.append(("tool", name, args))
        return json.dumps({"result": "Fact 1"})

    def kinds(self):
        return [e[0] for e in self.events]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "sessions").mkdir()
    (tmp_path / "plugins" / "mnemosyne").mkdir(parents=True)
    for var in ("MNEMOSYNE_INGEST_MODE", "MNEMOSYNE_SCOPE_MODE", "MNEMOSYNE_BACKENDS",
                "MNEMOSYNE_PREFETCH_ENABLED"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _provider(backends=("honcho", "hindsight")):
    p = make_provider()
    p._backends = list(backends)
    p._honcho = _RecordingProvider() if "honcho" in backends else None
    p._hindsight = _RecordingProvider() if "hindsight" in backends else None
    return p


def _schema_names(p):
    return [s["name"] for s in p.get_tool_schemas()]


# ---------------------------------------------------------------------------
# ingest.mode=approved_writes
# ---------------------------------------------------------------------------

def test_approved_writes_blocks_turn_ingest(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    p = _provider()
    p.sync_turn("secret user turn", "assistant turn")
    p.on_turn_start(1, "hello")
    p.on_session_end([{"role": "user", "content": "x"}])
    p.on_pre_compress([{"role": "user", "content": "x"}])
    p.on_delegation("task", "result")
    assert p._honcho.events == []
    assert p._hindsight.events == []


def test_approved_writes_keeps_memory_write_bridge(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    p = _provider(backends=("hindsight",))
    p.on_memory_write("add", "memory", "the deploy key lives in vault")
    kinds = p._hindsight.kinds()
    assert "on_memory_write" in kinds
    retains = [e for e in p._hindsight.events if e[0] == "tool" and e[1] == "hindsight_retain"]
    assert retains and "source:user_explicit" in retains[0][2]["tags"]


def test_approved_writes_hides_direct_write_tools(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    p = _provider()
    names = _schema_names(p)
    assert "memory_conclude" not in names
    assert "memory_recall" in names
    out = json.loads(p.handle_tool_call("memory_conclude", {"conclusion": "x"}))
    assert "error" in out
    out = json.loads(p.handle_tool_call("memory_profile", {"card": ["x"]}))
    assert "read-only" in out["error"]
    assert not [e for e in p._honcho.events if e[0] == "tool"]
    assert "built-in `memory` tool" in p.system_prompt_block()


def test_turns_mode_still_ingests(hermes_home):
    p = _provider()
    p.sync_turn("user turn", "assistant turn")
    assert "sync_turn" in p._honcho.kinds()
    assert "sync_turn" in p._hindsight.kinds()


def test_subagent_context_never_writes(hermes_home):
    p = _provider(backends=("hindsight",))
    p._writes_allowed = policy.writes_allowed("subagent")
    p.on_memory_write("add", "memory", "fact")
    p.sync_turn("u", "a")
    assert p._hindsight.events == []
    assert "memory_forget" not in _schema_names(p)


def test_unknown_ingest_mode_is_strict(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "typo")
    assert policy.approved_writes_only()


# ---------------------------------------------------------------------------
# prefetch.enabled
# ---------------------------------------------------------------------------

def test_prefetch_disabled_injects_nothing(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_ENABLED", "false")
    p = _provider()
    assert p.prefetch("anything") == ""
    p.queue_prefetch("anything")
    assert p._honcho.events == []
    assert p._hindsight.events == []


# ---------------------------------------------------------------------------
# scope.mode=chat
# ---------------------------------------------------------------------------

def test_chat_scope_partitions_bank_and_fact_store(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SCOPE_MODE", "chat")
    monkeypatch.setenv("MNEMOSYNE_INGEST_MODE", "approved_writes")
    rooms = {}
    for room in ("!a:example.org", "!b:example.org"):
        p = _provider(backends=("hindsight",))
        p.initialize("s-" + room, platform="matrix", chat_id=room, agent_context="primary")
        assert not p._blocked
        rooms[room] = (p._hindsight._bank_id, p._fact_store._db_path)
    (bank_a, db_a), (bank_b, db_b) = rooms.values()
    assert bank_a.startswith("hermes-") and bank_b.startswith("hermes-")
    assert bank_a != bank_b
    assert db_a != db_b
    assert db_a == policy.scope_dir("matrix:!a:example.org") / "fact_store.db"


def test_chat_scope_without_chat_id_fails_closed(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SCOPE_MODE", "chat")
    p = _provider(backends=("hindsight",))
    p.initialize("s1", platform="cron", agent_context="cron")
    assert p._blocked
    assert p.get_tool_schemas() == []
    assert p.system_prompt_block() == ""
    assert p.prefetch("q") == ""
    assert "error" in json.loads(p.handle_tool_call("memory_recall", {"query": "q"}))
    p.on_memory_write("add", "memory", "fact")
    assert p._hindsight.kinds() == []


def test_chat_scope_refuses_honcho(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SCOPE_MODE", "chat")
    p = _provider()
    p.initialize("s1", platform="matrix", chat_id="!a:example.org")
    assert p._blocked
    assert "honcho" in p._disabled_reason


def test_chat_scope_fails_closed_without_bank_id(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_SCOPE_MODE", "chat")
    p = _provider(backends=("hindsight",))
    p._hindsight._bank_id = None
    p.initialize("s1", platform="matrix", chat_id="!a:example.org")
    assert p._blocked


@pytest.mark.parametrize("env", [
    {"MNEMOSYNE_INGEST_MODE": "approved_writes"},
    {"MNEMOSYNE_SCOPE_MODE": "chat"},
])
def test_recovery_skipped(hermes_home, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    p = _provider(backends=("hindsight",))
    spawned = []
    monkeypatch.setattr(p, "_spawn_recovery", lambda: spawned.append(True))
    monkeypatch.setattr(mnemosyne, "initialize_cursor_if_missing", lambda: False)
    p.initialize("s1", platform="matrix", chat_id="!a:example.org")
    assert spawned == []


# ---------------------------------------------------------------------------
# backends and CLI
# ---------------------------------------------------------------------------

def test_backends_from_env(hermes_home, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_BACKENDS", "hindsight, bogus")
    assert policy.backends() == ["hindsight"]


def test_hindsight_only_exposes_hindsight_tools(hermes_home):
    p = _provider(backends=("hindsight",))
    assert set(_schema_names(p)) == {"memory_recall", "memory_reflect", "memory_forget"}
    assert p.is_available()


@pytest.mark.parametrize("env", [
    {"MNEMOSYNE_INGEST_MODE": "approved_writes"},
    {"MNEMOSYNE_SCOPE_MODE": "chat"},
])
def test_cli_import_refuses(hermes_home, monkeypatch, env, capsys):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cli, "_make_hindsight", lambda: pytest.fail("must not reach hindsight"))
    assert cli._cmd_import(None) == 2
    assert "Refusing" in capsys.readouterr().err
