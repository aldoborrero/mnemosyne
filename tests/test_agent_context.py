"""Sessions Hermes marks as non-writing must not write memory.

Hermes passes agent_context="cron" to scheduled jobs and "subagent" to
delegate_task children and documents that providers skip writes for them
(agent/agent_init.py); Honcho also skips "flush".
"""
from __future__ import annotations

import json

import pytest
from conftest import make_provider


class _Recorder:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.events.append(name)
            return "" if name == "on_pre_compress" else json.dumps({"result": "ok"})
        return record


def _provider(agent_context):
    p = make_provider()
    p._honcho, p._hindsight = _Recorder(), _Recorder()
    p._writes_allowed = agent_context not in ("cron", "subagent", "flush")
    return p


@pytest.mark.parametrize("ctx", ["cron", "subagent", "flush"])
def test_non_writing_contexts_do_not_write(ctx):
    p = _provider(ctx)
    p.sync_turn("user", "assistant")
    p.on_memory_write("add", "memory", "fact")
    p.on_session_end([{"role": "user", "content": "x"}])
    assert p.on_pre_compress([{"role": "user", "content": "x"}]) == ""
    p.on_delegation("task", "result")
    assert p._honcho.events == [] and p._hindsight.events == []
    names = [s["name"] for s in p.get_tool_schemas()]
    assert "memory_conclude" not in names and "memory_forget" not in names
    assert "memory_recall" in names
    for tool, args in (("memory_conclude", {"conclusion": "x"}), ("memory_forget", {"query": "x"}),
                       ("memory_profile", {"card": ["x"]})):
        assert "error" in json.loads(p.handle_tool_call(tool, args))
    assert p._honcho.events == []


def test_primary_context_still_writes():
    p = _provider("primary")
    p.on_memory_write("add", "memory", "fact")
    assert "on_memory_write" in p._hindsight.events
    assert "memory_conclude" in [s["name"] for s in p.get_tool_schemas()]


def test_initialize_reads_agent_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = make_provider()
    p._honcho, p._hindsight = _Recorder(), _Recorder()
    monkeypatch.setattr(p, "_inject_hindsight_routing_env", lambda: None)
    spawned = []
    monkeypatch.setattr(p, "_spawn_recovery", lambda: spawned.append(True))
    import conftest
    monkeypatch.setattr(conftest.mnemosyne, "initialize_cursor_if_missing", lambda: False)
    p.initialize("s1", agent_context="cron", platform="cron")
    assert p._writes_allowed is False and spawned == []
    p.initialize("s2", agent_context="primary", platform="cli")
    assert p._writes_allowed is True and spawned == [True]
