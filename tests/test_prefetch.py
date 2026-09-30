"""Prefetch cache and concurrency contracts through the provider interface."""

import os
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import make_provider

from mnemosyne import config
from mnemosyne import prefetch as cache_module


@pytest.fixture
def context(tmp_path, monkeypatch):
    settings = {
        ("prefetch", "peer_card_ttl_s"): 60,
        ("prefetch", "dedup_use_embeddings"): False,
    }
    monkeypatch.setattr(config, "plugin_dir", lambda: tmp_path)
    monkeypatch.setattr(
        config, "get", lambda *keys, default=None: settings.get(keys, default)
    )
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        cache_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    provider = make_provider()
    provider._honcho = Mock()
    provider._honcho.handle_tool_call.return_value = '{"result": ["tea"]}'
    provider._hindsight = Mock()
    provider._hindsight.handle_tool_call.return_value = '{"result": "hiking"}'
    yield SimpleNamespace(
        provider=provider,
        settings=settings,
        clock=clock,
        anchor=tmp_path / "anchor_card.md",
    )
    provider._executor.shutdown(wait=True)


def test_anchor_cached_until_mtime_changes_and_removed_file_disappears(
    context, monkeypatch
):
    context.anchor.write_text("# Title\n\n tea \nbooks\n")
    original_read = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    p = context.provider
    assert "# Pinned (anchor card)\n- tea\n- books" in p.prefetch("q")
    p.prefetch("q")
    assert reads.count(context.anchor) == 1

    old_mtime = context.anchor.stat().st_mtime_ns
    context.anchor.write_text("music\n")
    os.utime(context.anchor, ns=(old_mtime + 2_000_000_000,) * 2)
    assert "# Pinned (anchor card)\n- music" in p.prefetch("q")
    assert reads.count(context.anchor) == 2
    context.anchor.unlink()
    assert "# Pinned" not in p.prefetch("q")


def test_write_invalidates_only_profile_and_session_switch_invalidates_both(context):
    context.anchor.write_text("original")
    stamp = context.anchor.stat().st_mtime_ns
    p = context.provider
    p.prefetch("q")
    context.anchor.write_text("changed")
    os.utime(context.anchor, ns=(stamp, stamp))
    p._honcho.handle_tool_call.return_value = '{"result": ["coffee"]}'

    p.on_memory_write("add", "user", "coffee")
    out = p.prefetch("q")
    assert "# Pinned (anchor card)\n- original" in out
    assert "# User profile\n- coffee" in out

    p._honcho.handle_tool_call.return_value = '{"result": ["water"]}'
    p.on_session_switch("next")
    out = p.prefetch("q")
    assert "# Pinned (anchor card)\n- changed" in out
    assert "# User profile\n- water" in out


def test_nonwriting_session_does_not_invalidate_profile_on_memory_write(context):
    p = context.provider
    p.prefetch("q")
    p._writes_allowed = False
    p.on_memory_write("add", "user", "coffee")
    p.prefetch("q")
    p._honcho.handle_tool_call.assert_called_once_with("honcho_profile", {})


def test_profile_expires_at_ttl_boundary_and_hindsight_is_not_cached(context):
    p = context.provider
    p.prefetch("q1")
    context.clock.now = 159.999
    p.prefetch("q2")
    assert p._honcho.handle_tool_call.call_count == 1
    context.clock.now = 160
    p._honcho.handle_tool_call.return_value = '{"result": ["coffee"]}'
    assert "# User profile\n- coffee" in p.prefetch("q3")
    assert p._honcho.handle_tool_call.call_count == 2
    assert p._hindsight.handle_tool_call.call_count == 3


def test_profile_ttl_comes_from_configuration(context):
    context.settings[("prefetch", "peer_card_ttl_s")] = 2
    p = make_provider()
    try:
        p.prefetch("q1")
        context.clock.now = 101
        p.prefetch("q2")
        assert p._honcho.calls == 1
        context.clock.now = 102
        p.prefetch("q3")
        assert p._honcho.calls == 2
    finally:
        p._executor.shutdown(wait=True)


def test_prefetch_uses_current_backends_and_skips_empty_queries(context):
    p = context.provider
    p.prefetch("q")
    p._honcho = None
    assert p.prefetch("") == ""
    assert p._hindsight.handle_tool_call.call_count == 1
    p._hindsight = None
    assert p.prefetch("q") == ""


@pytest.mark.parametrize(
    "response,section",
    [
        ({"result": ["tea", "books"]}, "# User profile\n- tea\n- books"),
        ({"hint": "Set up your profile"}, "# User profile\n_Set up your profile_"),
        ({"card": ["write-only"]}, None),
        ("invalid JSON", None),
    ],
)
def test_profile_shapes_are_cached_including_empty_results(context, response, section):
    p = context.provider
    p._honcho.handle_tool_call.return_value = response
    for _ in range(2):
        out = p.prefetch("q")
        if section:
            assert section in out
        else:
            assert "# User profile" not in out
        assert "# Facts (relevant)\n- hiking" in out
    assert p._honcho.handle_tool_call.call_count == 1


def test_all_three_branches_run_concurrently(context, monkeypatch):
    context.anchor.write_text("pinned")
    barrier = Barrier(3, timeout=3)
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        barrier.wait()
        return original_read(path, *args, **kwargs)

    def profile(*args, **kwargs):
        barrier.wait()
        return '{"result": ["tea"]}'

    def recall(*args, **kwargs):
        barrier.wait()
        return '{"result": "hiking"}'

    monkeypatch.setattr(Path, "read_text", read)
    p = context.provider
    p._honcho.handle_tool_call.side_effect = profile
    p._hindsight.handle_tool_call.side_effect = recall
    assert p.prefetch("q") == (
        "# Pinned (anchor card)\n- pinned\n\n"
        "# User profile\n- tea\n\n# Facts (relevant)\n- hiking"
    )


def test_timed_out_branch_does_not_hide_ready_results(context):
    started, release = Event(), Event()

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(timeout=3)
        return '{"result": ["late"]}'

    context.settings[("prefetch", "parallel_timeout_s")] = 0.1
    p = context.provider
    p._honcho.handle_tool_call.side_effect = blocked
    try:
        out = p.prefetch("q")
        assert started.is_set()
        assert out == "# Facts (relevant)\n- hiking"
        assert p._last_prefetch == out
    finally:
        release.set()


def test_failed_branches_leave_anchor_available(context):
    context.anchor.write_text("pinned")
    p = context.provider
    p._honcho.handle_tool_call.side_effect = OSError("offline")
    p._hindsight.handle_tool_call.side_effect = OSError("offline")
    assert p.prefetch("q") == "# Pinned (anchor card)\n- pinned"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("x" * 1700, "x" * 1500),
        ("a" * 1100 + " " + "b" * 599, "a" * 1100),
        ("a" * 500 + " " + "b" * 1200, "a" * 500 + " " + "b" * 999),
    ],
)
def test_recall_query_and_token_limits_are_preserved(context, query, expected):
    context.settings[("prefetch", "hindsight_token_budget")] = 5000
    p = context.provider
    p.prefetch(query)
    p._hindsight.handle_tool_call.assert_called_once_with(
        "hindsight_recall", {"query": expected, "max_tokens": 4096}
    )


def test_queue_prefetch_forwards_session_even_if_one_backend_fails(context):
    p = context.provider
    p._honcho.queue_prefetch.side_effect = OSError("offline")
    p.queue_prefetch("q", session_id="session-1")
    p._honcho.queue_prefetch.assert_called_once_with("q", session_id="session-1")
    p._hindsight.queue_prefetch.assert_called_once_with("q", session_id="session-1")
