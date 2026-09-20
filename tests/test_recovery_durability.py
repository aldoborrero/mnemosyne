"""Recovery must never step over turns it failed to flush.

The regression: the cursor advanced by `len(pairs)` regardless of whether
Hindsight accepted them, so a mid-run failure silently skipped those turns
forever — in the one component whose whole job is not losing turns.
"""
from __future__ import annotations

import json
import warnings

import pytest

from _hermes_user_memory.mnemosyne import recovery


class _FlakyProvider:
    """Accepts retains until `fail_from`, then refuses every one."""

    def __init__(self, fail_from: int | None = None, delay: float = 0.0):
        self.fail_from = fail_from
        self.delay = delay
        self.accepted: list[str] = []
        self.calls = 0

    def handle_tool_call(self, name, args):
        assert name == "hindsight_retain"
        if self.delay:
            import time

            time.sleep(self.delay)
        self.calls += 1
        if self.fail_from is not None and self.calls > self.fail_from:
            raise RuntimeError("hindsight is down")
        self.accepted.append(args.get("content", ""))
        return "{}"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "sessions").mkdir()
    (tmp_path / "plugins" / "mnemosyne").mkdir(parents=True)
    return tmp_path


def _write_session(home, name, n_pairs):
    path = home / "sessions" / name
    lines = [json.dumps({"role": "session_meta", "id": name})]
    for i in range(n_pairs):
        lines.append(json.dumps({"role": "user", "content": f"q{i}"}))
        lines.append(json.dumps({"role": "assistant", "content": f"a{i}"}))
    path.write_text("\n".join(lines) + "\n")
    return path


def _cursor(home):
    p = home / "plugins" / "mnemosyne" / "recovery_cursor.json"
    return json.loads(p.read_text()) if p.exists() else {}


def test_sessions_dir_honours_hermes_home(hermes_home):
    assert recovery._sessions_dir() == hermes_home / "sessions"


def test_cursor_advances_over_every_flushed_pair(hermes_home):
    _write_session(hermes_home, "20260101_120000_aaaa.jsonl", 3)
    p = _FlakyProvider()

    res = recovery.replay_missed(p, max_pairs=50)

    assert res["replayed"] == 3
    assert len(p.accepted) == 3
    assert _cursor(hermes_home)["last_offset"] == 3


def test_cursor_stops_at_the_failed_pair(hermes_home):
    _write_session(hermes_home, "20260101_120000_aaaa.jsonl", 5)
    p = _FlakyProvider(fail_from=2)  # pairs 1 and 2 land, 3 fails

    res = recovery.replay_missed(p, max_pairs=50)

    assert res["replayed"] == 2
    assert res.get("stopped_at_failure") is True
    cur = _cursor(hermes_home)
    assert cur["last_offset"] == 2, "cursor must not step over the failed pair"
    assert cur["last_filename"] == "20260101_120000_aaaa.jsonl"


def test_next_run_resumes_exactly_at_the_failed_pair(hermes_home):
    """End to end: a failed run followed by a healthy one loses nothing."""
    _write_session(hermes_home, "20260101_120000_aaaa.jsonl", 5)

    first = _FlakyProvider(fail_from=2)
    recovery.replay_missed(first, max_pairs=50)
    assert [c.split("\n")[0] for c in first.accepted] == ["User: q0", "User: q1"]

    second = _FlakyProvider()
    res = recovery.replay_missed(second, max_pairs=50)

    assert res["replayed"] == 3
    # Resumes at q2 — no gap, no duplicate.
    assert [c.split("\n")[0] for c in second.accepted] == [
        "User: q2",
        "User: q3",
        "User: q4",
    ]
    assert _cursor(hermes_home)["last_offset"] == 5


def test_wall_clock_budget_stops_replay(hermes_home):
    _write_session(hermes_home, "20260101_120000_aaaa.jsonl", 20)
    p = _FlakyProvider(delay=0.02)

    res = recovery.replay_missed(p, max_pairs=50, max_seconds=0.05)

    assert res.get("stopped_at_deadline") is True
    assert 0 < res["replayed"] < 20
    # Everything it did flush is recorded, nothing more.
    assert _cursor(hermes_home)["last_offset"] == res["replayed"]


def test_max_pairs_limit_is_recorded(hermes_home):
    _write_session(hermes_home, "20260101_120000_aaaa.jsonl", 10)
    p = _FlakyProvider()

    res = recovery.replay_missed(p, max_pairs=4)

    assert res["replayed"] == 4
    assert res.get("stopped_at_limit") is True
    assert _cursor(hermes_home)["last_offset"] == 4


def test_cursor_timestamp_is_not_a_deprecated_utcnow(hermes_home):
    _write_session(hermes_home, "20260101_120000_aaaa.jsonl", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        recovery.replay_missed(_FlakyProvider(), max_pairs=1)
    # Timezone-aware, so it round-trips with an offset.
    from datetime import datetime

    ts = datetime.fromisoformat(_cursor(hermes_home)["updated_at"])
    assert ts.tzinfo is not None
