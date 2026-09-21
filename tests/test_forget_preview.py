"""Forget is a two-step, pinned operation.

The regression these cover: step 2 used to re-run `hindsight_recall` and
forget whatever came back. Hindsight ranks by reranker score and that score
drifts, so the user could approve list A and the system would delete list B.
`_DriftingProvider` below reproduces exactly that — it returns a different
result set on every call — and the tests assert the confirm step is immune.
"""
from __future__ import annotations

import json

import pytest

from _hermes_user_memory.mnemosyne import forget as forget_mod
from _hermes_user_memory.mnemosyne.fact_store import FactStore


class _DriftingProvider:
    """Returns a different recall result on every call, like a live reranker."""

    def __init__(self, *batches):
        self._batches = list(batches)
        self.calls = 0
        self.retained = []

    def handle_tool_call(self, name, args):
        if name == "hindsight_recall":
            batch = self._batches[min(self.calls, len(self._batches) - 1)]
            self.calls += 1
            numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(batch, 1))
            return json.dumps({"result": numbered})
        if name == "hindsight_retain":
            self.retained.append(args.get("content", ""))
            return "{}"
        raise AssertionError(f"unexpected tool {name}")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Keep tombstones synchronous and local so nothing escapes to a thread.
    monkeypatch.setenv("MNEMOSYNE_FORGET_TOMBSTONES_ASYNC", "0")
    return FactStore(db_path=tmp_path / "facts.db")


def _preview(store, provider, query="barsik"):
    return forget_mod.forget_by_query(store, provider, query)


def test_preview_forgets_nothing_and_returns_a_token(store):
    p = _DriftingProvider(["Barsik is a cat", "Barsik likes fish"])
    res = _preview(store, p)

    assert res["preview"] is True
    assert res["forgotten"] == []
    assert res["preview_token"]
    assert [c["text"] for c in res["candidates"]] == [
        "Barsik is a cat",
        "Barsik likes fish",
    ]
    assert not store.is_forgotten("Barsik is a cat")


def test_confirm_applies_the_pinned_list_not_a_fresh_recall(store):
    """The core regression: recall drifts between the two calls."""
    approved = ["Barsik is a cat", "Barsik likes fish"]
    drifted = ["Barsik died in 2019", "Something else entirely"]
    p = _DriftingProvider(approved, drifted)

    res = _preview(store, p)
    assert p.calls == 1

    done = forget_mod.forget_by_query(
        store, p, "barsik", confirmed=True, preview_token=res["preview_token"]
    )

    # No second recall at all, and only the approved texts were marked.
    assert p.calls == 1, "confirm must not search again"
    assert [e["text"] for e in done["forgotten"]] == approved
    for text in approved:
        assert store.is_forgotten(text)
    for text in drifted:
        assert not store.is_forgotten(text), f"forgot un-approved candidate: {text}"


def test_confirm_without_token_is_refused(store):
    p = _DriftingProvider(["Barsik is a cat"])
    _preview(store, p)

    res = forget_mod.forget_by_query(store, p, "barsik", confirmed=True)

    assert "preview_token required" in res["error"]
    assert res["forgotten"] == []
    assert not store.is_forgotten("Barsik is a cat")


def test_unknown_or_expired_token_is_refused(store, monkeypatch):
    p = _DriftingProvider(["Barsik is a cat"])
    res = _preview(store, p)

    bad = forget_mod.forget_by_query(
        store, p, "barsik", confirmed=True, preview_token="deadbeefdeadbeef"
    )
    assert "unknown or expired" in bad["error"]
    assert bad["forgotten"] == []

    # Same token, but the TTL has lapsed.
    monkeypatch.setattr(forget_mod, "_preview_ttl_s", lambda: 0.0)
    expired = forget_mod.forget_by_query(
        store, p, "barsik", confirmed=True, preview_token=res["preview_token"]
    )
    assert "unknown or expired" in expired["error"]
    assert not store.is_forgotten("Barsik is a cat")


def test_indices_select_a_subset_of_the_pinned_list(store):
    texts = ["fact one", "fact two", "fact three"]
    p = _DriftingProvider(texts)
    res = _preview(store, p, query="fact")

    done = forget_mod.forget_by_query(
        store,
        p,
        "fact",
        confirmed=True,
        preview_token=res["preview_token"],
        indices=[1, 3],
    )

    assert [e["text"] for e in done["forgotten"]] == ["fact one", "fact three"]
    assert store.is_forgotten("fact one")
    assert not store.is_forgotten("fact two")
    assert store.is_forgotten("fact three")


def test_out_of_range_and_junk_indices_are_ignored(store):
    p = _DriftingProvider(["only one"])
    res = _preview(store, p, query="only")

    done = forget_mod.forget_by_query(
        store,
        p,
        "only",
        confirmed=True,
        preview_token=res["preview_token"],
        indices=[0, 9, "x", 1],
    )
    assert [e["text"] for e in done["forgotten"]] == ["only one"]


def test_confirm_registers_a_semantic_signature(store):
    """Signatures are what hide paraphrases; both call paths must create one."""
    p = _DriftingProvider(["Barsik the cat died in 2019"])
    res = _preview(store, p)

    done = forget_mod.forget_by_query(
        store, p, "barsik", confirmed=True, preview_token=res["preview_token"]
    )

    assert done["signature_id"]
    sigs = store.list_signatures()
    assert len(sigs) == 1
    assert "barsik" in sigs[0]["tokens"]


def test_signature_tokens_match_the_read_side_filter_tokenizer(store):
    """The signature is built and matched with the same tokenizer, so the
    containment filter never scores across two vocabularies."""
    import _hermes_user_memory.mnemosyne as pkg
    import inspect

    src = inspect.getsource(pkg.MnemosyneMemoryProvider._filter_forgotten)
    assert "from .forget import _content_tokens" in src
    assert "from .dedup import _content_tokens" not in src
