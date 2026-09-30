"""Recall output contracts, shared by prefetch and the memory_recall tool."""

import json
from unittest.mock import Mock

import pytest
from conftest import make_provider

from mnemosyne import config, dedup, recall_processing
from mnemosyne.fact_store import FactStore
from mnemosyne.forget import _content_tokens


@pytest.fixture
def settings(monkeypatch):
    values = {("prefetch", "dedup_use_embeddings"): False}
    monkeypatch.setattr(
        config, "get", lambda *keys, default=None: values.get(keys, default)
    )
    return values


@pytest.fixture
def provider(settings, tmp_path):
    instance = make_provider()
    instance._fact_store = FactStore(db_path=tmp_path / "facts.db")
    yield instance
    instance._executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "data,expected",
    [
        ("1. tea\n2. tea | Involving: user\n3. hiking", "- tea\n- hiking"),
        (
            ["tea", {"text": "hiking"}, {"content": "books"}, {"body": "music"}, 7],
            "- tea\n- hiking\n- books\n- music",
        ),
        ({"result": "tea", "memories": ["ignored"]}, "- tea"),
        ({"result": "", "memories": [{"text": "tea", "content": "ignored"}]}, "- tea"),
        ({"data": {"matches": [{"body": "tea"}]}}, "- tea"),
        ([], ""),
        ({}, "{}"),
        (None, "None"),
        (42, "42"),
    ],
)
def test_recall_response_shapes(settings, data, expected):
    assert recall_processing.format_hindsight_results(data) == expected


def test_clustering_keeps_longest_representative_in_original_cluster_order(settings):
    text = "1. tea green\n2. mountain hiking\n3. tea green jasmine"
    assert recall_processing.format_hindsight_results(text) == (
        "- tea green jasmine\n- mountain hiking"
    )


def test_embedding_disagreement_keeps_similar_facts(settings, monkeypatch):
    settings[("prefetch", "dedup_use_embeddings")] = True
    monkeypatch.setattr(dedup, "_fetch_embeddings", lambda *a, **kw: [[1, 0], [0, 1]])
    assert recall_processing.format_hindsight_results(
        "1. Barsik sick\n2. Barsik died"
    ) == ("- Barsik sick\n- Barsik died")


def test_disabled_fuzzy_dedup_still_removes_exact_duplicates(settings):
    settings[("prefetch", "dedup_enabled")] = False
    assert recall_processing.format_hindsight_results(
        "1. tea\n2. TEA\n3. tea green"
    ) == ("- tea\n- tea green")


def test_forget_filter_preserves_unmatched_text_and_refreshes_signatures(provider):
    store = provider._fact_store
    store.mark_forgotten("tea")
    signature = store.add_signature(
        _content_tokens("Barsik cat fish garden"), examples=["Barsik cat fish garden"]
    )
    untouched = store.add_signature(_content_tokens("oranges"), examples=["oranges"])
    text = "- tea\n\n- [FORGOTTEN 2026-01-01] old fact\n- Barsik cat | Involving: user\n  - hiking  "

    assert (
        recall_processing.filter_forgotten(text, fact_store=provider._fact_store)
        == "\n  - hiking  "
    )
    signatures = {sig["id"]: sig for sig in store.list_signatures()}
    assert signatures[signature]["last_match_ts"]
    assert not signatures[untouched]["last_match_ts"]


def test_missing_store_leaves_text_unchanged(provider):
    provider._fact_store = None
    text = "\n- [FORGOTTEN 2026-01-01] old fact\n"
    assert (
        recall_processing.filter_forgotten(text, fact_store=provider._fact_store)
        == text
    )


def test_signature_read_failure_keeps_exact_forget_filter(provider, monkeypatch):
    provider._fact_store.mark_forgotten("tea")
    monkeypatch.setattr(
        provider._fact_store, "list_signatures", Mock(side_effect=OSError("offline"))
    )
    assert (
        recall_processing.filter_forgotten(
            "- tea\n- hiking", fact_store=provider._fact_store
        )
        == "- hiking"
    )


def test_invalid_threshold_and_touch_failure_do_not_lose_filtering(
    provider, settings, monkeypatch
):
    settings[("forget", "signature_jaccard_min")] = "invalid"
    provider._fact_store.add_signature(_content_tokens("Barsik cat"), examples=[])
    monkeypatch.setattr(
        provider._fact_store, "touch_signature", Mock(side_effect=OSError("offline"))
    )
    assert (
        recall_processing.filter_forgotten(
            "- Barsik cat\n- hiking", fact_store=provider._fact_store
        )
        == "- hiking"
    )


def test_conflict_annotations_preserve_section_and_fact_order(monkeypatch):
    monkeypatch.setattr(recall_processing, "today_iso", lambda: "2026-01-02")
    sections = [
        "# Pinned (anchor card)\n- tea",
        "# User profile\n- release scheduled for 2025",
        "# Facts (relevant)\n- release scheduled for 2026\n\n- hiking",
    ]
    assert recall_processing.apply_conflict_resolver(sections) == [
        sections[0],
        sections[1],
        "# Facts (relevant)\n[Hindsight, 2026-01-02] - release scheduled for 2026\n"
        "[Honcho profile] - release scheduled for 2025\n\n- hiking",
    ]
    assert recall_processing.apply_conflict_resolver(sections[:2]) == sections[:2]


@pytest.mark.parametrize(
    "text,limit,expected",
    [
        ("abc", 3, "abc"),
        ("abcdefgh", 5, "abcde\n…[truncated]"),
        ("abcdef\nghijkl", 10, "abcdef\n…[truncated]"),
        ("ab\ncdefghijk", 8, "ab\ncdefg\n…[truncated]"),
    ],
)
def test_output_truncation_preserves_newline_rule(text, limit, expected):
    assert recall_processing.truncate_to_chars(text, limit) == expected


def test_recall_tool_formats_deduplicates_and_filters(provider):
    provider._fact_store.mark_forgotten("tea")
    provider._hindsight._fixed = json.dumps({"result": "1. tea\n2. hiking\n3. hiking"})
    assert json.loads(
        provider.handle_tool_call("memory_recall", {"query": "hobbies"})
    ) == {"result": "- hiking"}
    provider._fact_store.mark_forgotten("hiking")
    assert json.loads(
        provider.handle_tool_call("memory_recall", {"query": "hobbies"})
    ) == {"result": "No relevant memories found."}


def test_prefetch_formats_filters_and_preserves_sections(provider, monkeypatch):
    monkeypatch.setattr(provider, "_read_anchor_card", lambda: "- pinned")
    provider._fact_store.mark_forgotten("tea")
    provider._hindsight._fixed = json.dumps({"result": "1. tea\n2. hiking\n3. hiking"})
    expected = (
        "# Pinned (anchor card)\n- pinned\n\n"
        "# User profile\n- alpha\n- beta\n\n# Facts (relevant)\n- hiking"
    )
    assert provider.prefetch("hobbies") == expected
    assert provider._last_prefetch == expected


def test_prefetch_preserves_branch_and_total_budgets(provider, settings, monkeypatch):
    monkeypatch.setattr(provider, "_read_anchor_card", lambda: "abcdefghijk")
    provider._honcho = None
    provider._hindsight = None
    settings[("prefetch", "anchor_token_budget")] = 2
    assert provider.prefetch("q") == "# Pinned (anchor card)\nabcdefgh\n…[truncated]"
    settings[("prefetch", "max_total_tokens")] = 4
    assert provider.prefetch("q") == "# Pinned (anchor\n…[truncated]"


def test_non_json_backend_response_keeps_existing_fallback(provider):
    raw = "1. tea\n2. tea"
    provider._hindsight._fixed = raw
    assert provider._fetch_hindsight_recall("tea", 100) == raw
    assert provider.handle_tool_call("memory_recall", {"query": "tea"}) == raw
