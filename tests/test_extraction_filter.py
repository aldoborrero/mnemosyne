"""Prevent recalled context from being ingested as new assistant information."""

import json
from unittest.mock import Mock

import pytest
from conftest import make_provider

from mnemosyne import config
from mnemosyne.extraction_filter import strip_prefetched


MEMORY = "one two three four five six seven eight nine"
NEW_INFORMATION = "A completely new fact about tomorrow."


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    values = {("prefetch", "dedup_use_embeddings"): False}
    monkeypatch.setattr(
        config, "get", lambda *keys, default=None: values.get(keys, default)
    )
    return values


@pytest.mark.parametrize(
    "assistant,context,expected",
    [
        ("", MEMORY, ""),
        (MEMORY, "", MEMORY),
        (MEMORY, "short context", MEMORY),
        (
            "one two three four five six seven",
            MEMORY,
            "one two three four five six seven",
        ),
        (
            "one two three four five six seven eight",
            MEMORY,
            "one two three four five six seven eight",
        ),
        (MEMORY, MEMORY, ""),
        (MEMORY + " ten eleven", MEMORY, ""),  # Two of four shingles: exactly 50%.
        (MEMORY + " ten eleven twelve", MEMORY, MEMORY + " ten eleven twelve"),
        (
            "tea " * 20,
            "tea " * 20,
            "tea " * 20,
        ),  # One distinct shingle, despite length.
        (
            "Mi CAFÉ, está junto al río detrás del árbol.",
            "mi café está junto al río detrás del árbol",
            "",
        ),
        (
            "nine eight seven six five four three two one",
            MEMORY,
            "nine eight seven six five four three two one",
        ),
    ],
)
def test_filter_boundaries_and_normalization(assistant, context, expected):
    assert strip_prefetched(assistant, context) == expected


def test_preserves_retained_paragraph_order_and_whitespace():
    assistant = "\n\n" + MEMORY + "\n\n  \n\n" + NEW_INFORMATION + "\n\n"
    assert (
        strip_prefetched(assistant, MEMORY) == "\n\n  \n\n" + NEW_INFORMATION + "\n\n"
    )


def test_disabled_filter_returns_response_verbatim(settings):
    settings[("prefetch", "strip_from_extraction")] = False
    assistant = MEMORY + "\n\n" + NEW_INFORMATION
    assert strip_prefetched(assistant, MEMORY) == assistant


@pytest.mark.parametrize("enabled", [True, False])
def test_sync_turn_filters_only_assistant_and_forwards_to_both_backends(
    settings, enabled
):
    settings[("prefetch", "strip_from_extraction")] = enabled
    provider = make_provider(honcho_card=(), hindsight_text=MEMORY)
    try:
        # Use actual prefetch to populate the context consumed by the write filter.
        assert MEMORY in provider.prefetch("remember")
        provider._honcho = Mock()
        provider._hindsight = Mock()
        provider._fact_store = Mock()
        assistant = MEMORY + "\n\n" + NEW_INFORMATION
        user = "  " + MEMORY + "  "

        provider.sync_turn(user, assistant, session_id="test-session")

        expected = NEW_INFORMATION if enabled else assistant
        for backend in (provider._honcho, provider._hindsight):
            backend.sync_turn.assert_called_once_with(
                user, expected, session_id="test-session"
            )
        provider._fact_store.bump.assert_called_once_with(user, source="conversation")
    finally:
        provider._executor.shutdown(wait=True)


def test_latest_prefetch_replaces_the_context_used_for_filtering():
    provider = make_provider(honcho_card=(), hindsight_text=MEMORY)
    try:
        provider.prefetch("first")
        provider._hindsight._fixed = json.dumps({"result": "New unrelated context"})
        provider.prefetch("second")
        provider._honcho = Mock()
        provider._hindsight = Mock()

        provider.sync_turn("user", MEMORY)

        for backend in (provider._honcho, provider._hindsight):
            backend.sync_turn.assert_called_once_with("user", MEMORY, session_id="")
    finally:
        provider._executor.shutdown(wait=True)
