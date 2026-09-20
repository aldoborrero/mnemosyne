"""The three similarity heuristics share one tokenizer.

They each used to carry their own copy. The copies drifted, and because a
forget signature is built by one module and matched by another, the
containment threshold ended up being scored across two vocabularies.
"""
from __future__ import annotations

from _hermes_user_memory.mnemosyne import conflict, dedup, forget, text_utils


def test_dedup_and_forget_tokenise_identically():
    """Non-negotiable: forget signatures are matched by the recall filter."""
    samples = [
        "Assistant deleted the user's Barsik information",
        "Пользователя зовут Aldo, ассистент это запомнил",
        "User asked to forget the cat",
        "",
    ]
    for s in samples:
        assert dedup._content_tokens(s) == forget._content_tokens(s), s


def test_speaker_nouns_are_dropped_on_both_sides():
    toks = dedup._content_tokens("User: Assistant deleted Barsik")
    assert "barsik" in toks
    assert "user" not in toks and "assistant" not in toks


def test_conflict_drops_pronouns_but_dedup_keeps_them():
    """Different jobs, different stoplists — deliberately, and only here."""
    text = "my cat Barsik"
    assert "my" not in conflict._tokens(text)
    assert "my" in dedup._content_tokens(text)


def test_single_characters_and_empties_are_dropped():
    assert text_utils.content_tokens("a I x yz") == {"yz"}
    assert text_utils.content_tokens("") == set()
    assert text_utils.content_tokens(None) == set()


def test_jaccard_is_symmetric_and_bounded():
    a, b = {"x", "y"}, {"y", "z"}
    assert text_utils.jaccard(a, b) == text_utils.jaccard(b, a)
    assert text_utils.jaccard(a, a) == 1.0
    assert text_utils.jaccard(a, set()) == 0.0


def test_containment_ignores_haystack_size():
    """The property forget relies on: a 1-token query matches a long line."""
    needle = {"barsik"}
    short = {"barsik", "cat"}
    long = {"barsik"} | {f"w{i}" for i in range(30)}
    assert text_utils.containment(needle, short) == 1.0
    assert text_utils.containment(needle, long) == 1.0
    # Jaccard would collapse on the long one — that is why it is not used here.
    assert text_utils.jaccard(needle, long) < 0.05
    assert text_utils.containment(set(), short) == 0.0


def test_dedup_fallback_defaults_match_config_defaults():
    """The inline `default=` fallbacks disagreed with config.py's _DEFAULTS."""
    import inspect

    from _hermes_user_memory.mnemosyne import config

    src = inspect.getsource(dedup.cluster_lines)
    defaults = config._DEFAULTS["prefetch"]
    assert f'"dedup_jaccard_min", default={defaults["dedup_jaccard_min"]}' in src
    assert f'"dedup_cosine_min", default={defaults["dedup_cosine_min"]}' in src


def test_conflict_detection_still_works_after_the_swap():
    assert conflict.is_contradiction(
        "The server runs on port 8080", "The server runs on port 9090"
    )
    assert not conflict.is_contradiction(
        "The server runs on port 8080", "I like strawberry ice cream"
    )
