"""Plan item 6 — conflict detector for two-voice display.

Rule-based contradiction signals:
  * shared topic (significant lexical overlap), AND
  * differing named entities OR differing numbers/dates.

This is intentionally cheap and approximate — no LLM call. Used at
prefetch fusion time and at pre-write dedup time. False positives are
acceptable; the plugin marks both items with source/timestamp labels and
shows them side-by-side for the agent to reason about.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from .text_utils import PRONOUN_STOP, content_tokens, jaccard

# Treat any token starting with an uppercase letter as a named entity
# candidate. Crude but works for many cases (names, places, products).
_ENTITY_RE = re.compile(r"\b[A-ZА-ЯЁ][\w\-]{1,}\b", flags=re.UNICODE)
_NUMBER_RE = re.compile(r"\b\d[\d,.\:\-/]*\b")


def _tokens(text: str) -> Set[str]:
    # Pronoun-aware stoplist: this compares a profile line against a fact
    # line, and both saying "my" is not evidence of a shared topic.
    return content_tokens(text, PRONOUN_STOP)


def _entities(text: str) -> Set[str]:
    return {m.lower() for m in _ENTITY_RE.findall(text)}


def _numbers(text: str) -> Set[str]:
    return set(_NUMBER_RE.findall(text))


def topic_overlap(a: str, b: str) -> float:
    """Jaccard over content tokens. 0 means disjoint, 1 means identical bag."""
    return jaccard(_tokens(a), _tokens(b))


def is_contradiction(a: str, b: str, *, topic_threshold: float = 0.3) -> bool:
    """True when a and b seem to talk about the same topic but disagree on
    a key entity or number."""
    if topic_overlap(a, b) < topic_threshold:
        return False
    ea, eb = _entities(a), _entities(b)
    if ea and eb and not (ea & eb):
        return True
    na, nb = _numbers(a), _numbers(b)
    if na and nb and not (na & nb):
        return True
    return False


def label_pair(a: str, a_meta: dict, b: str, b_meta: dict) -> Tuple[str, str]:
    """Return (a, b) reformatted with source/recency labels for two-voice
    display. `meta` may carry keys: source, when (ISO date), label."""
    return (
        f"[{_label(a_meta)}] {a}",
        f"[{_label(b_meta)}] {b}",
    )


def _label(meta: dict) -> str:
    parts: List[str] = []
    if meta.get("label"):
        parts.append(meta["label"])
    if meta.get("source"):
        parts.append(meta["source"])
    if meta.get("when"):
        parts.append(meta["when"])
    return ", ".join(parts) if parts else "memory"
