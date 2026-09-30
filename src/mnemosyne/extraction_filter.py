"""Suppress repeated recalled context before ingesting an assistant response.

The provider supplies its latest prefetch payload. This module holds no session
state and never processes the user's message.
"""

from __future__ import annotations

import logging
from typing import List

from . import config

__all__ = ["strip_prefetched"]

logger = logging.getLogger(__name__)


_SHINGLE_SIZE = 8  # words per shingle
_SHINGLE_HIT_RATIO = 0.5  # fraction of a paragraph's shingles that must
# be in the prefetch to qualify for stripping


def strip_prefetched(assistant_content: str, prefetched_context: str) -> str:
    """Drop paragraphs from `assistant_content` that mostly repeat
    text we just put into the prompt via prefetch.

    Why: Hindsight.sync_turn LLM-extracts facts from the JSON of the
    turn. If the assistant cited or paraphrased the prefetched
    memories, the extractor mints fresh records of them — every
    session. Recall picks them up next time, and the bank grows
    without bound (the Barsik loop).

    Approach: word-shingles, paragraph granularity. If a paragraph's
    shingle hit rate against the prefetch is high, drop it. Conservative
    thresholds: paragraphs with fewer than two distinct shingles are
    untouched, and paragraphs below the overlap threshold survive."""
    if not assistant_content or not prefetched_context:
        return assistant_content
    if not bool(config.get("prefetch", "strip_from_extraction", default=True)):
        return assistant_content

    prefetch_shingles = _build_shingles(prefetched_context)
    if not prefetch_shingles:
        return assistant_content

    kept_paragraphs: List[str] = []
    for para in assistant_content.split("\n\n"):
        if not para.strip():
            kept_paragraphs.append(para)
            continue
        para_shingles = _build_shingles(para)
        if len(para_shingles) < 2:
            kept_paragraphs.append(para)
            continue
        hits = sum(1 for sh in para_shingles if sh in prefetch_shingles)
        ratio = hits / len(para_shingles)
        if ratio >= _SHINGLE_HIT_RATIO:
            logger.debug(
                "mnemosyne: stripped paraphrased paragraph "
                "(%.0f%% shingle overlap with prefetch)",
                ratio * 100,
            )
            continue
        kept_paragraphs.append(para)
    return "\n\n".join(kept_paragraphs)


def _build_shingles(text: str) -> set:
    if not text:
        return set()
    # Lowercase + strip non-word characters; same logic as
    # fact_store._canonical_key but token-level so word order matters.
    import re as _re

    words = _re.findall(r"\w+", text.lower(), flags=_re.UNICODE)
    if len(words) < _SHINGLE_SIZE:
        return set()
    return {
        tuple(words[i : i + _SHINGLE_SIZE])
        for i in range(len(words) - _SHINGLE_SIZE + 1)
    }
