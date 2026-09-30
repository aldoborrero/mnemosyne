"""Shared text normalisation for the similarity heuristics.

Three modules used to carry their own copy of "split into content tokens,
drop stopwords": `dedup` (recall-time clustering), `forget` (forget-signature
build and match) and `conflict` (contradiction detection). The copies drifted
— `dedup._STOP` and `forget._FORGET_STOP` ended up 38 tokens apart — and
because a forget signature is built by one and matched by the other, the
containment threshold was being scored across two different vocabularies.

One tokenizer here, with the stoplists named for the job they do:

* ``GENERIC_STOP``  — EN/RU function words. The shared base.
* ``SPEAKER_STOP``  — GENERIC plus the speaker nouns that show up in every
  recall line ("user", "assistant" and their RU forms). Used by anything
  comparing recall lines to each other or to a forget signature, where those
  words are pure noise.
* ``PRONOUN_STOP``  — GENERIC plus personal pronouns, for contradiction
  detection, which compares a profile line against a fact line and must not
  score "my"/"your" as shared topic.

Keep new entries in the list they belong to rather than adding a fourth set.
"""

from __future__ import annotations

import re
from typing import Set

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

# EN/RU function words. Shared by every caller.
GENERIC_STOP: Set[str] = {
    # English
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "for",
    "with",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "not",
    "no",
    # Russian
    "и",
    "в",
    "на",
    "не",
    "что",
    "это",
    "как",
    "по",
    "из",
    "к",
    "у",
    "о",
    "от",
    "за",
    "со",
    "до",
    "для",
    "над",
    "под",
    "при",
    "без",
    "то",
    "же",
    "ли",
    "бы",
    "ну",
    "вот",
    "там",
    "тут",
    "тоже",
    "также",
    "очень",
    "ещё",
    "уже",
    "был",
    "была",
    "были",
    "было",
    "есть",
    "нет",
    "да",
    "об",
    "а",
    "но",
    "или",
    "если",
    "так",
    "его",
    "её",
    "их",
}

# Speaker nouns: noise in recall lines, which are all "User: … Assistant: …".
SPEAKER_STOP: Set[str] = GENERIC_STOP | {
    "user",
    "user's",
    "assistant",
    "пользователь",
    "пользователя",
    "пользователю",
    "ассистент",
}

# Personal pronouns: a profile line and a fact line both saying "my" is not
# evidence they share a topic.
PRONOUN_STOP: Set[str] = GENERIC_STOP | {
    "i",
    "me",
    "my",
    "you",
    "your",
    "he",
    "she",
    "we",
    "they",
    "them",
    "their",
    "his",
    "her",
    "мне",
    "меня",
    "мой",
    "моя",
    "мои",
    "ты",
    "вы",
    "он",
    "она",
    "они",
}


def content_tokens(text: str, stop: Set[str] = SPEAKER_STOP) -> Set[str]:
    """Lowercase word tokens, minus `stop` and single characters."""
    return {
        t.lower()
        for t in _WORD_RE.findall(text or "")
        if len(t) > 1 and t.lower() not in stop
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Symmetric overlap: |a ∩ b| / |a ∪ b|. 0 when either side is empty."""
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def containment(needle: Set[str], haystack: Set[str]) -> float:
    """Asymmetric overlap: |needle ∩ haystack| / |needle|.

    The right question when the two sides have very different sizes — "is
    this short thing covered by that long thing?" — where Jaccard would
    punish the size gap and score a real match near zero.
    """
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)
