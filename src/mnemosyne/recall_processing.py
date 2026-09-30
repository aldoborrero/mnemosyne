"""Format and filter recalled memories before exposing them to Hermes.

The provider and prefetcher own backend calls and caches. This module processes
their results and bounds recall queries, using an explicit store for forget filtering.
"""

from __future__ import annotations

from typing import Any, List, Optional

from . import config
from .conflict import is_contradiction, label_pair
from .fact_store import FactStore, today_iso
from .forget import is_forgotten as _is_forgotten
from .tool_schemas import RECALL_QUERY_MAX_CHARS

__all__ = [
    "apply_conflict_resolver",
    "filter_forgotten",
    "format_hindsight_results",
    "truncate_to_chars",
    "truncate_recall_query",
]


def truncate_to_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.5:
        cut = cut[:last_nl]
    return cut + "\n…[truncated]"


def format_hindsight_results(data: Any) -> str:
    if isinstance(data, str):
        return _dedupe_recall_text(data)
    if isinstance(data, list):
        joined = "\n".join(_extract_text(item) for item in data if _extract_text(item))
        return _dedupe_recall_text(joined)
    if isinstance(data, dict):
        # 'result' is the key the hermes hindsight plugin uses for
        # numbered-list recall output. Check it first.
        for key in (
            "result",
            "memories",
            "results",
            "items",
            "matches",
            "data",
            "text",
        ):
            v = data.get(key)
            if v:
                return format_hindsight_results(v)
    return str(data)


def _dedupe_recall_text(text: str) -> str:
    """Hindsight returns top-N candidates ranked by similarity to the
    QUERY, not pairwise-distinct. With 40+ paraphrases of one event in
    the bank, all of them survive the existing exact-canonical-key
    filter and bloat context.

    Hybrid clustering (see ``dedup.cluster_lines``):
      - Jaccard token overlap groups paraphrases.
      - Embedding cosine confirms the merge — pairs that pass
        Jaccard but fail cosine are kept apart, protecting against
        "Barsik got sick" vs "Barsik died" type collapses.
    Configurable via ``prefetch.dedup_*`` keys."""
    if not text:
        return text
    from .dedup import cluster_lines

    cleaned: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        content = stripped
        for prefix in range(1, 100):
            pfx = f"{prefix}. "
            if content.startswith(pfx):
                content = content[len(pfx) :]
                break
        head, sep, _ = content.partition(" | Involving:")
        content = head if sep else content
        if content:
            cleaned.append(content)

    kept = cluster_lines(cleaned)
    return "\n".join(f"- {item}" for item in kept)


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text") or item.get("content") or item.get("body") or ""
    return ""


def filter_forgotten(text: str, *, fact_store: Optional[FactStore]) -> str:
    """Drop lines whose canonical key was point-forgotten OR whose
    token set hits any stored semantic forget signature.

    Matching uses **candidate-containment**: ``|cand ∩ sig| / |cand|``.
    Asks "is most of this candidate's vocabulary covered by the
    forget signature?" — the right question for asymmetric sizes (a
    signature accumulates many tokens across an op; a candidate is
    one line). Symmetric Jaccard fails here: a 6-token candidate vs
    an 11-token signature with 4 shared words scores 0.31 — below
    any safe threshold — even though every content word in the
    candidate IS in the signature. Containment scores 0.66 and
    catches the paraphrase as intended.
    """
    if not fact_store:
        return text

    # Refresh the in-memory signature cache on demand. The table is
    # tiny (designed to stay <1k rows), so we just re-read it on
    # each filter pass for correctness; cost is microseconds.
    try:
        sigs = fact_store.list_signatures()
    except Exception:
        sigs = []
    try:
        cont_min = float(config.get("forget", "signature_jaccard_min", default=0.5))
    except Exception:
        cont_min = 0.5

    # Pre-compute signature token sets once per call.
    sig_tokens: List[tuple] = []
    for sig in sigs:
        tokens = set(sig.get("tokens") or [])
        if tokens:
            sig_tokens.append((sig.get("id"), tokens))

    # Same tokenizer the signature was built with in forget.py —
    # dedup's stoplist differs, which would score containment
    # across two different vocabularies.
    from .forget import _content_tokens

    kept: List[str] = []
    sig_hits: set = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        # Already-tombstoned lines from Hindsight
        if "[FORGOTTEN" in line:
            continue
        # Exact-key forget marks
        if _is_forgotten(fact_store, line):
            continue
        # Semantic signature filter (candidate-containment)
        if sig_tokens:
            content = stripped
            # strip "1. " number prefix and trailing "| Involving:"
            head, sep, _ = content.partition(" | Involving:")
            content = head if sep else content
            line_tokens = _content_tokens(content)
            if line_tokens:
                matched = False
                for sid, tokens in sig_tokens:
                    inter = line_tokens & tokens
                    cont = len(inter) / len(line_tokens)
                    if cont >= cont_min:
                        matched = True
                        if sid:
                            sig_hits.add(sid)
                        break
                if matched:
                    continue
        kept.append(line)

    # Bump last_match_ts on signatures that did real work — keeps
    # vacuum from dropping useful sigs.
    for sid in sig_hits:
        try:
            fact_store.touch_signature(int(sid))
        except Exception:
            pass

    return "\n".join(kept)


def apply_conflict_resolver(sections: List[str]) -> List[str]:
    """If we detect a contradiction between the user profile and a fact,
    annotate both inline. Best-effort; rule-based detector."""
    if len(sections) < 3:
        return sections
    anchor, profile, facts = sections[0], sections[1], sections[2]
    profile_lines = [ln for ln in profile.splitlines() if ln.startswith("- ")]
    annotated_facts: List[str] = []
    today = today_iso()
    for fact_line in facts.splitlines():
        if not fact_line.strip() or fact_line.startswith("#"):
            annotated_facts.append(fact_line)
            continue
        conflict = False
        for profile_line in profile_lines:
            if is_contradiction(fact_line, profile_line):
                a, b = label_pair(
                    fact_line,
                    {"label": "Hindsight", "when": today},
                    profile_line,
                    {"label": "Honcho profile"},
                )
                annotated_facts.append(a)
                annotated_facts.append(b)
                conflict = True
                break
        if not conflict:
            annotated_facts.append(fact_line)
    return [anchor, profile, "\n".join(annotated_facts)]


def truncate_recall_query(query: str) -> str:
    """Trim query so it never trips Hindsight's 500-token recall limit.
    Prefers to cut at a word boundary near the end."""
    if not query:
        return query
    if len(query) <= RECALL_QUERY_MAX_CHARS:
        return query
    cut = query[:RECALL_QUERY_MAX_CHARS]
    last_space = cut.rfind(" ")
    if last_space > RECALL_QUERY_MAX_CHARS * 0.7:
        cut = cut[:last_space]
    return cut
