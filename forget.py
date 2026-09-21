"""Plan item 11 — explicit forgetting.

Two paths:

1. Point delete via built-in memory: when the agent calls
   `memory(action='remove', target=user, old_text=...)`, the
   `on_memory_write(action='remove')` hook on MnemosyneMemoryProvider
   forwards to `forget_text(...)` here, which marks the canonical key
   in fact_store and writes a tombstone to Hindsight.

2. Fuzzy forget via tool/CLI: `memory_forget(query)` does a recall,
   runs the candidates through `forget_text(...)` for each match.

We can't physically delete from Hindsight (no public API). What we do:
- Mark the canonical key in fact_store as forgotten — at recall time,
  Mnemosyne filters out any candidate whose normalized text matches a
  forgotten key.
- Write a tombstone to Hindsight: a new memory whose body is
  `[FORGOTTEN <date>] <original>` with tags `forgotten:<date>` and
  `supersedes:<key>`. This lets recall callers that bypass our filter
  still see the marker.
- Append to `forgotten.jsonl` for an audit trail.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .text_utils import SPEAKER_STOP, containment, content_tokens
from .fact_store import FactStore, _canonical_key, today_iso

logger = logging.getLogger(__name__)


def _audit_log_path() -> Path:
    return config.plugin_dir() / "forgotten.jsonl"


def _append_audit(entry: Dict[str, Any]) -> None:
    path = _audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug("mnemosyne.forget: audit log write failed: %s", exc)


def is_forgotten(fact_store: FactStore, text: str) -> bool:
    """Used by the recall filter — does this candidate match a forgotten key?"""
    return fact_store.is_forgotten(text)


def forget_text(
    fact_store: FactStore,
    hindsight_provider: Optional[Any],
    text: str,
    *,
    reason: str = "user_request",
) -> Dict[str, Any]:
    """Mark a single fact as forgotten. Returns audit entry."""
    key = _canonical_key(text)
    when = today_iso()

    fact_store.mark_forgotten(text)

    tombstone_written = False
    if hindsight_provider is not None:
        try:
            hindsight_provider.handle_tool_call(
                "hindsight_retain",
                {
                    "content": f"[FORGOTTEN {when}] {text}",
                    "tags": [f"forgotten:{when}", f"supersedes:{key}", f"reason:{reason}"],
                },
            )
            tombstone_written = True
        except Exception as exc:
            logger.debug("mnemosyne.forget: tombstone write failed: %s", exc)

    entry = {
        "ts": when,
        "canonical_key": key,
        "text": text,
        "reason": reason,
        "tombstone_written": tombstone_written,
    }
    _append_audit(entry)
    return entry


# Tokenizer and stoplist live in text_utils now, shared with dedup.py: the
# signature built here is matched by the read-side filter, so the two sides
# must normalise identically. Kept under the old private names.
def _content_tokens(text: str) -> set:
    return content_tokens(text, SPEAKER_STOP)


def _containment(query_tokens: set, cand_tokens: set) -> float:
    """How much of the query is present in the candidate.

    Jaccard punishes long candidates against short queries (a 1-token query
    matched in a 20-token candidate scores 0.05) — that breaks the forget UX
    when the agent passes a single concrete keyword like "Barsik".
    """
    return containment(query_tokens, cand_tokens)


def _apply_forget(
    fact_store: FactStore,
    hindsight_provider: Optional[Any],
    chosen_texts: List[str],
    query: str,
) -> Dict[str, Any]:
    """Apply a confirmed forget to `chosen_texts`.

    Single implementation shared by every caller (tool and CLI alike) so
    the two can never drift again: before this existed the CLI path went
    through ``forget_text`` per item and never registered a semantic
    signature, so ``hermes mnemosyne forget`` hid only exact canonical
    keys while ``memory_forget`` also hid paraphrases.

    Phase 1 is synchronous and fast: mark canonical keys forgotten and
    register ONE semantic signature for the whole op. The signature is
    what actually drops paraphrases at recall time, so the user sees the
    effect immediately. Phase 2 (tombstones) is fire-and-forget.
    """
    op_tokens: set = set()
    for text in chosen_texts:
        try:
            fact_store.mark_forgotten(text)
        except Exception as exc:
            logger.debug(
                "mnemosyne.forget: mark_forgotten failed for %r: %s", text[:80], exc
            )
        op_tokens |= _content_tokens(text)
    # Also fold in query tokens so the sig fires on future paraphrases
    # that share the user's intent vocabulary.
    op_tokens |= _content_tokens(query)

    sig_id = 0
    if op_tokens:
        try:
            merge = float(config.get("forget", "signature_merge_jaccard",
                                     default=0.7))
            sig_id = fact_store.add_signature(
                list(op_tokens),
                examples=chosen_texts[:5],
                query=query,
                merge_jaccard=merge,
            )
        except Exception as exc:
            logger.debug("mnemosyne.forget: add_signature failed: %s", exc)

    audit_now = today_iso()
    audit_entries: List[Dict[str, Any]] = []
    for text in chosen_texts:
        entry = {
            "ts": audit_now,
            "canonical_key": _canonical_key(text),
            "text": text,
            "reason": "forget_by_query",
            "tombstone_written": False,  # filled in by background worker
            "sig_id": sig_id,
        }
        _append_audit(entry)
        audit_entries.append(entry)

    # Phase 2 — fire-and-forget tombstones. Each retain takes ~5-15s
    # because Hindsight runs them through embedding + reranker. We
    # don't block the agent on this; the signature filter already hides
    # the records from recall.
    write_ts = bool(config.get("forget", "write_tombstones", default=True))
    async_ts = bool(config.get("forget", "write_tombstones_async", default=True))
    if write_ts and hindsight_provider is not None and chosen_texts:
        if async_ts:
            from . import _spawn_tombstone_writer  # late import — set in __init__.py
            _spawn_tombstone_writer(hindsight_provider, chosen_texts, audit_now)
        else:
            for text in chosen_texts:
                _write_tombstone(hindsight_provider, text, audit_now)

    return {
        "forgotten": audit_entries,
        "ts": audit_now,
        "signature_id": sig_id,
        "tombstones": ("queued" if (write_ts and async_ts)
                       else "written" if write_ts else "skipped"),
    }


def _preview_ttl_s() -> float:
    try:
        return float(config.get("forget", "preview_ttl_s", default=3600))
    except Exception:
        return 3600.0


def forget_by_query(
    fact_store: FactStore,
    hindsight_provider: Optional[Any],
    query: str,
    *,
    confirmed: bool = False,
    preview_token: Optional[str] = None,
    indices: Optional[List[int]] = None,
    max_items: int = 30,
    min_overlap: float = 0.0,
) -> Dict[str, Any]:
    """Recall candidates matching `query`, mark the chosen ones as forgotten.

    Two steps, and the second one never re-runs recall:

      * Step 1 (``confirmed=False``) is a **dry-run**. It recalls
        candidates, persists that exact list under an opaque
        ``preview_token``, and returns both. Nothing is marked.
      * Step 2 (``confirmed=True`` + that ``preview_token``) applies the
        forget to the **pinned snapshot**. ``indices`` (1-based) selects
        a subset of it.

    Why the token is mandatory: Hindsight ranks recall by reranker score,
    and that score drifts between calls. Re-recalling in step 2 — what
    this function used to do — means the user can approve list A and the
    system forgets list B. That is the failure mode behind the May 6
    incident. Pinning removes it: step 2 can only ever forget rows the
    user actually saw.

    ``confirmed=True`` without a live token forgets nothing and returns an
    error, so a caller that skips the preview fails safe rather than
    deleting a fresh, unreviewed recall.

    ``max_items`` defaults to 30, matching what recall returns.
    ``min_overlap`` defaults to 0 (off). Raise it to filter out candidates
    whose query tokens aren't present (containment, not Jaccard).
    """
    # ---- Step 2: apply a pinned preview. No recall happens here. ----
    if confirmed:
        if not preview_token:
            return {
                "error": (
                    "preview_token required. Call memory_forget without "
                    "confirmed first, show the candidates to the user, then "
                    "re-invoke with confirmed=true and the preview_token from "
                    "that response. Nothing was forgotten."
                ),
                "forgotten": [],
                "candidates": [],
            }
        pinned = fact_store.load_preview(preview_token, max_age_s=_preview_ttl_s())
        if pinned is None:
            return {
                "error": (
                    "preview_token is unknown or expired. Re-run the preview "
                    "and confirm against the fresh token. Nothing was forgotten."
                ),
                "forgotten": [],
                "candidates": [],
            }
        candidates = pinned["candidates"]
        query = pinned["query"] or query

        if indices:
            picked = []
            for raw_i in indices:
                try:
                    i = int(raw_i)
                except (TypeError, ValueError):
                    continue
                if 1 <= i <= len(candidates):
                    picked.append(candidates[i - 1])
            chosen_texts = [c["text"] for c in picked]
        else:
            chosen_texts = [c["text"] for c in candidates]

        result = _apply_forget(fact_store, hindsight_provider, chosen_texts, query)
        result["candidates"] = [c["text"] for c in candidates]
        result["query"] = query
        result["note"] = (
            "Marked as forgotten in the read-side filter — these "
            "memories will not appear in future recall results "
            "regardless of paraphrasing. Tombstone records in the "
            "Hindsight bank are written in the background as a "
            "secondary safety net (no need to wait for them)."
        )
        return result

    # ---- Step 1: recall, score, pin. ----
    if hindsight_provider is None:
        return {"error": "hindsight unavailable", "forgotten": [], "candidates": []}

    try:
        raw = hindsight_provider.handle_tool_call(
            "hindsight_recall",
            {"query": query, "max_tokens": 4096},
        )
    except Exception as exc:
        return {"error": f"recall failed: {exc}", "forgotten": [], "candidates": []}

    raw_candidates = _extract_candidates(raw)
    query_tokens = _content_tokens(query)

    # Score every candidate by query-containment (how much of the query
    # is present), drop any below threshold. Containment, not Jaccard,
    # so a short query like "Barsik" matches long candidates that
    # mention it. Default threshold is 0 — show everything recall would
    # show, let the agent + user decide.
    scored: List[Dict[str, Any]] = []
    seen: set = set()
    for c in raw_candidates:
        text = c if isinstance(c, str) else (c.get("text") or c.get("content") or "")
        text = (text or "").strip()
        if not text:
            continue
        ckey = text.lower()
        if ckey in seen:
            continue
        seen.add(ckey)
        overlap = _containment(query_tokens, _content_tokens(text))
        if overlap < min_overlap:
            continue
        scored.append({"text": text, "overlap": round(overlap, 3)})

    scored.sort(key=lambda x: x["overlap"], reverse=True)
    candidates = scored[:max_items]

    token = secrets.token_hex(8)
    try:
        fact_store.purge_previews(max_age_s=_preview_ttl_s())
        fact_store.save_preview(token, query, candidates)
    except Exception as exc:
        logger.warning("mnemosyne.forget: could not pin preview: %s", exc)
        return {
            "error": (
                f"could not persist the preview ({exc}); refusing to offer a "
                "forget that cannot be confirmed safely."
            ),
            "forgotten": [],
            "candidates": [],
        }

    return {
        "preview": True,
        "preview_token": token,
        "forgotten": [],
        "candidates": [
            {"index": i + 1, "text": c["text"], "overlap": c["overlap"]}
            for i, c in enumerate(candidates)
        ],
        "instructions": (
            "DRY-RUN. Show these candidates to the user verbatim, get explicit "
            "confirmation, then re-invoke memory_forget with confirmed=true AND "
            f"preview_token=\"{token}\" (optionally indices=[1,3,...] to pick a "
            "subset). The confirm step applies exactly this list — it does not "
            "search again. Without confirmed=true nothing is forgotten."
        ),
        "query": query,
        "min_overlap": min_overlap,
    }


def _write_tombstone(hindsight_provider: Any, text: str, when: str) -> bool:
    """Single tombstone retain. Blocking. Logs and returns False on error."""
    if hindsight_provider is None:
        return False
    try:
        hindsight_provider.handle_tool_call(
            "hindsight_retain",
            {
                "content": f"[FORGOTTEN {when}] {text}",
                "tags": [f"forgotten:{when}", "reason:forget_by_query"],
            },
        )
        return True
    except Exception as exc:
        logger.debug("mnemosyne.forget: tombstone write failed: %s", exc)
        return False


def _extract_candidates(raw_recall_result: Any) -> List[Any]:
    """Best-effort extraction of candidate strings from Hindsight's recall.

    Hindsight (via the hermes plugin) returns a JSON string of the shape
    ``{"result": "1. fact A\n2. fact B\n…"}`` — a numbered plain-text list,
    not a structured array. We also keep the older list/dict shapes for
    forward-compat with future Hindsight versions.
    """
    if isinstance(raw_recall_result, list):
        return raw_recall_result
    if isinstance(raw_recall_result, str):
        try:
            data = json.loads(raw_recall_result)
        except Exception:
            return _split_numbered_lines(raw_recall_result)
        return _extract_from_dict(data)
    if isinstance(raw_recall_result, dict):
        return _extract_from_dict(raw_recall_result)
    return []


def _extract_from_dict(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    # Numbered-list format used by hermes hindsight plugin
    result = data.get("result")
    if isinstance(result, str) and result.strip():
        items = _split_numbered_lines(result)
        if items:
            return items
    if isinstance(result, list) and result:
        return result
    # Other recall shapes (future / direct API)
    for key in ("memories", "results", "items", "matches", "data"):
        v = data.get(key)
        if isinstance(v, list) and v:
            return v
    return []


_NUMBER_PREFIX = re.compile(r"^\s*\d+\.\s*", flags=re.UNICODE)


def _split_numbered_lines(text: str) -> List[str]:
    """Parse "1. foo\n2. bar | Involving: …\n3. baz" into clean fact strings.

    Drops the empty-result placeholder. Strips per-line metadata after the
    " | Involving: …" separator that Hindsight appends."""
    if not text:
        return []
    if "No relevant memories found" in text:
        return []
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _NUMBER_PREFIX.sub("", line)
        head, sep, _ = line.partition(" | ")
        out.append(head if sep else line)
    # Dedup while preserving order — Hindsight often returns the same fact
    # multiple times in different surface forms.
    seen: set = set()
    deduped: List[str] = []
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


# Tool schema served via the curated tools surface
MEMORY_FORGET_SCHEMA = {
    "name": "memory_forget",
    "description": (
        "TWO-STEP forget tool. First call WITHOUT confirmed=true returns a "
        "preview list of candidate facts plus a preview_token — show the list "
        "to the user verbatim and ask them to confirm. Then call again with "
        "confirmed=true AND that preview_token (optionally indices=[1,3,...] "
        "to pick specific candidates). The second call forgets exactly the "
        "previewed list; it does not search again, so what the user approved "
        "is what gets forgotten. Forgotten facts will not appear in future "
        "memory_recall results. Use ONLY when the user explicitly asks to "
        "forget something specific."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to forget. Phrase it CONCRETELY — the more specific "
                    "the words, the more accurate the match. Vague queries "
                    "('cat', 'project') will match too broadly and the "
                    "preview will reject most candidates."
                ),
            },
            "confirmed": {
                "type": "boolean",
                "description": (
                    "Set to true ONLY after the user has explicitly approved "
                    "the candidates from the previous preview call. Must be "
                    "sent together with preview_token. Default false "
                    "(preview only)."
                ),
            },
            "preview_token": {
                "type": "string",
                "description": (
                    "The preview_token returned by the preview call. Required "
                    "with confirmed=true: it identifies the exact candidate "
                    "list the user approved, which is what gets forgotten. "
                    "Without it nothing is forgotten."
                ),
            },
            "indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Optional 1-based indices into the preview identified by "
                    "preview_token, to narrow the forget to a subset. Omit to "
                    "forget every candidate in that preview."
                ),
            },
            "max_items": {
                "type": "integer",
                "description": "Max candidates to consider (default 30, matching recall).",
            },
        },
        "required": ["query"],
    },
}
