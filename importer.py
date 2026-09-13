"""Plan item 8 — bulk import of historical session transcripts.

Sends the last N days of ~/.hermes/sessions/*.jsonl into Hindsight, tagged
with the real date from each filename and `bulk_import:true`. Resumable
via its own cursor (separate from recovery's per-process cursor).

CLI: hermes mnemosyne import [--days 90] [--min-turns 5]
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config
from .recovery import (
    _filename_to_iso_date,
    _iter_turns,
    _list_session_files,
    _pair_user_assistant,
    _retain_pair,
)

logger = logging.getLogger(__name__)


def _import_cursor_path() -> Path:
    return config.plugin_dir() / "import_cursor.json"


def _load_import_cursor() -> Dict[str, Any]:
    p = _import_cursor_path()
    if not p.exists():
        return {"completed_files": [], "in_progress_file": None, "in_progress_pair": 0}
    try:
        with p.open() as f:
            return json.load(f) or {"completed_files": [], "in_progress_file": None,
                                    "in_progress_pair": 0}
    except Exception:
        return {"completed_files": [], "in_progress_file": None, "in_progress_pair": 0}


def _save_import_cursor(cursor: Dict[str, Any]) -> None:
    p = _import_cursor_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(cursor, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def _filter_files_by_age(files: List[Path], *, days: int) -> List[Path]:
    cutoff = date.today() - timedelta(days=days)
    out: List[Path] = []
    for f in files:
        iso = _filename_to_iso_date(f.name)
        if not iso:
            continue
        try:
            d = date.fromisoformat(iso)
        except Exception:
            continue
        if d >= cutoff:
            out.append(f)
    return out


def _read_pairs(path: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Parse one transcript into user→assistant pairs.

    Callers keep the result. This used to be a _count_pairs() that threw the
    parse away to return a length, so every transcript was read and JSON-parsed
    twice — once to apply min_turns, once to send it.
    """
    return _pair_user_assistant([r for _, r in _iter_turns(path)])


def run_import(
    hindsight_provider: Any,
    *,
    days: Optional[int] = None,
    min_turns: Optional[int] = None,
    on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run bulk import. Returns summary dict.

    on_progress(stage, info_dict) — optional callback for live progress.
    """
    if hindsight_provider is None:
        return {"error": "hindsight unavailable", "imported_pairs": 0}

    days = days if days is not None else int(config.get("import", "default_days", default=90))
    min_turns = min_turns if min_turns is not None else int(
        config.get("import", "min_turns", default=5)
    )

    files = _list_session_files()
    files = _filter_files_by_age(files, days=days)

    cursor = _load_import_cursor()
    completed = set(cursor.get("completed_files") or [])
    in_progress_file = cursor.get("in_progress_file")
    in_progress_pair = int(cursor.get("in_progress_pair") or 0)

    todo: List[Tuple[Path, List[Tuple[Dict[str, Any], Dict[str, Any]]]]] = []
    for f in files:
        if f.name in completed:
            continue
        pairs = _read_pairs(f)
        if len(pairs) < min_turns:
            completed.add(f.name)
            continue
        todo.append((f, pairs))

    cursor["completed_files"] = sorted(completed)
    _save_import_cursor(cursor)

    if not todo:
        return {"imported_pairs": 0, "imported_files": 0, "skipped": "nothing to do"}

    total_pairs = sum(len(pairs) for _, pairs in todo)
    imported = 0
    files_done = 0

    if on_progress:
        on_progress("start", {"files": len(todo), "pairs": total_pairs,
                               "days": days, "min_turns": min_turns})

    try:
        for f, pairs in todo:
            iso = _filename_to_iso_date(f.name) or "unknown"
            n = len(pairs)

            start = 0
            if f.name == in_progress_file:
                start = in_progress_pair
                in_progress_file = None  # only resume the first matching file

            for i, (u, a) in enumerate(pairs[start:], start=start):
                ok = _retain_pair(
                    hindsight_provider, u, a, iso_date=iso,
                    extra_tags=["bulk_import:true"],
                )
                if ok:
                    imported += 1

                # Cursor checkpoint every 25 pairs in case of interruption.
                if (i + 1) % 25 == 0:
                    cursor["in_progress_file"] = f.name
                    cursor["in_progress_pair"] = i + 1
                    _save_import_cursor(cursor)
                    if on_progress:
                        on_progress("checkpoint", {
                            "file": f.name, "pairs_done": i + 1, "pairs_total": n,
                            "imported": imported,
                        })

            completed.add(f.name)
            files_done += 1
            cursor["completed_files"] = sorted(completed)
            cursor["in_progress_file"] = None
            cursor["in_progress_pair"] = 0
            _save_import_cursor(cursor)
            if on_progress:
                on_progress("file_done", {
                    "file": f.name, "files_done": files_done,
                    "files_total": len(todo), "imported": imported,
                })

    except KeyboardInterrupt:
        if on_progress:
            on_progress("interrupted", {"imported": imported})
        return {"imported_pairs": imported, "imported_files": files_done,
                "interrupted": True}

    return {"imported_pairs": imported, "imported_files": files_done,
            "interrupted": False}
