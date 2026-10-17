"""Pure helpers for the Streamlit UI (no streamlit imports, easy to test)."""

import difflib
import os
import re
from collections import Counter
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

STAGES = [
    "Vault Reader",
    "Concept Extractor",
    "Concept Verifier",
    "Relationship Extractor",
    "Relationship Verifier",
    "Link Writer",
    "Summary Reporter",
]

_STAGE_RE = re.compile(r"---\s*\[([^\]]+)\]\s*---")


def detect_stage(message: str) -> Optional[str]:
    """Returns the stage name if the log line is a '--- [Stage] ---' header."""
    m = _STAGE_RE.search(message or "")
    if not m:
        return None
    name = m.group(1).strip()
    return name if name in STAGES else None


def stage_states(current: Optional[str], finished: bool = False, failed: bool = False) -> List[str]:
    """Per-stage state for the tracker: 'pending', 'running', 'done' or 'failed'."""
    idx = STAGES.index(current) if current in STAGES else -1
    states = []
    for i in range(len(STAGES)):
        if (finished and not failed) or i < idx:
            states.append("done")
        elif i == idx:
            states.append("failed" if failed else "running")
        else:
            states.append("pending")
    return states


def count_excluded_dirs(vault: str, excluded: Iterable[str]) -> int:
    """Counts vault folders matching the excluded-folder patterns."""
    pats = {d.strip().lower().strip("/\\") for d in excluded if d and d.strip()}
    if not pats:
        return 0
    n = 0
    for root, dirs, _ in os.walk(vault):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        rel_root = os.path.relpath(root, vault).replace("\\", "/").lower()
        for d in list(dirs):
            rel = d.lower() if rel_root == "." else f"{rel_root}/{d.lower()}"
            if d.lower() in pats or rel in pats:
                n += 1
                dirs.remove(d)
    return n


def note_diff(before: str, after: str, title: str = "note") -> str:
    return "".join(
        difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            (after or "").splitlines(keepends=True),
            fromfile=f"{title} (before)",
            tofile=f"{title} (after)",
        )
    )


def strip_md(name) -> str:
    name = str(name or "")
    return name[:-3] if name.lower().endswith(".md") else name


def link_rows(links: List[dict]) -> List[dict]:
    return [
        {
            "From": strip_md(l.get("from_note")),
            "To": strip_md(l.get("to_note")),
            "Relationship": l.get("relationship"),
            "Evidence": l.get("evidence"),
            "Score": l.get("score"),
        }
        for l in links
    ]


def filter_rows(rows: List[dict], query: str) -> List[dict]:
    q = (query or "").strip().lower()
    if not q:
        return rows
    return [r for r in rows if any(q in str(v).lower() for v in r.values() if v is not None)]


def _graph_selection(links: List[dict], query: str, top_n: int):
    """Shared filter for the graph views: (edge links kept, degree, matched names)."""
    pairs = []
    for l in links:
        a, b = strip_md(l.get("from_note")), strip_md(l.get("to_note"))
        if a and b and a != b:
            pairs.append((a, b, l))
    degree = Counter()
    for a, b, _ in pairs:
        degree[a] += 1
        degree[b] += 1
    keep = {n for n, _ in degree.most_common(top_n)} if top_n else set(degree)
    q = (query or "").strip().lower()
    matched = set()
    if q:
        matched = {n for n in degree if q in n.lower()}
        keep = set(matched)
        for a, b, _ in pairs:  # include direct neighbours of matches
            if a in matched or b in matched:
                keep.update((a, b))
    return [(a, b, l) for a, b, l in pairs if a in keep and b in keep], degree, matched


def graph_data(links: List[dict], query: str = "", top_n: int = 40) -> dict:
    """Nodes/edges for the interactive graph, de-duplicated per (from, to) pair.

    Returns {"nodes": [{id, label, degree, match}], "edges": [{from, to,
    relationship, evidence, score}]}; degree counts all links in the vault.
    """
    kept, degree, matched = _graph_selection(links, query, top_n)
    edges = {}
    for a, b, l in kept:
        score = l.get("score")
        try:
            score = round(float(score), 3) if score is not None else None
        except (TypeError, ValueError):
            score = None
        cur = edges.get((a, b))
        if cur is None:
            edges[(a, b)] = {
                "from": a, "to": b,
                "relationship": str(l.get("relationship") or ""),
                "evidence": str(l.get("evidence") or ""),
                "score": score,
            }
        elif score is not None and (cur["score"] is None or score > cur["score"]):
            cur["score"] = score
    names = sorted({n for e in edges for n in e})
    return {
        "nodes": [{"id": n, "label": n, "degree": degree[n], "match": n in matched} for n in names],
        "edges": [edges[k] for k in sorted(edges)],
    }


def format_timestamp(value, now: Optional[datetime] = None) -> Tuple[str, str]:
    """ISO timestamp -> ('Oct 17, 2026 · 21:14', '3h ago'); falls back to the raw text."""
    try:
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return str(value or ""), ""
    now = now or datetime.now(dt.tzinfo)
    secs = (now - dt).total_seconds()
    if secs < 0:
        age = ""
    elif secs < 60:
        age = "just now"
    elif secs < 3600:
        age = f"{int(secs // 60)}m ago"
    elif secs < 86400:
        age = f"{int(secs // 3600)}h ago"
    else:
        age = f"{int(secs // 86400)}d ago"
    return f"{dt:%b} {dt.day}, {dt:%Y} · {dt:%H:%M}", age
