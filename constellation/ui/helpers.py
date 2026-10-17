"""Pure helpers for the Streamlit UI (no streamlit imports, easy to test)."""

import difflib
import os
import re
from collections import Counter
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


def stage_markdown(current: Optional[str], finished: bool = False, failed: bool = False) -> str:
    """Renders the 7-stage tracker as markdown (pending / running / done)."""
    idx = STAGES.index(current) if current in STAGES else -1
    rows = []
    for i, name in enumerate(STAGES):
        if finished or i < idx:
            icon = "✅"
        elif i == idx:
            icon = "❌" if failed else "⏳"
        else:
            icon = "⚪"
        label = f"**{name}**" if i == idx and not finished else name
        rows.append(f"{icon} {label}")
    return "  \n".join(rows)


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


def graph_dot(links: List[dict], query: str = "", top_n: int = 40) -> Tuple[str, int, int]:
    """Builds a DOT string from links. Returns (dot, nodes_shown, edges_shown)."""
    edges = [(strip_md(l.get("from_note")), strip_md(l.get("to_note"))) for l in links]
    edges = [(a, b) for a, b in edges if a and b and a != b]
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    keep = {n for n, _ in degree.most_common(top_n)} if top_n else set(degree)
    q = (query or "").strip().lower()
    if q:
        matched = {n for n in degree if q in n.lower()}
        keep = set(matched)
        for a, b in edges:  # include direct neighbours of matches
            if a in matched or b in matched:
                keep.update((a, b))
    shown = sorted({(a, b) for a, b in edges if a in keep and b in keep})
    nodes = {n for e in shown for n in e}

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "digraph G {",
        '  graph [bgcolor="transparent", overlap=false, splines=true];',
        '  node [shape=ellipse, style=filled, fillcolor="#7c5cff", fontcolor="white", '
        'color="#5b3fd6", fontsize=11];',
        '  edge [color="#9a9aaa", arrowsize=0.6];',
    ]
    for n in sorted(nodes):
        lines.append(f'  "{esc(n)}";')
    for a, b in shown:
        lines.append(f'  "{esc(a)}" -> "{esc(b)}";')
    lines.append("}")
    return "\n".join(lines), len(nodes), len(shown)
