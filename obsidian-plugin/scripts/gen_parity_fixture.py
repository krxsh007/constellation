"""Regenerates tests/fixtures/python-parity.json from the Python reference
implementation so the TypeScript port can be checked against it.

Run from the repo root:  python obsidian-plugin/scripts/gen_parity_fixture.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from constellation.core.utils import _fallback_parse, _repair_truncated_json  # noqa: E402
from constellation.core.state import RelationshipExtractionOutput  # noqa: E402
from constellation.nodes.concepts import _normalise_tags  # noqa: E402
from constellation.nodes.io import build_effective_links, render_note, strip_auto_sections  # noqa: E402

STRIP_INPUTS = [
    "# A\n\nBody text.",
    "# A\n\nBody.\n\n## Tags\n#x #y\n\n## Related Links\n- [[B]]\n- [[C]]\n",
    "# A\n\nBody.\n\n## Tags\n#x #y\n\nMore thoughts.\n",
    "# A\n\nBody.\n\n## Related Links\n- [[B]]\n\n## My Notes\nhello\n",
    "# A\n\n## Related Links\n- [[B]]\n- a manual todo\n",
    "# A\n\nBody.\n\n## Tags\n#x\nMore.\n",
    "# A\n\n```py\n# not a tag\n```\n\n## Tags\n#x\n",
    "# A\n\nBody.\n\n## Related Links\n- [[B]] — Core runtime platform\n- [[C]] - Secondary service\n",
    "## Tags\n#x\n",
    "",
    "---\ntags: [a]\n---\n# A\n\n## Related Links\n- [[Folder/B|B]]\n- [[C]]: why\n\n\n",
    "# A\r\n\r\nBody\r\n\r\n## Tags\r\n#x #y\r\n",
    "## Related Links\n```\n- [[inside fence]]\n```\n- [[after]]\n",
    "Text\n## Tags\n#a #b\n## Related Links\n- [[X]]\n  \n- [[Y]]\ntrailing user line\n\n",
    "# A\n\n## Tags\n#a ##b\n\n## Related Links\n-[[NoSpace]]\n",
]

RENDER_INPUTS = [
    {"base": "# A\n\nBody.", "tags": ["#x", "#y"], "targets": [{"target": "B", "evidence": "", "score": 0.0}], "max": 5, "annotate": False},
    {"base": "", "tags": ["#tag"], "targets": [], "max": 5, "annotate": False},
    {"base": "", "tags": [], "targets": [], "max": 5, "annotate": False},
    {"base": "# Root\n\n\n", "tags": None, "targets": [
        {"target": t, "evidence": f"ev {t}", "score": s} for t, s in
        [("E", 0.9), ("A", 0.1), ("C", 0.5), ("B", 0.5), ("D", 0.7), ("A", 0.3)]
    ], "max": 3, "annotate": True},
    {"base": "# Root", "tags": ["#t"], "targets": [
        {"target": "Docker", "evidence": "Platform\nfor containers", "score": 0.2},
        {"target": "Kubernetes", "evidence": "", "score": 0.4},
    ], "max": 0, "annotate": True},
]

LINKS = [
    {"from_note": "a.md", "to_note": "b.md", "evidence": "ab", "score": 0.8},
    {"from_note": "a.md", "to_note": "c.md", "evidence": "", "score": 0.6},
    {"from_note": "c.md", "to_note": "a.md", "evidence": "ca", "score": 0.9},
    {"from_note": "b.md", "to_note": "d.md", "evidence": "bd", "score": 0.1},
    {"from_note": "d.md", "to_note": "d.md", "evidence": "self", "score": 1.0},
    {"from_note": "a.md", "to_note": "e.md", "evidence": "ae", "score": 0.6},
]


def link(source="Docker", target="Kubernetes", **over):
    base = {"source": source, "target": target, "relationship": "uses", "evidence": "K8s schedules containers.",
            "from_note": "a.md", "to_note": "b.md"}
    base.update(over)
    return base


full = json.dumps({"links": [link(), link("Redis", "Postgres")]})
REPAIR_INPUTS = [
    full[: full.index('{"source": "Redis"')] + '{"source": "Redis", "target": "Post',
    json.dumps({"links": [link(evidence="a very long ev")]})[:-14],
    '{"links": [' + json.dumps(link(evidence='he said \\"hi\\" then left')) + ']}',
    '{"links": [{"source": "A", "target": "B", "relationship": "uses", "evidence": "x", "from_note": "a", "to_note": ',
    '{"concepts": [{"concept_name": "X", "tags": ["a", "b"',
]


def parse_sources(raw):
    try:
        return [l.source for l in _fallback_parse(raw, RelationshipExtractionOutput).links]
    except Exception:
        return None


fixture = {
    "strip": [{"input": s, "output": strip_auto_sections(s)} for s in STRIP_INPUTS],
    "render": [
        {**r, "output": render_note(r["base"], r["tags"], r["targets"], max_links=r["max"], annotate=r["annotate"])}
        for r in RENDER_INPUTS
    ],
    "tags": {
        "input": ["#machine-learning", "deep learning", "#AI,", "PYTHON!", "#tech/web", "", "##double", "  spaced  out  "],
        "output": _normalise_tags(["#machine-learning", "deep learning", "#AI,", "PYTHON!", "#tech/web", "", "##double", "  spaced  out  "]),
    },
    "effective": [
        {"links": LINKS, "backlinks": b, "max": m, "output": build_effective_links(LINKS, backlinks=b, max_links=m)}
        for b in (False, True) for m in (0, 1, 2)
    ],
    "repair": [{"input": s, "output": _repair_truncated_json(s), "sources": parse_sources(s)} for s in REPAIR_INPUTS],
}

out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "python-parity.json"
out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}")
