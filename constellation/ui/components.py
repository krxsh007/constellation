"""HTML fragments for the Streamlit UI (pure functions, no streamlit imports).

Everything that ends up inside an HTML string is escaped here, so callers can
pass note titles, log lines and diffs straight through.
"""

import base64
import json
import re
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from constellation.ui.helpers import STAGES, stage_states

UI_DIR = Path(__file__).resolve().parent

# Small stroke icons (24x24, Lucide-style paths), coloured with currentColor.
_ICON_PATHS = {
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "sliders": '<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/>'
               '<path d="M20 21v-5"/><path d="M20 12V3"/><path d="M2 14h4"/><path d="M10 8h4"/><path d="M18 16h4"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1'
              'c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "archive": '<rect width="20" height="5" x="2" y="3" rx="1"/><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8"/>'
               '<path d="M10 12h4"/>',
    "graph": '<circle cx="5" cy="6" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="12" cy="18" r="2"/>'
             '<path d="m6.8 7 4.2 9"/><path d="m17.4 6.4-4.6 9.8"/><path d="M7 6h10"/>',
    "sparkles": '<path d="M9.94 14.06 8.5 19l-1.44-4.94L2 12.5l5.06-1.44L8.5 6l1.44 5.06L15 12.5z"/>'
                '<path d="M19 3v4"/><path d="M21 5h-4"/>',
    "key": '<circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/>'
           '<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/>'
           '<path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>',
    "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>',
}


def _svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def icon(name: str, size: int = 16, cls: str = "cx-icon") -> str:
    """An inline icon tinted with the text colour.

    st.html sanitizes inline <svg>, so the glyph is applied as a CSS mask over
    a span whose background is currentColor.
    """
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" '
           'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
           f'{_ICON_PATHS.get(name, "")}</svg>')
    uri = _svg_data_uri(svg)
    return (f'<span class="{cls}" aria-hidden="true" style="width:{int(size)}px;height:{int(size)}px;'
            f'-webkit-mask-image:url({uri});mask-image:url({uri})"></span>')


def svg_img(svg: str, width: int, height: int, cls: str = "") -> str:
    """Multi-colour artwork as an <img> (inline <svg> doesn't survive st.html's sanitizer)."""
    if "xmlns=" not in svg:
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    c = f' class="{cls}"' if cls else ""
    return f'<img{c} src="{_svg_data_uri(svg)}" width="{int(width)}" height="{int(height)}" alt="">'


_LOGO_SVG = """<svg viewBox="0 0 48 48" width="48" height="48">
<g stroke="#7f838d" stroke-width="1.5" fill="none" stroke-linecap="round">
<path d="M8 33 L18 14 L30 21 L40 9"/><path d="M30 21 L35 37"/></g>
<g fill="#ececef"><circle cx="8" cy="33" r="3"/><circle cx="18" cy="14" r="3.5"/>
<circle cx="30" cy="21" r="3"/><circle cx="35" cy="37" r="3"/></g>
<circle cx="40" cy="9" r="4" fill="#9d8ff8"/></svg>"""
LOGO_SVG = svg_img(_LOGO_SVG, 30, 30, "cx-logo")


def load_css() -> str:
    """The theme stylesheet, or '' when it isn't shipped (e.g. a non-editable install)."""
    try:
        return (UI_DIR / "theme.css").read_text(encoding="utf-8")
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Small building blocks
# --------------------------------------------------------------------------- #
def chip(label: str, state: str = "muted", title: str = "") -> str:
    """A status pill. state: ok | warn | err | muted | accent."""
    t = f' title="{escape(title)}"' if title else ""
    return f'<span class="cx-chip cx-chip--{escape(state)}"{t}><i></i>{escape(label)}</span>'


def section_label(text: str, icon_name: Optional[str] = None) -> str:
    ic = icon(icon_name, 14) if icon_name else ""
    return f'<div class="cx-section">{ic}<span>{escape(text)}</span></div>'


def card_header(title: str, subtitle: str = "", icon_name: Optional[str] = None, right: str = "") -> str:
    """Title row for a card. `right` is trusted HTML (e.g. chips)."""
    ic = f'<span class="cx-card-icon">{icon(icon_name, 16)}</span>' if icon_name else ""
    sub = f'<div class="cx-card-sub">{escape(subtitle)}</div>' if subtitle else ""
    return (f'<div class="cx-card-head">{ic}<div class="cx-card-titles"><div class="cx-card-title">'
            f'{escape(title)}</div>{sub}</div><div class="cx-card-right">{right}</div></div>')


def callout(text: str, kind: str = "info", icon_name: Optional[str] = None) -> str:
    """Inline notice. kind: info | ok | warn | err. `text` may contain `code` spans."""
    ic = icon(icon_name or {"ok": "check", "warn": "alert", "err": "x"}.get(kind, "info"), 16)
    body = re.sub(r"`([^`]+)`", r"<code>\1</code>", escape(text))
    return f'<div class="cx-callout cx-callout--{escape(kind)}">{ic}<div>{body}</div></div>'


def empty_state(title: str, body: str, art: str = "stars") -> str:
    arts = {
        "stars": """<svg viewBox="0 0 120 80" width="120" height="80" aria-hidden="true">
<g stroke="#33363e" fill="none"><path d="M18 58 L44 26 L72 42 L100 18"/>
<path d="M72 42 L88 66" stroke-dasharray="3 4"/></g><g fill="#7f838d">
<circle cx="18" cy="58" r="3"/><circle cx="44" cy="26" r="4"/><circle cx="72" cy="42" r="3"/>
<circle cx="100" cy="18" r="4.5"/></g><circle cx="88" cy="66" r="3" fill="none" stroke="#9d8ff8"
stroke-opacity="1" stroke-dasharray="2 2"/></svg>""",
        "vault": """<svg viewBox="0 0 120 80" width="120" height="80" aria-hidden="true">
<rect x="30" y="20" width="60" height="44" rx="8" fill="none" stroke="#4a4d56"/>
<path d="M30 30 h22 l6 -8 h32" fill="none" stroke="#4a4d56"/>
<g fill="#7f838d"><circle cx="48" cy="46" r="2.5"/><circle cx="62" cy="40" r="3"/><circle cx="74" cy="50" r="2.5"/></g>
<path d="M48 46 L62 40 L74 50" stroke="#9d8ff8" fill="none"/></svg>""",
        "archive": """<svg viewBox="0 0 120 80" width="120" height="80" aria-hidden="true">
<rect x="32" y="18" width="56" height="14" rx="4" fill="none" stroke="#4a4d56"/>
<path d="M36 32 v26 a5 5 0 0 0 5 5 h38 a5 5 0 0 0 5 -5 v-26" fill="none" stroke="#4a4d56"/>
<path d="M52 42 h16" stroke="#9d8ff8" stroke-linecap="round"/>
<circle cx="96" cy="16" r="2" fill="#7f838d"/><circle cx="22" cy="60" r="1.6" fill="#7f838d"/></svg>""",
    }
    return (f'<div class="cx-empty">{svg_img(arts.get(art, arts["stars"]), 120, 80)}<div class="cx-empty-title">{escape(title)}</div>'
            f'<div class="cx-empty-body">{escape(body)}</div></div>')


# --------------------------------------------------------------------------- #
# Hero + metrics
# --------------------------------------------------------------------------- #
def hero(chips_html: str, tagline: str) -> str:
    return (f'<header class="cx-hero"><div class="cx-hero-brand">{LOGO_SVG}<div>'
            f'<h1 class="cx-hero-title">Constellation</h1><p class="cx-hero-tag">{escape(tagline)}</p>'
            f'</div></div><div class="cx-hero-chips">{chips_html}</div></header>')


def metric_cards(items: Sequence[Tuple], cols: int = 4, compact: bool = False) -> str:
    """items: (label, value[, hint[, accent]]) tuples -> a responsive card grid.

    accent is one of '', 'violet', 'cyan', 'red'.
    """
    cells = []
    for item in items:
        label, value = item[0], item[1]
        hint = item[2] if len(item) > 2 else ""
        accent = item[3] if len(item) > 3 else ""
        cls = f" cx-metric--{escape(accent)}" if accent else ""
        hint_html = f'<div class="cx-metric-hint">{escape(str(hint))}</div>' if hint else ""
        cells.append(f'<div class="cx-metric{cls}"><div class="cx-metric-label">{escape(str(label))}</div>'
                     f'<div class="cx-metric-value">{escape(str(value))}</div>{hint_html}</div>')
    cls = "cx-metrics cx-metrics--compact" if compact else "cx-metrics"
    return f'<div class="{cls}" style="--cx-cols:{int(cols)}">{"".join(cells)}</div>'


# --------------------------------------------------------------------------- #
# Stage stepper + log
# --------------------------------------------------------------------------- #
def stepper(current: Optional[str], finished: bool = False, failed: bool = False) -> str:
    """The 7-stage pipeline tracker: one step per stage with a data-state attribute."""
    states = stage_states(current, finished, failed)
    marks = {"done": icon("check", 13, "cx-icon cx-step-mark"), "failed": icon("x", 13, "cx-icon cx-step-mark")}
    steps = []
    for i, (name, state) in enumerate(zip(STAGES, states), start=1):
        mark = marks.get(state, f'<span class="cx-step-num">{i}</span>')
        steps.append(f'<li class="cx-step" data-state="{state}"><span class="cx-step-dot">{mark}</span>'
                     f'<span class="cx-step-label">{escape(name)}</span></li>')
    return f'<ol class="cx-stepper">{"".join(steps)}</ol>'


def log_panel(lines: Iterable[str], height: int = 320, empty: str = "Waiting for output...") -> str:
    """Monospace log panel that keeps the newest line in view."""
    text = "\n".join(str(x) for x in lines)
    body = escape(text) if text.strip() else f'<span class="cx-log-empty">{escape(empty)}</span>'
    return f'<div class="cx-log" style="max-height:{int(height)}px"><pre>{body}</pre></div>'


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def diff_html(diff: str) -> str:
    """Renders a unified diff with line numbers and green/red line backgrounds."""
    if not (diff or "").strip():
        return '<div class="cx-diff cx-diff--empty">No changes for this note.</div>'
    rows: List[str] = []
    old = new = 0
    adds = dels = 0
    for raw in diff.splitlines():
        if raw.startswith(("--- ", "+++ ")):
            continue
        m = _HUNK_RE.match(raw)
        if m:
            old, new = int(m.group(1)), int(m.group(2))
            rows.append(f'<tr class="cx-d-hunk"><td></td><td></td><td>{escape(raw)}</td></tr>')
            continue
        if raw.startswith("+"):
            cls, a, b, new = "add", "", str(new), new + 1
            adds += 1
        elif raw.startswith("-"):
            cls, a, b, old = "del", str(old), "", old + 1
            dels += 1
        elif raw.startswith("\\"):
            rows.append(f'<tr class="cx-d-meta"><td></td><td></td><td>{escape(raw)}</td></tr>')
            continue
        else:
            cls, a, b = "ctx", str(old), str(new)
            old, new = old + 1, new + 1
        sign, text = (raw[:1] or " "), raw[1:]
        rows.append(f'<tr class="cx-d-{cls}"><td class="cx-d-n">{a}</td><td class="cx-d-n">{b}</td>'
                    f'<td><span class="cx-d-sign">{escape(sign)}</span>{escape(text)}</td></tr>')
    head = (f'<div class="cx-diff-head"><span class="cx-d-stat cx-d-stat--add">+{adds}</span>'
            f'<span class="cx-d-stat cx-d-stat--del">&minus;{dels}</span></div>')
    return f'<div class="cx-diff">{head}<div class="cx-diff-scroll"><table>{"".join(rows)}</table></div></div>'


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
VIS_NETWORK_URL = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def safe_json(data) -> str:
    """JSON that is safe inside a <script> element (no '</script>' or '<!--' breakouts)."""
    return (json.dumps(data, ensure_ascii=False)
            .replace("</", "<\\/").replace("<!--", "<\\!--")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def graph_html(data: dict, height: int = 620) -> str:
    """Self-contained page (for components.html) with a vis-network star map."""
    template = (UI_DIR / "graph.html").read_text(encoding="utf-8")
    return (template.replace("__VIS_URL__", VIS_NETWORK_URL)
            .replace("__HEIGHT__", str(int(height)))
            .replace("/*__DATA__*/null", safe_json(data)))
