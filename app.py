"""Constellation -- local Streamlit UI.

Run with:  streamlit run app.py
"""

import csv
import io
import json
import time
import traceback
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Constellation", page_icon="✦", layout="wide")

from constellation import demo
from constellation.core import config, vectorstore
from constellation.core import log as logmod
from constellation.core.backup import list_backups, restore_backup
from constellation.core.lock import VaultLockedError, vault_lock
from constellation.core.utils import clean_path
from constellation.nodes.io import CACHE_FILENAME, count_notes, reset_vault_state, strip_vault
from constellation.ui import components as ui
from constellation.ui import helpers

TAGLINE = "Links your Obsidian notes with LLM-extracted concepts, tags and wikilinks."


def init_state() -> None:
    """Seeds session_state once so the vault path and options survive reruns.

    Options live in a plain dict rather than as widget keys: Streamlit discards
    a widget's keyed state on any run where the widget isn't drawn (the options
    panel is hidden until the vault path is valid), which reset them silently.
    """
    st.session_state.setdefault("vault_path", "")
    st.session_state.setdefault("opts", {
        "dry_run": True,
        "force": False,
        "max_links": max(1, min(20, config.MAX_LINKS_PER_NOTE or 5)),
        "annotate": config.ANNOTATE_LINKS,
        "excluded": ", ".join(config.EXCLUDED_DIRS),
        "backlinks": config.BACKLINKS,
        "tags_fm": config.TAGS_IN_FRONTMATTER,
    })


def _option(widget, label: str, name: str, *args, **kwargs) -> None:
    opts = st.session_state["opts"]
    opts[name] = widget(label, *args, value=opts[name], key=f"w_{name}", **kwargs)


def parse_keys(text: str):
    """Accepts keys separated by newlines or commas."""
    return [k.strip() for chunk in text.splitlines() for k in chunk.split(",") if k.strip()]


def vault_dir() -> str:
    return clean_path(st.session_state.get("vault_path", ""))


def is_demo_vault(path: str) -> bool:
    return bool(path) and path == st.session_state.get("demo_vault")


def start_demo() -> None:
    """Button callback: copy the bundled demo vault to a temp folder, point the
    app at it with preview-friendly options and queue a run. Runs as a callback
    so it can set the vault path before the text input is drawn."""
    try:
        path = str(demo.copy_demo_vault())
    except (OSError, ValueError) as e:
        st.session_state["demo_error"] = str(e)
        return
    st.session_state.update(vault_path=path, demo_vault=path, demo_autorun=True)
    st.session_state["opts"].update(dry_run=True, force=False, excluded=", ".join(demo.DEMO_EXCLUDED_DIRS))
    # Applied now (not only at run time) so the vault card already counts 15 notes.
    config.set_excluded_dirs(demo.DEMO_EXCLUDED_DIRS)
    # Drop the widgets' own state so they redraw from the updated options.
    for name in ("dry_run", "force", "excluded"):
        st.session_state.pop(f"w_{name}", None)


def inject_theme() -> None:
    css = ui.load_css()
    if css:
        st.html(f"<style>{css}</style>")


# --------------------------------------------------------------------------- #
# Status chips
# --------------------------------------------------------------------------- #
def provider_chips() -> list:
    """Chips for the active key/provider setup (read after the sidebar applied it)."""
    llm, emb = config.LLM_PROVIDER, config.EMBEDDING_PROVIDER
    chips = []
    for name, keys, needed in (("Groq", config.GROQ_API_KEYS, llm == "groq"),
                               ("Google", config.GOOGLE_API_KEYS, emb == "google")):
        if not needed:
            chips.append(ui.chip(f"{name} · off", "muted", f"{name} isn't used by the current providers"))
        elif keys:
            chips.append(ui.chip(f"{name} · {len(keys)} key{'s' if len(keys) != 1 else ''}", "ok"))
        else:
            chips.append(ui.chip(f"{name} · no key", "err", f"Add a {name} key in the sidebar or .env"))
    if "ollama" in (llm, emb):
        chips.append(ui.chip("Ollama · local", "accent", config.OLLAMA_BASE_URL))
    else:
        chips.append(ui.chip("Ollama · off", "muted"))
    return chips


def vault_chip(info) -> str:
    path = vault_dir()
    if not path:
        return ui.chip("No vault", "muted")
    if info is None:
        return ui.chip("Vault not found", "err", path)
    if not info["notes"]:
        return ui.chip("Vault · no notes", "warn", path)
    return ui.chip(f"Vault · {info['notes']} notes", "cyan", path)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.html(f'<div class="cx-side-brand">{ui.LOGO_SVG}<div><b>Constellation</b>'
                f'<small>Obsidian knowledge graph</small></div></div>')

        st.html(ui.section_label("Providers", "cpu"))
        llms, embs = ["groq", "ollama"], ["google", "ollama"]
        llm_provider = config.LLM_PROVIDER
        emb_provider = config.EMBEDDING_PROVIDER
        llm_provider = st.selectbox(
            "LLM provider", llms, index=llms.index(llm_provider) if llm_provider in llms else 0)
        emb_provider = st.selectbox(
            "Embedding provider", embs, index=embs.index(emb_provider) if emb_provider in embs else 0)
        if "ollama" in (llm_provider, emb_provider):
            st.caption(f"Ollama at `{config.OLLAMA_BASE_URL}` runs fully on this machine.")

        st.html(ui.section_label("Status", "shield"))
        chips_slot = st.empty()

        st.html(ui.section_label("Configuration", "key"))
        with st.expander("API keys"):
            st.caption("Separate several keys with commas; extras rotate on rate limits.")
            groq_input = st.text_input("Groq keys", type="password", key="groq_keys")
            google_input = st.text_input("Google keys", type="password", key="google_keys")

        with st.expander("Models"):
            ext = st.text_input("Extraction", value=config.LLM_EXTRACTION)
            ver = st.text_input("Verification", value=config.LLM_VERIFICATION)
            rel = st.text_input("Relationship", value=config.LLM_RELATIONSHIP)
            st.caption(f"Embeddings: `{config.EMBEDDING_MODEL}`")

    if parse_keys(groq_input):
        config.set_groq_keys(parse_keys(groq_input) + config.GROQ_API_KEYS)
    if parse_keys(google_input):
        config.set_google_keys(parse_keys(google_input) + config.GOOGLE_API_KEYS)
    config.set_models(ext, ver, rel)
    config.set_providers(llm=llm_provider, embeddings=emb_provider)

    chips_slot.html(f'<div class="cx-side-chips">{"".join(provider_chips())}</div>')


# --------------------------------------------------------------------------- #
# Run tab
# --------------------------------------------------------------------------- #
def vault_info(path: str) -> dict:
    vault = Path(path)
    cache = vault / CACHE_FILENAME
    last_run = None
    if cache.exists():
        last_run = datetime.fromtimestamp(cache.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {
        "has_obsidian": (vault / ".obsidian").is_dir(),
        "notes": count_notes(vault),
        "excluded": helpers.count_excluded_dirs(str(vault), config.EXCLUDED_DIRS),
        "cache": cache.exists(),
        "index": vectorstore.index_exists(str(vault)),
        "last_run": last_run,
    }


def current_vault_info():
    """vault_info for the current path, or None when it isn't a directory."""
    path = vault_dir()
    return vault_info(path) if path and Path(path).is_dir() else None


def render_vault_panel(info) -> bool:
    """Path input + validation card. Returns True when the vault can be run."""
    with st.container(key="card_vault"):
        st.html(ui.card_header("Vault", "The folder your Obsidian notes live in.", "folder",
                               right=vault_chip(info)))
        st.text_input(
            "Obsidian vault folder",
            key="vault_path",
            placeholder="/Users/you/Documents/MyVault",
            help="Absolute path to the folder your notes live in.",
            label_visibility="collapsed",
        )
        st.button("Try the demo vault", key="demo_btn", on_click=start_demo, icon=":material/science:",
                  help="Copies a 15-note example vault to a temp folder and previews a run with canned "
                       "model answers. No API keys needed.")
        if st.session_state.get("demo_error"):
            st.html(ui.callout(f"Couldn't copy the demo vault: {st.session_state.pop('demo_error')}", "err"))
        path = vault_dir()
        if not path:
            return False
        if is_demo_vault(path):
            st.html(ui.callout("Demo vault: a temporary copy of `examples/demo-vault`. Runs on it use canned "
                               "model answers, so no API keys or network are needed.", "info", "shield"))
        if info is None:
            st.html(ui.callout("That path is not a directory. Paste the full path to your vault.", "err"))
            return False
        if not info["has_obsidian"]:
            st.html(ui.callout("No `.obsidian` folder found, so this may not be an Obsidian vault.", "warn"))
        if info["notes"] == 0:
            st.html(ui.callout("No `.md` notes found in that folder.", "warn"))
        st.html(ui.metric_cards([
            ("Notes", info["notes"], "", "violet"),
            ("Excluded", info["excluded"]),
            ("Cache", "present" if info["cache"] else "none"),
            ("Concept index", "built" if info["index"] else "none"),
            ("Last run", info["last_run"] or "never"),
        ], cols=5, compact=True))
    return info["notes"] > 0


def render_options() -> None:
    with st.container(key="card_options"):
        st.html(ui.card_header("Options", "Saved for this session.", "sliders"))
        left, right = st.columns(2, gap="medium")
        with left:
            st.html(ui.section_label("Safety"))
            _option(st.checkbox, "Preview only (dry run, don't modify notes)", "dry_run")
            _option(st.checkbox, "Force: clear cache and concept index first", "force")
            st.html(ui.section_label("Linking"))
            _option(st.slider, "Max links per note", "max_links", 1, 20)
            _option(st.checkbox, "Annotate links with reasoning", "annotate")
        with right:
            st.html(ui.section_label("Output"))
            _option(st.checkbox, "Write backlinks", "backlinks",
                    help="Every link A → B also adds B → A.")
            _option(st.checkbox, "Tags in frontmatter", "tags_fm",
                    help="Write tags to the YAML `tags:` list instead of a `## Tags` section.")
            st.html(ui.section_label("Scope"))
            _option(st.text_input, "Excluded folders (comma-separated)", "excluded",
                    help="e.g. Templates, Archive, Daily Notes")


def render_launch(can_run: bool) -> bool:
    """The run card. Returns True when the run button was clicked."""
    opts = st.session_state["opts"]
    with st.container(key="card_launch"):
        st.html(ui.card_header("Launch", "Seven stages, from reading to report.", "play"))
        if opts["dry_run"]:
            mode = ('<div class="cx-mode cx-mode--preview"><span class="cx-card-icon">'
                    f'{ui.icon("shield", 15)}</span><div><b>Preview mode</b>'
                    '<span>Nothing is written. Review diffs in Results.</span></div></div>')
        else:
            mode = ('<div class="cx-mode cx-mode--write"><span class="cx-card-icon">'
                    f'{ui.icon("alert", 15)}</span><div><b>Write mode</b>'
                    '<span>Notes are edited. A backup is taken first.</span></div></div>')
        bullets = [
            f"Up to {opts['max_links']} links per note",
            "Backlinks on" if opts["backlinks"] else "Backlinks off",
            "Tags in frontmatter" if opts["tags_fm"] else "Tags in a ## Tags section",
        ]
        if opts["force"]:
            bullets.append("Cache and index cleared first")
        items = "".join(f"<li>{ui.icon('check', 13)}{b}</li>" for b in bullets)
        st.html(f'{mode}<ul class="cx-checklist">{items}</ul>')
        label = "Run preview" if opts["dry_run"] else "Run Constellation"
        clicked = st.button(label, type="primary", disabled=not can_run, width="stretch",
                            key="run_btn", icon=":material/play_arrow:")
        st.caption("Only Constellation's own Tags and Related Links sections are rewritten. "
                   "Backups are listed under History & Undo.")
    return clicked


def show_error(e: Exception) -> None:
    if isinstance(e, VaultLockedError):
        st.html(ui.callout("Another run is in progress on this vault. Wait for it to finish and try again.", "err"))
    else:
        st.html(ui.callout(f"Something went wrong: {e}", "err"))
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exception(e)), language="text")


def execute_run(path: str) -> None:
    """Runs the pipeline with live stage tracking and stores the result."""
    from main import run_pipeline

    offline = is_demo_vault(path)
    missing = [] if offline else config.missing_keys()
    if missing:
        st.html(ui.callout(f"Missing API key(s): {', '.join(missing)}. Add them in the sidebar or `.env`.", "err"))
        return

    apply_options()
    dry_run = st.session_state["opts"]["dry_run"]
    lines: list = []
    current: dict = {"stage": None}
    started = time.time()

    with st.status("Starting…", expanded=True) as status:
        tracker = st.empty()
        tracker.html(ui.stepper(None))
        with st.expander("Raw log", expanded=False):
            log_box = st.empty()
            log_box.html(ui.log_panel([]))

        def sink(message: str):
            lines.append(message)
            stage = helpers.detect_stage(message)
            if stage:
                current["stage"] = stage
                status.update(label=f"Running · {stage}")
                tracker.html(ui.stepper(stage))
            log_box.html(ui.log_panel(lines[-400:]))

        logmod.set_sink(sink)
        try:
            if st.session_state["opts"]["force"]:
                st.toast(f"Reset: {reset_vault_state(path)}")
            if offline:
                # A short per-call delay lets the stage tracker visibly advance.
                with demo.offline_backends(latency=0.03):
                    result = run_pipeline(path, dry_run=dry_run)
            else:
                result = run_pipeline(path, dry_run=dry_run)
            st.session_state.update(
                result=result, result_dry_run=dry_run,
                result_duration=time.time() - started, log=list(lines))
            tracker.html(ui.stepper(current["stage"], finished=True))
            status.update(label=f"Finished in {time.time() - started:.1f}s", state="complete", expanded=False)
        except Exception as e:
            st.session_state["log"] = list(lines)
            tracker.html(ui.stepper(current["stage"], failed=True))
            status.update(label="Failed", state="error")
            show_error(e)
            return
        finally:
            logmod.clear_sink()
    st.html(ui.callout("Done. See the Results, Graph and History tabs.", "ok"))


def apply_options() -> None:
    o = st.session_state["opts"]
    config.set_max_links(o["max_links"])
    config.set_annotate_links(o["annotate"])
    config.set_excluded_dirs(o["excluded"])
    config.set_backlinks(bool(o["backlinks"]))
    config.set_tags_in_frontmatter(bool(o["tags_fm"]))


def render_run_tab(info) -> None:
    can_run = render_vault_panel(info)
    if info is None:
        if not vault_dir():
            st.html(ui.empty_state("Connect your vault",
                                   "Paste the path to your Obsidian vault above, or try the demo vault to see a run without API keys.",
                                   "vault"))
        return
    with st.container(key="run_layout"):
        left, right = st.columns([1.7, 1], gap="medium")
        with left:
            render_options()
        with right:
            clicked = render_launch(can_run)
    if st.session_state.pop("demo_autorun", False) or clicked:
        execute_run(vault_dir())
    elif st.session_state.get("log"):
        with st.expander("Last run log"):
            st.html(ui.log_panel(st.session_state["log"][-500:], height=360))


# --------------------------------------------------------------------------- #
# Results tab
# --------------------------------------------------------------------------- #
def to_csv(rows: list) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


TABLE_COLUMNS = {
    "links": {
        "From": st.column_config.TextColumn("From", width=140),
        "To": st.column_config.TextColumn("To", width=140),
        "Relationship": st.column_config.TextColumn("Relationship", width=115),
        "Evidence": st.column_config.TextColumn("Evidence", width=340, help="Why the link was made"),
        "Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f",
                                                 width=110),
    },
    "tags": {
        "Note": st.column_config.TextColumn("Note", width=220),
        "Tags": st.column_config.TextColumn("Tags", width="large"),
    },
    "concepts": {
        "Note": st.column_config.TextColumn("Note", width=140),
        "Concept": st.column_config.TextColumn("Concept", width=160),
        "Category": st.column_config.TextColumn("Category", width=100),
        "Importance": st.column_config.ProgressColumn("Importance", min_value=0, max_value=10, format="%d",
                                                      width=110),
        "Explanation": st.column_config.TextColumn("Explanation", width=320),
    },
}


def table_with_downloads(rows: list, name: str) -> None:
    if not rows:
        st.html(ui.empty_state(f"No {name} yet", f"This run didn't produce any {name}."))
        return
    c_search, c_csv, c_json = st.columns([5, 1, 1], vertical_alignment="bottom")
    query = c_search.text_input("Search", key=f"search_{name}", placeholder=f"Filter {name}…",
                                label_visibility="collapsed")
    shown = helpers.filter_rows(rows, query)
    c_csv.download_button("CSV", to_csv(shown), f"{name}.csv", "text/csv", key=f"{name}_csv",
                          icon=":material/download:", width="stretch")
    c_json.download_button("JSON", json.dumps(shown, indent=2, default=str), f"{name}.json",
                           "application/json", key=f"{name}_json", icon=":material/download:", width="stretch")
    st.html(f'<div class="cx-rowcount"><b>{len(shown)}</b> of {len(rows)} rows</div>')
    config = dict(TABLE_COLUMNS.get(name) or {})
    if name == "links":
        # Scores are similarity plus a bonus per extra supporting concept pair,
        # so they can exceed 1 -- scale the bar to the data, not to 0..1.
        top = max((float(r.get("Score") or 0) for r in rows), default=1.0)
        config["Score"] = st.column_config.ProgressColumn("Score", min_value=0.0, max_value=max(1.0, top),
                                                          format="%.2f", width=110)
    st.dataframe(shown, width="stretch", hide_index=True, column_config=config,
                 height=min(38 + 35 * max(len(shown), 1), 520))


def render_metrics(result: dict) -> None:
    rep = result.get("run_report") or {}
    failed = rep.get("notes_failed", result.get("failed_notes")) or []
    tokens = (rep.get("input_tokens") or 0) + (rep.get("output_tokens") or 0)
    duration = rep.get("duration_s", st.session_state.get("result_duration"))
    tags_total = sum(len(t) for t in result.get("tags_by_note", {}).values())
    cards = [
        ("Notes scanned", rep.get("notes_total", len(result.get("notes", [])))),
        ("Processed", rep.get("notes_processed", len(result.get("new_notes", [])))),
        ("Written", rep.get("notes_written", result.get("notes_written", 0))),
        ("New links", rep.get("links_new", len(result.get("new_links", []))), "", "violet"),
        ("Tags", tags_total, "", "cyan"),
        ("Failed notes", len(failed), "", "red" if failed else ""),
        ("Tokens used", f"{tokens:,}" if tokens else "n/a"),
        ("Duration", f"{duration:.1f}s" if duration else "n/a"),
    ]
    st.html(ui.metric_cards(cards, cols=4))


def render_previews(preview: dict) -> None:
    names = sorted(preview)
    changed = [n for n in names if preview[n].get("before") != preview[n].get("after")]
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    pick = c1.selectbox("Note", changed or names, key="preview_pick")
    c2.html(f'<div class="cx-rowcount"><b>{len(changed)}</b> of {len(names)} notes would change</div>')
    entry = preview.get(pick) or {}
    diff = helpers.note_diff(entry.get("before", ""), entry.get("after", ""), pick or "note")
    st.html(ui.diff_html(diff))


def render_results_tab() -> None:
    result = st.session_state.get("result")
    if not result:
        st.html(ui.empty_state("No results yet",
                               "Run the pipeline from the Run tab. Links, tags and diffs show up here."))
        return
    if st.session_state.get("result_dry_run"):
        st.html(ui.callout("Preview run: no files were modified. Review the Diffs tab, then run again "
                           "with Preview off to write.", "info", "shield"))
    else:
        st.html(ui.callout(f"Updated {result.get('notes_written', 0)} note(s).", "ok"))
    render_metrics(result)

    rep = result.get("run_report") or {}
    failed = rep.get("notes_failed") or result.get("failed_notes") or []
    if failed:
        with st.expander(f"{len(failed)} failed note(s)", icon=":material/error:"):
            st.write(failed)
    skipped = rep.get("skipped_due_to_edit") or []
    if skipped:
        with st.expander(f"{len(skipped)} note(s) skipped because they were edited during the run",
                         icon=":material/edit_note:"):
            st.write(skipped)

    t_links, t_tags, t_concepts, t_diffs, t_log = st.tabs(["Links", "Tags", "Concepts", "Diffs", "Log"])
    with t_links:
        table_with_downloads(helpers.link_rows(result.get("links", [])), "links")
    with t_tags:
        table_with_downloads(
            [{"Note": n, "Tags": " ".join(t)} for n, t in sorted(result.get("tags_by_note", {}).items())], "tags")
    with t_concepts:
        table_with_downloads(
            [{"Note": c.get("source_note"), "Concept": c.get("concept_name"), "Category": c.get("category"),
              "Importance": c.get("importance_score"), "Explanation": c.get("explanation")}
             for c in result.get("concepts", [])], "concepts")
    with t_diffs:
        if result.get("preview"):
            render_previews(result["preview"])
        else:
            st.html(ui.empty_state("No diffs for this run",
                                   "Per-note diffs appear here after a preview run, when the pipeline provides them."))
    with t_log:
        log = st.session_state.get("log", [])
        st.html(ui.log_panel(log, height=520, empty="No log captured for this run."))
        if log:
            st.download_button("Download log", "\n".join(log), "constellation.log", "text/plain",
                               key="log_dl", icon=":material/download:")


# --------------------------------------------------------------------------- #
# Graph tab
# --------------------------------------------------------------------------- #
def render_graph_tab() -> None:
    result = st.session_state.get("result")
    links = (result or {}).get("links", [])
    if not links:
        st.html(ui.empty_state("No constellation to draw yet",
                               "Run the pipeline first; every link between notes becomes a line between stars."))
        return
    c1, c2 = st.columns([2, 1], gap="medium", vertical_alignment="bottom")
    query = c1.text_input("Search note", key="graph_q", placeholder="Show a note and its neighbours")
    top_n = c2.slider("Most-connected notes", 5, 200, 40, key="graph_n")
    data = helpers.graph_data(links, query, top_n)
    n_nodes, n_edges = len(data["nodes"]), len(data["edges"])
    if n_nodes == 0:
        st.html(ui.empty_state("Nothing matches that search", "Try part of a note title, or clear the search."))
        return
    chips = [ui.chip(f"{n_nodes} notes", "accent"), ui.chip(f"{n_edges} links", "cyan")]
    if query.strip():
        matches = sum(1 for n in data["nodes"] if n["match"])
        chips.append(ui.chip(f"{matches} match{'es' if matches != 1 else ''}", "ok"))
    chips.append(ui.chip("Click a star to trace its links · drag to rearrange", "muted"))
    st.html(f'<div class="cx-inline-chips">{"".join(chips)}</div>')
    components.html(ui.graph_html(data, height=620), height=630)


# --------------------------------------------------------------------------- #
# History & Undo tab
# --------------------------------------------------------------------------- #
def render_backups(path: str) -> None:
    try:
        backups = list_backups(path)
    except Exception as e:
        show_error(e)
        return
    count = ui.chip(f"{len(backups)} backup{'s' if len(backups) != 1 else ''}", "accent" if backups else "muted")
    st.html(ui.card_header("Backups", "Taken automatically before any note is modified.", "history", right=count))
    if not backups:
        st.html(ui.empty_state("No backups yet",
                               "One is created before Constellation modifies your notes.", "archive"))
        return
    for i, b in enumerate(backups):
        when, age = helpers.format_timestamp(b.get("created"))
        with st.container(key=f"card_backup_{i}"):
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            meta = " · ".join(x for x in (when, age, f"{b.get('file_count', '?')} files") if x)
            c1.html(f'<div class="cx-backup"><span class="cx-card-icon">{ui.icon("archive", 15)}</span>'
                    f'<div><div class="cx-backup-name">{escape(str(b.get("name")))}</div>'
                    f'<div class="cx-backup-meta">{escape(meta)}</div></div></div>')
            if c2.button("Restore", key=f"restore_{b.get('name')}", icon=":material/settings_backup_restore:",
                         width="stretch"):
                try:
                    with vault_lock(path):
                        restored = restore_backup(path, b["name"])
                    st.html(ui.callout(f"Restored {restored} file(s) from {b['name']}.", "ok"))
                except Exception as e:
                    show_error(e)


def render_strip(path: str) -> None:
    with st.container(key="card_danger"):
        st.html(ui.card_header(
            "Danger zone", "Remove everything Constellation wrote.", "trash",
            right=ui.chip("Edits notes", "err")))
        st.caption("Strips the `## Tags` / `## Related Links` sections and frontmatter tags Constellation added, "
                   "and deletes its cache and index. A backup is taken first, so this can be restored above.")
        confirm = st.checkbox("I understand this edits my notes", key="confirm_strip")
        if not st.button("Strip Constellation output", disabled=not confirm, key="strip_btn",
                         icon=":material/delete_sweep:"):
            return
        try:
            res = strip_vault(path, dry_run=False)
            msg = f"Cleaned {res.get('notes_cleaned', 0)} note(s)."
            if res.get("backup_dir"):
                msg += f" Backup: `{res['backup_dir']}`"
            st.html(ui.callout(msg, "ok"))
        except Exception as e:
            show_error(e)


def render_history_tab() -> None:
    path = vault_dir()
    if not path or not Path(path).is_dir():
        st.html(ui.empty_state("No vault selected", "Set a valid vault path in the Run tab to see its backups.",
                               "vault"))
        return
    with st.container(key="card_backups"):
        render_backups(path)
    render_strip(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    inject_theme()
    render_sidebar()
    info = current_vault_info()
    chips = provider_chips() + [vault_chip(info)]
    if is_demo_vault(vault_dir()):
        chips.append(ui.chip("Demo · offline", "accent", "Canned model answers; no API calls"))
    st.html(ui.hero("".join(chips), TAGLINE))
    t_run, t_results, t_graph, t_hist = st.tabs(["Run", "Results", "Graph", "History & Undo"])
    with t_run:
        render_run_tab(info)
    with t_results:
        render_results_tab()
    with t_graph:
        render_graph_tab()
    with t_hist:
        render_history_tab()


main()
