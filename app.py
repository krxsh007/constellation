"""Constellation -- local Streamlit UI.

Run with:  streamlit run app.py
"""

import csv
import io
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Constellation", page_icon="🌌", layout="wide")

from constellation.core import config, log as logmod, vectorstore
from constellation.core.backup import list_backups, restore_backup
from constellation.core.lock import VaultLockedError
from constellation.core.utils import clean_path
from constellation.nodes.io import CACHE_FILENAME, count_notes, reset_vault_state, strip_vault
from constellation.ui import helpers


def init_state() -> None:
    """Seeds session_state once so the vault path and options survive reruns."""
    defaults = {
        "vault_path": "",
        "opt_dry_run": True,
        "opt_force": False,
        "opt_max_links": max(1, min(20, config.MAX_LINKS_PER_NOTE or 5)),
        "opt_annotate": config.ANNOTATE_LINKS,
        "opt_excluded": ", ".join(config.EXCLUDED_DIRS),
        "opt_backlinks": config.BACKLINKS,
        "opt_tags_fm": config.TAGS_IN_FRONTMATTER,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def parse_keys(text: str):
    """Accepts keys separated by newlines or commas."""
    return [k.strip() for chunk in text.splitlines() for k in chunk.split(",") if k.strip()]


def vault_dir() -> str:
    return clean_path(st.session_state.get("vault_path", ""))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _pill(name: str, n: int, needed: bool) -> str:
    if not needed:
        color, text = "#888", f"{name}: not needed"
    elif n:
        color, text = "#1a9d5b", f"{name}: {n} key(s) loaded"
    else:
        color, text = "#d6453d", f"{name}: missing"
    return (f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:999px;font-size:0.8rem">{text}</span>')


def render_sidebar() -> None:
    with st.sidebar:
        st.header("🌌 Constellation")

        llm_provider = config.LLM_PROVIDER
        emb_provider = config.EMBEDDING_PROVIDER
        with st.expander("Providers"):
            llms, embs = ["groq", "ollama"], ["google", "ollama"]
            llm_provider = st.selectbox(
                "LLM provider", llms, index=llms.index(llm_provider) if llm_provider in llms else 0)
            emb_provider = st.selectbox(
                "Embedding provider", embs, index=embs.index(emb_provider) if emb_provider in embs else 0)
            if "ollama" in (llm_provider, emb_provider):
                st.caption(f"Ollama at `{config.OLLAMA_BASE_URL}` — runs fully on this machine.")

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

    with st.sidebar:
        st.markdown(_pill("Groq", len(config.GROQ_API_KEYS), llm_provider == "groq"), unsafe_allow_html=True)
        st.markdown(_pill("Google", len(config.GOOGLE_API_KEYS), emb_provider == "google"),
                    unsafe_allow_html=True)


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


def render_vault_panel() -> bool:
    """Path input + validation. Returns True when the vault can be run."""
    st.text_input(
        "Obsidian vault folder",
        key="vault_path",
        placeholder="/Users/you/Documents/MyVault",
        help="Absolute path to the folder your notes live in.",
    )
    path = vault_dir()
    if not path:
        st.info("Enter the path to your vault to get started.")
        return False
    if not Path(path).is_dir():
        st.error("That path is not a directory. Paste the full path to your vault.")
        return False
    info = vault_info(path)
    if not info["has_obsidian"]:
        st.warning("No `.obsidian` folder found — this may not be an Obsidian vault.")
    if info["notes"] == 0:
        st.warning("No `.md` notes found in that folder.")
    cols = st.columns(5)
    cols[0].metric("Notes", info["notes"])
    cols[1].metric("Excluded folders", info["excluded"])
    cols[2].metric("Cache", "present" if info["cache"] else "none")
    cols[3].metric("Concept index", "built" if info["index"] else "not built")
    cols[4].metric("Last run", info["last_run"] or "never")
    return info["notes"] > 0


def render_options() -> None:
    with st.container(border=True):
        st.subheader("Options")
        left, right = st.columns(2)
        with left:
            st.markdown("**Safety**")
            st.checkbox("Preview only (dry run — don't modify notes)", key="opt_dry_run")
            st.checkbox("Force: clear cache and concept index first", key="opt_force")
            st.markdown("**Linking**")
            st.slider("Max links per note", 1, 20, key="opt_max_links")
            st.checkbox("Annotate links with reasoning", key="opt_annotate")
        with right:
            st.markdown("**Output**")
            st.checkbox("Write backlinks", key="opt_backlinks",
                        help="Every link A → B also adds B → A.")
            st.checkbox("Tags in frontmatter", key="opt_tags_fm",
                        help="Write tags to the YAML `tags:` list instead of a `## Tags` section.")
            st.markdown("**Scope**")
            st.text_input("Excluded folders (comma-separated)", key="opt_excluded",
                          help="e.g. Templates, Archive, Daily Notes")


def apply_options() -> None:
    s = st.session_state
    config.set_max_links(s["opt_max_links"])
    config.set_annotate_links(s["opt_annotate"])
    config.set_excluded_dirs(s["opt_excluded"])
    config.set_backlinks(bool(s["opt_backlinks"]))
    config.set_tags_in_frontmatter(bool(s["opt_tags_fm"]))


def show_error(e: Exception) -> None:
    if isinstance(e, VaultLockedError):
        st.error("Another run is in progress on this vault. Wait for it to finish and try again.")
    else:
        st.error(f"Something went wrong: {e}")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exception(e)), language="text")


def execute_run(path: str) -> None:
    """Runs the pipeline with live stage tracking and stores the result."""
    from main import run_pipeline

    missing = config.missing_keys()
    if missing:
        st.error(f"Missing API key(s): {', '.join(missing)}. Add them in the sidebar or `.env`.")
        return

    apply_options()
    dry_run = st.session_state["opt_dry_run"]
    lines: list = []
    current = {"stage": None}
    started = time.time()

    with st.status("Starting…", expanded=True) as status:
        tracker = st.empty()
        tracker.markdown(helpers.stage_markdown(None))
        with st.expander("Raw log"):
            log_box = st.empty()

        def sink(message: str):
            lines.append(message)
            stage = helpers.detect_stage(message)
            if stage:
                current["stage"] = stage
                status.update(label=f"Running: {stage}…")
                tracker.markdown(helpers.stage_markdown(stage))
            log_box.code("\n".join(lines[-400:]), language="text")

        logmod.set_sink(sink)
        try:
            if st.session_state["opt_force"]:
                st.toast(f"Reset: {reset_vault_state(path)}")
            result = run_pipeline(path, dry_run=dry_run)
            st.session_state.update(
                result=result, result_dry_run=dry_run,
                result_duration=time.time() - started, log=list(lines))
            tracker.markdown(helpers.stage_markdown(current["stage"], finished=True))
            status.update(label="Finished", state="complete", expanded=False)
        except Exception as e:
            st.session_state["log"] = list(lines)
            tracker.markdown(helpers.stage_markdown(current["stage"], failed=True))
            status.update(label="Failed", state="error")
            show_error(e)
            return
        finally:
            logmod.clear_sink()
    st.success("Done — see the Results, Graph and History tabs.")


def render_run_tab() -> None:
    can_run = render_vault_panel()
    if not Path(vault_dir() or "/nonexistent").is_dir():
        return
    render_options()
    st.info("Constellation only rewrites its own `## Tags` / `## Related Links` sections. "
            "Try Preview first; backups are listed under History & Undo.", icon="⚠️")
    if st.button("🚀 Run Constellation", type="primary", disabled=not can_run, width="stretch"):
        execute_run(vault_dir())
    elif st.session_state.get("log"):
        with st.expander("Last run log"):
            st.code("\n".join(st.session_state["log"][-500:]), language="text")


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


def table_with_downloads(rows: list, name: str) -> None:
    if not rows:
        st.write(f"No {name} yet.")
        return
    query = st.text_input("Search", key=f"search_{name}", placeholder="filter rows…")
    shown = helpers.filter_rows(rows, query)
    st.caption(f"{len(shown)} of {len(rows)} rows")
    st.dataframe(shown, width="stretch", hide_index=True)
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("⬇ CSV", to_csv(shown), f"{name}.csv", "text/csv", key=f"{name}_csv")
    c2.download_button("⬇ JSON", json.dumps(shown, indent=2, default=str), f"{name}.json",
                       "application/json", key=f"{name}_json")


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
        ("New links", rep.get("links_new", len(result.get("new_links", [])))),
        ("Tags", tags_total),
        ("Failed notes", len(failed)),
        ("Tokens used", f"{tokens:,}" if tokens else "n/a"),
        ("Duration", f"{duration:.1f}s" if duration else "n/a"),
    ]
    for row in (cards[:4], cards[4:]):
        for col, (label, val) in zip(st.columns(4), row):
            col.metric(label, val)


def render_previews(preview: dict) -> None:
    names = sorted(preview)
    changed = [n for n in names if preview[n].get("before") != preview[n].get("after")]
    st.caption(f"{len(changed)} of {len(names)} notes would change")
    pick = st.selectbox("Note", changed or names, key="preview_pick")
    entry = preview.get(pick) or {}
    diff = helpers.note_diff(entry.get("before", ""), entry.get("after", ""), pick)
    st.code(diff or "(no changes)", language="diff")


def render_results_tab() -> None:
    result = st.session_state.get("result")
    if not result:
        st.info("No results yet. Run the pipeline from the Run tab.")
        return
    if st.session_state.get("result_dry_run"):
        st.info("Preview run — no files were modified.")
    else:
        st.success(f"Updated {result.get('notes_written', 0)} note(s).")
    render_metrics(result)

    rep = result.get("run_report") or {}
    failed = rep.get("notes_failed") or result.get("failed_notes") or []
    if failed:
        with st.expander(f"⚠️ {len(failed)} failed note(s)"):
            st.write(failed)
    skipped = rep.get("skipped_due_to_edit") or []
    if skipped:
        with st.expander(f"{len(skipped)} note(s) skipped because they were edited during the run"):
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
            st.write("Per-note diffs appear here after a preview run, when the pipeline provides them.")
    with t_log:
        st.code("\n".join(st.session_state.get("log", [])), language="text")


# --------------------------------------------------------------------------- #
# Graph tab
# --------------------------------------------------------------------------- #
def render_graph_tab() -> None:
    result = st.session_state.get("result")
    links = (result or {}).get("links", [])
    if not links:
        st.info("No links to draw yet. Run the pipeline first.")
        return
    c1, c2 = st.columns([2, 1])
    query = c1.text_input("Search note", key="graph_q", placeholder="show a note and its neighbours")
    top_n = c2.slider("Most-connected notes", 5, 200, 40, key="graph_n")
    dot, n_nodes, n_edges = helpers.graph_dot(links, query, top_n)
    st.caption(f"Showing {n_nodes} notes and {n_edges} links")
    if n_nodes == 0:
        st.warning("Nothing matches that search.")
        return
    st.graphviz_chart(dot, width="stretch")


# --------------------------------------------------------------------------- #
# History & Undo tab
# --------------------------------------------------------------------------- #
def render_backups(path: str) -> None:
    st.subheader("Backups")
    try:
        backups = list_backups(path)
    except Exception as e:
        show_error(e)
        return
    if not backups:
        st.write("No backups yet. One is created before notes are modified.")
    for b in backups:
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        c1.write(f"**{b.get('name')}**")
        c2.write(str(b.get("created")))
        c3.write(f"{b.get('file_count', '?')} files")
        if c4.button("Restore", key=f"restore_{b.get('name')}"):
            try:
                st.success(f"Restored {restore_backup(path, b['name'])} file(s) from {b['name']}.")
            except Exception as e:
                show_error(e)


def render_strip(path: str) -> None:
    st.subheader("Remove everything Constellation wrote")
    st.caption("Strips the `## Tags` / `## Related Links` sections and frontmatter tags Constellation added, "
               "and deletes its cache and index. A backup is taken first, so this can be restored above.")
    confirm = st.checkbox("I understand this edits my notes", key="confirm_strip")
    if not st.button("Strip Constellation output", disabled=not confirm):
        return
    try:
        res = strip_vault(path, dry_run=False)
        msg = f"Cleaned {res.get('notes_cleaned', 0)} note(s)."
        if res.get("backup_dir"):
            msg += f" Backup: `{res['backup_dir']}`"
        st.success(msg)
    except Exception as e:
        show_error(e)


def render_history_tab() -> None:
    path = vault_dir()
    if not path or not Path(path).is_dir():
        st.info("Set a valid vault path in the Run tab.")
        return
    render_backups(path)
    st.divider()
    render_strip(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    render_sidebar()
    st.title("🌌 Constellation")
    st.caption("Links your Obsidian notes with LLM-extracted concepts, tags and `[[wikilinks]]`.")
    t_run, t_results, t_graph, t_hist = st.tabs(["Run", "Results", "Graph", "History & Undo"])
    with t_run:
        render_run_tab()
    with t_results:
        render_results_tab()
    with t_graph:
        render_graph_tab()
    with t_hist:
        render_history_tab()


main()
