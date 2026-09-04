"""Constellation -- local Streamlit UI.

Run with:  streamlit run app.py
"""

import shutil
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Constellation", page_icon="🌌", layout="wide")

from core import config, log as logmod, vectorstore
from core.utils import clean_path
from main import run_pipeline
from nodes.io import CACHE_FILENAME, count_notes

VAULT_KEY = "vault_path"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_vault(path: str) -> bool:
    cleaned = clean_path(path)
    return bool(cleaned) and Path(cleaned).is_dir()


def vault_stats(path: str) -> dict:
    vault = Path(clean_path(path))
    cache = vault / CACHE_FILENAME
    processed = 0
    if cache.exists():
        try:
            import json
            processed = len(json.loads(cache.read_text(encoding="utf-8")).get("files", {}))
        except Exception:
            processed = 0
    return {
        "notes": count_notes(vault),
        "processed": processed,
        "indexed": vectorstore.index_exists(str(vault)),
    }


def reset_vault_state(path: str) -> str:
    """Deletes the cache and FAISS index so the next run re-processes everything."""
    vault = Path(clean_path(path))
    removed = []
    cache = vault / CACHE_FILENAME
    if cache.exists():
        cache.unlink()
        removed.append(CACHE_FILENAME)
    index = vectorstore.index_path(str(vault))
    if index.exists():
        shutil.rmtree(index, ignore_errors=True)
        removed.append(vectorstore.INDEX_DIRNAME)
    return ", ".join(removed) if removed else "nothing to reset"


def parse_keys(text: str):
    """Accepts keys separated by newlines or commas."""
    return [k.strip() for chunk in text.splitlines() for k in chunk.split(",") if k.strip()]


# --------------------------------------------------------------------------- #
# Sidebar -- credentials and models
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")

    env_groq = len(config.GROQ_API_KEYS)
    env_google = len(config.GOOGLE_API_KEYS)
    if env_groq and env_google:
        st.success(f"Loaded {env_groq} Groq and {env_google} Google key(s) from `.env`")
    else:
        st.warning("Some API keys are missing from `.env` — add them below.")

    with st.expander("API keys", expanded=not (env_groq and env_google)):
        st.caption("One key per line. Extra keys are rotated automatically on rate limits.")
        groq_input = st.text_area("Groq keys (extraction & verification)", height=80, key="groq_keys")
        google_input = st.text_area("Google keys (embeddings)", height=80, key="google_keys")

    with st.expander("Models"):
        extraction_model = st.text_input("Extraction", value=config.LLM_EXTRACTION)
        verification_model = st.text_input("Verification", value=config.LLM_VERIFICATION)
        relationship_model = st.text_input("Relationship", value=config.LLM_RELATIONSHIP)

    st.caption(f"Embeddings: `{config.EMBEDDING_MODEL}`")


# Apply sidebar overrides on top of whatever .env provided.
if parse_keys(groq_input):
    config.set_groq_keys(parse_keys(groq_input) + config.GROQ_API_KEYS)
if parse_keys(google_input):
    config.set_google_keys(parse_keys(google_input) + config.GOOGLE_API_KEYS)
config.set_models(extraction_model, verification_model, relationship_model)


# --------------------------------------------------------------------------- #
# Main -- vault selection
# --------------------------------------------------------------------------- #
st.title("🌌 Constellation")
st.caption(
    "Reads every note in your vault, extracts concepts with an LLM, then writes "
    "Obsidian tags and `[[wikilinks]]` back into the files so the graph view connects up."
)

vault_path = st.text_input(
    "Obsidian vault folder",
    value=st.session_state.get(VAULT_KEY, ""),
    placeholder=r"C:\Users\you\Documents\MyVault",
    help="The absolute path to the folder your notes live in.",
)
st.session_state[VAULT_KEY] = vault_path

valid = is_vault(vault_path)
if vault_path and not valid:
    st.error("That folder does not exist. Paste the full path to your vault.")

if valid:
    stats = vault_stats(vault_path)
    c1, c2, c3 = st.columns(3)
    c1.metric("Markdown notes", stats["notes"])
    c2.metric("Already processed", stats["processed"])
    c3.metric("Concept index", "built" if stats["indexed"] else "not built")

    if stats["notes"] == 0:
        st.warning("No `.md` files found in that folder.")

    with st.expander("Options"):
        dry_run = st.checkbox(
            "Preview only (don't modify my notes)",
            value=False,
            help="Runs the whole pipeline and shows what it would write, leaving files untouched.",
        )
        force = st.checkbox(
            "Re-process every note (clears the cache and concept index)",
            value=False,
            help="By default only new or edited notes cost API calls.",
        )

    st.info(
        "Constellation appends a `## Tags` and `## Related Links` section to your notes. "
        "It only rewrites those two sections, but a backup of your vault is still a good idea "
        "before the first run.",
        icon="⚠️",
    )

    run = st.button("Run linker", type="primary", disabled=stats["notes"] == 0)

    # ----------------------------------------------------------------------- #
    # Run
    # ----------------------------------------------------------------------- #
    if run:
        if config.missing_keys():
            st.error(f"Missing API key(s): {', '.join(config.missing_keys())}. Add them in the sidebar.")
            st.stop()

        if force:
            st.toast(f"Reset: {reset_vault_state(vault_path)}")

        st.subheader("Progress")
        log_area = st.container(height=380)
        placeholder = log_area.empty()
        lines = []

        def sink(message: str):
            lines.append(message)
            placeholder.code("\n".join(lines[-500:]), language="text")

        logmod.set_sink(sink)
        try:
            with st.spinner("Running the pipeline — this can take a few minutes on a large vault..."):
                result = run_pipeline(clean_path(vault_path), dry_run=dry_run)
            st.session_state["result"] = result
            st.session_state["result_dry_run"] = dry_run
            st.session_state["log"] = list(lines)
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Pipeline failed: {e}")
            st.code(traceback.format_exc(), language="text")
        finally:
            logmod.clear_sink()

    # ----------------------------------------------------------------------- #
    # Results
    # ----------------------------------------------------------------------- #
    result = st.session_state.get("result")
    if result:
        was_dry_run = st.session_state.get("result_dry_run", False)
        st.subheader("Results")
        if was_dry_run:
            st.info("Preview run — no files were modified.")
        else:
            st.success(f"Updated {result.get('notes_written', 0)} note(s) in your vault.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Notes scanned", len(result.get("notes", [])))
        m2.metric("Notes processed", len(result.get("new_notes", [])))
        m3.metric("Concepts", len(result.get("concepts", [])))
        m4.metric("Links (new / total)",
                  f"{len(result.get('new_links', []))} / {len(result.get('links', []))}")

        tab_links, tab_tags, tab_concepts, tab_log = st.tabs(
            ["Links", "Tags", "Concepts", "Log"]
        )

        with tab_links:
            links = result.get("links", [])
            if links:
                st.dataframe(
                    [
                        {
                            "From note": l.get("from_note"),
                            "Wikilink": f"[[{str(l.get('to_note', '')).removesuffix('.md')}]]",
                            "Relationship": l.get("relationship"),
                            "Why": l.get("evidence"),
                        }
                        for l in links
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.write("No links yet. Notes need at least a couple of related topics to connect.")

        with tab_tags:
            tags_by_note = result.get("tags_by_note", {})
            if tags_by_note:
                st.dataframe(
                    [{"Note": note, "Tags": " ".join(tags)} for note, tags in sorted(tags_by_note.items())],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.write("No tags generated.")

        with tab_concepts:
            concepts = result.get("concepts", [])
            if concepts:
                st.dataframe(
                    [
                        {
                            "Note": c.get("source_note"),
                            "Concept": c.get("concept_name"),
                            "Category": c.get("category"),
                            "Importance": c.get("importance_score"),
                            "Explanation": c.get("explanation"),
                        }
                        for c in concepts
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.write("No concepts extracted.")

        with tab_log:
            st.code("\n".join(st.session_state.get("log", [])), language="text")
else:
    st.stop()
