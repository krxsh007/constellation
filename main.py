"""NeuroNote pipeline.

Builds the LangGraph pipeline and exposes two entry points:

    run_pipeline(vault_dir)   -- blocking, returns the final state
    arun_pipeline(vault_dir)  -- the async version

Run ``streamlit run app.py`` for the UI, or ``python main.py`` for the CLI.
"""

import asyncio

from langgraph.graph import StateGraph, END

from core import config, utils
from core.log import log
from core.state import AgentState
from nodes.concepts import concept_extractor, concept_verifier
from nodes.io import link_writer, summary_reporter, vault_reader
from nodes.relations import relationship_extractor, relationship_verifier

# --- Build graph ---
graph = StateGraph(AgentState)

graph.add_node("vault_reader",           vault_reader)
graph.add_node("concept_extractor",      concept_extractor)
graph.add_node("concept_verifier",       concept_verifier)
graph.add_node("relationship_extractor", relationship_extractor)
graph.add_node("relationship_verifier",  relationship_verifier)
graph.add_node("link_writer",            link_writer)
graph.add_node("summary_reporter",       summary_reporter)

graph.set_entry_point("vault_reader")

graph.add_edge("vault_reader",           "concept_extractor")
graph.add_edge("concept_extractor",      "concept_verifier")
graph.add_edge("concept_verifier",       "relationship_extractor")
graph.add_edge("relationship_extractor", "relationship_verifier")
graph.add_edge("relationship_verifier",  "link_writer")
graph.add_edge("link_writer",            "summary_reporter")
graph.add_edge("summary_reporter",       END)

app = graph.compile()


def initial_state(vault_dir: str, dry_run: bool = False) -> dict:
    return {
        "notes": [],
        "new_notes": [],
        "raw_concepts": [],
        "concepts": [],
        "new_concepts": [],
        "raw_links": [],
        "links": [],
        "new_links": [],
        "dir": str(vault_dir),
        "cache_path": "",
        "file_hashes": {},
        "raw_tags_by_note": {},
        "tags_by_note": {},
        "dry_run": dry_run,
        "notes_written": 0,
    }


async def arun_pipeline(vault_dir: str, dry_run: bool = False) -> dict:
    """Runs the full pipeline over a vault and returns the final state."""
    missing = config.missing_keys()
    if missing:
        raise RuntimeError(f"Missing API key(s): {', '.join(missing)}. Add them to .env or the sidebar.")

    utils.reset_key_rotation()
    log(config.describe_config())
    return await app.ainvoke(initial_state(vault_dir, dry_run))


def run_pipeline(vault_dir: str, dry_run: bool = False) -> dict:
    """Blocking wrapper around arun_pipeline."""
    return asyncio.run(arun_pipeline(vault_dir, dry_run))


if __name__ == "__main__":
    import os
    import sys

    vault = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OBSIDIAN_VAULT_DIR", "")
    if not vault:
        vault = input("Please enter your Obsidian vault directory path: ").strip()

    final_state = run_pipeline(vault)
    print(f"\nNotes updated: {final_state.get('notes_written', 0)}")
