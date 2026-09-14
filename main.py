"""Constellation pipeline.

Builds the LangGraph pipeline and exposes two entry points:

    run_pipeline(vault_dir)   -- blocking, returns the final state
    arun_pipeline(vault_dir)  -- the async version

Run ``streamlit run app.py`` for the UI, or ``python main.py`` for the CLI.
"""

import asyncio

from langgraph.graph import StateGraph, END

from constellation.core import config, utils
from constellation.core.log import log
from constellation.core.state import AgentState
from constellation.nodes.concepts import concept_extractor, concept_verifier
from constellation.nodes.io import link_writer, summary_reporter, vault_reader
from constellation.nodes.relations import relationship_extractor, relationship_verifier

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


def parse_args(argv=None):
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Constellation: Autonomous Obsidian Knowledge Graph Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "vault",
        nargs="?",
        default=os.getenv("OBSIDIAN_VAULT_DIR", ""),
        help="Path to the Obsidian vault directory (or use --vault).",
    )
    parser.add_argument(
        "--vault",
        dest="vault_flag",
        default=None,
        help="Path to the Obsidian vault directory.",
    )
    parser.add_argument(
        "-n", "--dry-run", "--preview",
        dest="dry_run",
        action="store_true",
        help="Preview changes without modifying vault notes or cache.",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force full re-processing by deleting cache and FAISS index.",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=None,
        help=f"Maximum wikilinks to append per note (default: {config.MAX_LINKS_PER_NOTE}).",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Append reasoning annotations to wikilinks: - [[Target]] — reason.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated vault folders to exclude (e.g. 'Templates,Archive').",
    )
    return parser.parse_args(argv)


def cli(argv=None):
    import os
    import sys
    from constellation.nodes.io import reset_vault_state

    args = parse_args(argv)
    vault = args.vault_flag or args.vault
    if not vault:
        vault = input("Please enter your Obsidian vault directory path: ").strip()

    if args.max_links is not None:
        config.set_max_links(args.max_links)
    if args.annotate:
        config.set_annotate_links(True)
    if args.exclude is not None:
        config.set_excluded_dirs(args.exclude)

    vault_path = utils.clean_path(vault)
    if not vault_path or not os.path.isdir(vault_path):
        print(f"Error: '{vault}' is not a valid directory.")
        sys.exit(1)

    if args.force:
        res = reset_vault_state(vault_path)
        print(f"Force re-processing requested. Reset: {res}")

    print(f"Running Constellation on: {vault_path}")
    if args.dry_run:
        print("[Preview mode: no files will be written]")

    final_state = run_pipeline(vault_path, dry_run=args.dry_run)
    print(f"\nFinished. Notes updated: {final_state.get('notes_written', 0)}")


if __name__ == "__main__":
    cli()
