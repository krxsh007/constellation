"""Constellation pipeline.

Builds the LangGraph pipeline and exposes two entry points:

    run_pipeline(vault_dir)   -- blocking, returns the final state
    arun_pipeline(vault_dir)  -- the async version

Run ``streamlit run app.py`` for the UI, ``python main.py`` for the CLI, or
``constellation demo`` for an offline run on the bundled example vault.
"""

import asyncio
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from constellation.core import config, usage, utils
from constellation.core.lock import VaultLockedError, vault_lock
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


def initial_state(vault_dir: str, dry_run: bool = False) -> AgentState:
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
        "failed_notes": [],
        "skipped_due_to_edit": [],
        "previous_links": [],
        "previous_tags": {},
        "previous_rendered": {},
        "pair_similarity": {},
        "rendered_links": {},
        "backup_dir": None,
        "previous_fm_tags": {},
        "fm_tags_by_note": {},
        "previous_settings": {},
        "resumed_notes": [],
        "preview": {},
        "written_hashes": {},
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_report": {},
    }


async def arun_pipeline(vault_dir: str, dry_run: bool = False) -> dict:
    """Runs the full pipeline over a vault and returns the final state."""
    missing = config.missing_keys()
    if missing:
        raise RuntimeError(f"Missing API key(s): {', '.join(missing)}. Add them to .env or the sidebar.")

    vault = Path(utils.clean_path(str(vault_dir or "")) or ".")
    if not str(vault_dir or "").strip() or not vault.is_dir():
        raise ValueError(f"The path '{vault_dir}' is not a valid directory.")

    utils.reset_key_rotation()
    usage.reset()
    log(config.describe_config())
    # Raises VaultLockedError if another run is already working on this vault.
    with vault_lock(vault):
        return await app.ainvoke(initial_state(vault_dir, dry_run))


def run_pipeline(vault_dir: str, dry_run: bool = False) -> dict:
    """Blocking wrapper around arun_pipeline."""
    return asyncio.run(arun_pipeline(vault_dir, dry_run))


# --- Watch mode ---------------------------------------------------------------

def snapshot_mtimes(vault_dir) -> dict:
    """{absolute note path: mtime_ns} for every note the pipeline would read."""
    from constellation.nodes.io import iter_notes

    snapshot = {}
    for file in iter_notes(Path(vault_dir)):
        try:
            snapshot[str(file.resolve())] = file.stat().st_mtime_ns
        except OSError:
            continue
    return snapshot


def detect_changes(before: dict, after: dict, written_hashes: dict = None) -> list:
    """Paths added, removed or modified between two snapshots -- except files
    whose content is exactly what the tool itself just wrote, so a run never
    re-triggers itself."""
    from constellation.nodes.io import content_hash

    written_hashes = written_hashes or {}
    changed = []
    for path in sorted(set(before) | set(after)):
        if path not in after or path not in before:
            changed.append(path)
            continue
        if before[path] == after[path]:
            continue
        expected = written_hashes.get(path)
        if expected is not None:
            try:
                if content_hash(Path(path).read_text(encoding="utf-8")) == expected:
                    continue
            except OSError:
                pass
        changed.append(path)
    return changed


def watch(vault_dir: str, interval: float, dry_run: bool = False, run=None,
          sleep=None, max_runs: int = None) -> int:
    """Runs the pipeline, then re-runs it whenever a note changes (polling every
    `interval` seconds) until interrupted. Returns the number of runs.

    The snapshot is taken BEFORE each run, so edits made while a run is in
    progress still trigger the next one; the run's own writes are filtered out
    by comparing against its post-write hashes.
    """
    import time

    run = run or run_pipeline
    sleep = sleep or time.sleep
    runs = 0
    while True:
        before = snapshot_mtimes(vault_dir)
        written = {}
        try:
            state = run(vault_dir, dry_run=dry_run)
            written = state.get("written_hashes") or {}
            _print_report(state)
        except VaultLockedError as e:
            log(f"Skipping this cycle: {e}")
        runs += 1
        if max_runs is not None and runs >= max_runs:
            return runs

        log(f"\nWatching {vault_dir} for changes every {interval:g}s (Ctrl+C to stop)...")
        while True:
            sleep(interval)
            after = snapshot_mtimes(vault_dir)
            changed = detect_changes(before, after, written)
            if changed:
                log(f"Detected changes in {len(changed)} note(s); re-running.")
                break
            # Our own writes are acknowledged; compare against the new baseline.
            before, written = after, {}


# --- CLI ----------------------------------------------------------------------

def _print_report(state: dict) -> None:
    from constellation.nodes.io import format_run_report

    report = state.get("run_report") or {}
    if report:
        print("\n" + format_run_report(report))
    else:
        print(f"\nFinished. Notes updated: {state.get('notes_written', 0)}")


def parse_args(argv=None):
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Constellation: Autonomous Obsidian Knowledge Graph Generator",
        epilog="Try it without API keys: constellation demo",
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
    parser.add_argument(
        "--backlinks",
        action="store_true",
        help="Also write the reverse link B -> A for every link A -> B.",
    )
    parser.add_argument(
        "--frontmatter-tags",
        action="store_true",
        help="Write tags into the YAML frontmatter 'tags:' list instead of a ## Tags section.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["groq", "ollama"],
        default=None,
        help="LLM provider (default from LLM_PROVIDER, else groq).",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["google", "ollama"],
        default=None,
        help="Embedding provider (default from EMBEDDING_PROVIDER, else google).",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Keep running: re-run whenever a note changes, polling every SECONDS.",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--strip",
        action="store_true",
        help="Remove everything Constellation wrote from all notes (backed up first), "
             "then delete the cache and index.",
    )
    actions.add_argument(
        "--list-backups",
        action="store_true",
        help="List the pre-write backups stored in the vault, newest first.",
    )
    actions.add_argument(
        "--restore",
        metavar="NAME",
        default=None,
        help="Restore the notes saved in backup NAME (see --list-backups).",
    )
    return parser.parse_args(argv)


def apply_args_to_config(args) -> None:
    if args.max_links is not None:
        config.set_max_links(args.max_links)
    if args.annotate:
        config.set_annotate_links(True)
    if args.exclude is not None:
        config.set_excluded_dirs(args.exclude)
    if args.backlinks:
        config.set_backlinks(True)
    if args.frontmatter_tags:
        config.set_tags_in_frontmatter(True)
    if args.llm_provider or args.embedding_provider:
        config.set_providers(llm=args.llm_provider, embeddings=args.embedding_provider)


def demo_cli(argv=None) -> None:
    """``constellation demo``: copies the bundled example vault to a temp folder
    and runs the real pipeline on it with canned LLM/embedding answers, so it
    needs no API keys and no network."""
    import argparse

    from constellation import demo
    from constellation.core import log as logmod
    from constellation.ui.helpers import note_diff, strip_md

    parser = argparse.ArgumentParser(
        prog="constellation demo",
        description="Run Constellation offline on the bundled demo vault (no API keys needed).",
    )
    parser.add_argument("--write", action="store_true",
                        help="Write tags and links into the copied vault instead of only previewing them.")
    parser.add_argument("--dir", default=None, metavar="PATH",
                        help="Copy the demo vault here (must be empty) instead of a temp folder.")
    parser.add_argument("--note", default="Kubernetes.md",
                        help="Note whose diff to print.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the pipeline log.")
    args = parser.parse_args(argv)

    if not args.verbose:
        logmod.set_sink(lambda _msg: None)
    try:
        vault, state = demo.run_demo(args.dir, dry_run=not args.write)
    finally:
        logmod.clear_sink()

    print(f"Demo vault: {vault}")
    print("Mode: " + ("write (notes in the copy were updated)" if args.write else "preview (nothing written)"))
    _print_report(state)

    print("\nTags")
    for note, tags in sorted(state.get("tags_by_note", {}).items()):
        print(f"  {strip_md(note):<20} {' '.join(tags)}")

    print("\nLinks (score-ranked)")
    for link in sorted(state.get("links", []), key=lambda l: (l["from_note"], -l.get("score", 0))):
        print(f"  {strip_md(link['from_note']):<20} -> {strip_md(link['to_note']):<20} "
              f"{link.get('score', 0):.2f}  {link.get('relationship', '')}")

    preview = state.get("preview") or {}
    if args.note in preview:
        entry = preview[args.note]
        print(f"\nDiff for {args.note}")
        print(note_diff(entry["before"], entry["after"], args.note))
    if args.write:
        print("Open the folder above in Obsidian to explore the links.")
    else:
        print("Run `constellation demo --write` to apply the changes to the copy, "
              "or `streamlit run app.py` and click 'Try the demo vault'.")


def cli(argv=None):
    import os
    import sys

    from constellation.core.backup import list_backups, restore_backup
    from constellation.nodes.io import reset_vault_state, strip_vault

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "demo":
        demo_cli(argv[1:])
        return

    args = parse_args(argv)
    vault = args.vault_flag or args.vault
    if not vault:
        vault = input("Please enter your Obsidian vault directory path: ").strip()

    apply_args_to_config(args)

    vault_path = utils.clean_path(vault)
    if not vault_path or not os.path.isdir(vault_path):
        print(f"Error: '{vault}' is not a valid directory.")
        sys.exit(1)

    try:
        if args.list_backups:
            backups = list_backups(vault_path)
            if not backups:
                print("No backups found.")
            for b in backups:
                print(f"{b['name']}  {b['created']}  {b['file_count']} file(s)")
            return

        if args.restore:
            with vault_lock(vault_path):
                count = restore_backup(vault_path, args.restore)
            print(f"Restored {count} file(s) from backup '{args.restore}'.")
            return

        if args.strip:
            result = strip_vault(vault_path, dry_run=args.dry_run)
            verb = "Would clean" if args.dry_run else "Cleaned"
            print(f"{verb} {result['notes_cleaned']} note(s).")
            if result["backup_dir"]:
                print(f"Backup: {result['backup_dir']}")
            return

        if args.force:
            res = reset_vault_state(vault_path)
            print(f"Force re-processing requested. Reset: {res}")

        print(f"Running Constellation on: {vault_path}")
        if args.dry_run:
            print("[Preview mode: no files will be written]")

        if args.watch is not None:
            try:
                watch(vault_path, max(args.watch, 1.0), dry_run=args.dry_run)
            except KeyboardInterrupt:
                print("\nStopped watching.")
            return

        final_state = run_pipeline(vault_path, dry_run=args.dry_run)
        _print_report(final_state)
    except (VaultLockedError, ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()

