import hashlib
import json
import re
from pathlib import Path

from core.log import log
from core.state import AgentState

CACHE_FILENAME = ".linker_cache.json"
# Directories Obsidian / this tool own -- never treat their contents as notes.
IGNORED_DIRS = {".obsidian", ".trash", ".git", ".linker_faiss_index"}


_TAG_LINE = re.compile(r"^#[^\s#]+(?:\s+#[^\s#]+)*$")
_LINK_LINE = re.compile(r"^-\s*\[\[[^\]]+\]\]$")


def _is_auto_line(section: str, stripped: str) -> bool:
    """True only for lines this tool itself would have written."""
    if section == "tags":
        return bool(_TAG_LINE.match(stripped))
    return bool(_LINK_LINE.match(stripped))


def strip_auto_sections(content: str) -> str:
    """Extracts the hand-written part of a note by removing the ## Tags and
    ## Related Links sections.

    Only lines this tool would have generated (a run of hashtags, or `- [[link]]`
    bullets) are dropped. The moment something else shows up -- prose, a heading,
    a code fence -- the section is considered over and everything from there on is
    kept, so text you add at the bottom of a note survives the next run.
    """
    in_code_block = False
    kept = []
    section = None      # None | "tags" | "links"
    pending_blanks = []

    def end_section():
        """Closes an auto-section, leaving exactly one blank line where it was so
        repeated runs don't accumulate whitespace."""
        nonlocal section, pending_blanks
        section = None
        pending_blanks = []
        while kept and not kept[-1].strip():
            kept.pop()
        if kept:
            kept.append("")

    for line in content.splitlines():
        stripped = line.strip()
        is_fence = stripped.startswith("```")
        if is_fence:
            in_code_block = not in_code_block

        # Never interpret markdown structure inside a fenced code block.
        if in_code_block or is_fence:
            if section:
                end_section()
            kept.append(line)
            continue

        if stripped == "## Tags":
            section, pending_blanks = "tags", []
            continue
        if stripped == "## Related Links":
            section, pending_blanks = "links", []
            continue

        if section:
            if not stripped:
                pending_blanks.append(line)
                continue
            if _is_auto_line(section, stripped):
                pending_blanks = []
                continue
            # Anything else is the user's own content -- stop stripping.
            end_section()

        kept.append(line)

    return "\n".join(kept).rstrip()


def get_core_hash(content: str) -> str:
    """MD5 of the note content, ignoring the auto-generated Tags/Related Links sections."""
    return hashlib.md5(strip_auto_sections(content).encode("utf-8")).hexdigest()


def iter_notes(directory: Path):
    """Yields every user-authored .md file in the vault."""
    for file in directory.rglob("*.md"):
        if any(part in IGNORED_DIRS for part in file.relative_to(directory).parts[:-1]):
            continue
        yield file


def count_notes(directory) -> int:
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(1 for _ in iter_notes(directory))


def vault_reader(state: AgentState):
    directory_path = (state.get("dir") or "").strip()
    if not directory_path:
        raise ValueError("No Obsidian vault directory was provided.")

    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"The path '{directory_path}' is not a valid directory.")

    cache_path = directory / CACHE_FILENAME
    cache = {"files": {}, "concepts": [], "links": [], "tags": {}}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            log(f"Warning: could not read cache ({e}). Treating the vault as new.")

    cached_files = cache.get("files", {})
    all_cached_concepts = cache.get("concepts", [])
    all_cached_links = cache.get("links", [])
    cached_tags = cache.get("tags", {})

    notes = []
    new_notes = []
    current_hashes = {}
    current_titles = set()

    for file in iter_notes(directory):
        title = file.name
        try:
            content = file.read_text(encoding="utf-8")
        except Exception as e:
            log(f"Warning: skipping '{title}' ({e})")
            continue

        current_titles.add(title)
        content_hash = get_core_hash(content)
        current_hashes[title] = content_hash

        note_obj = {"title": title, "path": str(file.resolve()), "content": content}
        notes.append(note_obj)

        if title not in cached_files or cached_files[title].get("hash") != content_hash:
            new_notes.append(note_obj)

    valid_titles = current_titles - {n["title"] for n in new_notes}

    retained_concepts = [c for c in all_cached_concepts if c.get("source_note") in valid_titles]
    retained_links = [
        l for l in all_cached_links
        if l.get("from_note") in valid_titles and l.get("to_note") in valid_titles
    ]
    retained_tags = {k: v for k, v in cached_tags.items() if k in valid_titles}

    log("\n--- [Vault Reader] ---")
    log(f"Found {len(notes)} notes in vault.")
    log(f"Notes needing processing: {len(new_notes)}")
    if retained_concepts or retained_links or retained_tags:
        log(f"Reused {len(retained_concepts)} cached concepts, {len(retained_links)} links, {len(retained_tags)} tag sets.")

    return {
        "notes": notes,
        "new_notes": new_notes,
        "dir": str(directory_path),
        "cache_path": str(cache_path),
        "file_hashes": current_hashes,
        "concepts": retained_concepts,
        "links": retained_links,
        "tags_by_note": retained_tags,
        "raw_tags_by_note": {}
    }


def render_note(base_content: str, tags, targets) -> str:
    """Rebuilds a note's text with the auto-generated Tags / Related Links sections."""
    content = base_content
    if tags:
        content += "\n\n## Tags\n" + " ".join(tags)
    if targets:
        links_block = "\n".join(f"- [[{t}]]" for t in sorted(targets))
        content += "\n\n## Related Links\n" + links_block
    return content + "\n"


def link_writer(state: AgentState):
    log("\n--- [Link Writer] ---")
    dry_run = state.get("dry_run", False)
    if dry_run:
        log("Preview mode: no files will be modified.")

    links = state.get("links", [])
    note_to_path = {n["title"]: n["path"] for n in state.get("notes", [])}
    note_to_content = {n["title"]: n["content"] for n in state.get("notes", [])}

    links_by_source = {}
    for link in links:
        from_note = link.get("from_note")
        to_note = link.get("to_note")
        if from_note and to_note:
            # Obsidian wikilinks reference the note name without the .md suffix.
            target_title = to_note[:-3] if to_note.lower().endswith(".md") else to_note
            links_by_source.setdefault(from_note, set()).add(target_title)

    tags_by_note = state.get("tags_by_note", {})

    # Only rewrite notes that changed this run or that gained a link -- untouched
    # notes keep whatever is already on disk.
    new_notes_titles = {note["title"] for note in state.get("new_notes", [])}
    new_link_sources = {l["from_note"] for l in state.get("new_links", []) if l.get("from_note")}
    allowed_updates = new_notes_titles | new_link_sources

    notes_to_update = (set(links_by_source) | set(tags_by_note)) & allowed_updates

    written = 0
    for note_title in sorted(notes_to_update):
        file_path_str = note_to_path.get(note_title)
        original_content = note_to_content.get(note_title)

        if not file_path_str or original_content is None:
            continue

        note_tags = tags_by_note.get(note_title)
        targets = links_by_source.get(note_title)
        new_content = render_note(strip_auto_sections(original_content), note_tags, targets)

        if new_content == original_content:
            continue

        updates = []
        if targets:
            updates.append(f"{len(targets)} links")
        if note_tags:
            updates.append(f"{len(note_tags)} tags")
        summary = " and ".join(updates)

        if dry_run:
            log(f"Would write {summary} to {note_title}")
            written += 1
            continue

        file_path = Path(file_path_str)
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            temp_path.write_text(new_content, encoding="utf-8")
            temp_path.replace(file_path)
            written += 1
            log(f"Wrote {summary} to {note_title}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            log(f"Failed to write to {note_title}: {e}")

    if not notes_to_update:
        log("No notes needed updating.")

    return {"notes_written": written}


def summary_reporter(state: AgentState):
    log("\n--- [Summary Reporter] ---")
    links = state.get("links", [])
    log(f"Done. {len(links)} total links, {len(state.get('new_links', []))} created this run.")

    if state.get("dry_run", False):
        log("Preview mode: cache not written, so the next real run will redo this work.")
        return {}

    cache_path = state.get("cache_path")
    if cache_path:
        cache_data = {
            "files": {title: {"hash": h} for title, h in state.get("file_hashes", {}).items()},
            "concepts": state.get("concepts", []),
            "links": links,
            "tags": state.get("tags_by_note", {})
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
            log(f"Saved cache to {cache_path}")
        except Exception as e:
            log(f"Warning: failed to save cache: {e}")

    return {}
