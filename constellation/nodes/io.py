import hashlib
import json
import re
from pathlib import Path

from constellation.core import config
from constellation.core.backup import create_backup
from constellation.core.lock import vault_lock
from constellation.core.log import log
from constellation.core.state import AgentState

CACHE_FILENAME = ".linker_cache.json"
# Directories Obsidian / this tool own -- never treat their contents as notes.
IGNORED_DIRS = {".obsidian", ".trash", ".git", ".linker_faiss_index", ".constellation_backups"}
REPORT_FILENAME = ".linker_report.json"


_TAG_LINE = re.compile(r"^#[^\s#]+(?:\s+#[^\s#]+)*$")
_LINK_LINE = re.compile(r"^-\s*\[\[[^\]]+\]\](?:\s*(?:[—–-]|:)\s*.*)?$")


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


def get_core_hash(content: str, tool_tags=None) -> str:
    """MD5 of the note content, ignoring the auto-generated Tags/Related Links
    sections and any frontmatter tags Constellation added (tool_tags) -- without
    that, writing frontmatter tags would make every note look edited next run."""
    if tool_tags:
        content = apply_frontmatter_tags(content, [], tool_tags)[0]
    return hashlib.md5(strip_auto_sections(content).encode("utf-8")).hexdigest()


# --- Minimal YAML frontmatter handling ---------------------------------------
# Only the top-level `tags:` key is ever touched, line by line, so the rest of
# the user's frontmatter (comments, ordering, quoting) survives byte-for-byte.
# A full YAML round-trip would reformat it on every run.

_FM_TAGS_KEY = re.compile(r"^(tags|tag)\s*:\s*(.*)$")
_FM_LIST_ITEM = re.compile(r"^\s+-\s*(.*)$|^-\s+(.*)$")


def _split_frontmatter(content: str):
    """Returns (frontmatter_lines, body) or (None, content) if there is none."""
    if not content.startswith("---"):
        return None, content
    lines = content.split("\n")
    if lines[0].strip() != "---":
        return None, content
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            return lines[1:i], "\n".join(lines[i + 1:])
    return None, content


def _tag_norm(tag: str) -> str:
    return str(tag).strip().strip("'\"").lstrip("#").strip().lower()


def _parse_fm_tags(fm_lines):
    """Finds the top-level tags key: (start, end, key, style, values) or None.

    Supports `tags: [a, b]` (inline), a `tags:` block list, and `tags: a, b`
    (scalar, rewritten as a block list if we change it).
    """
    for i, line in enumerate(fm_lines):
        m = _FM_TAGS_KEY.match(line)
        if not m:
            continue
        key, rest = m.group(1), m.group(2).strip()
        if rest.startswith("[") and rest.endswith("]"):
            values = [v.strip() for v in rest[1:-1].split(",") if v.strip()]
            return i, i + 1, key, "inline", values
        if rest:
            values = [v for v in re.split(r"[,\s]+", rest) if v]
            return i, i + 1, key, "scalar", values
        values = []
        j = i + 1
        while j < len(fm_lines):
            item = _FM_LIST_ITEM.match(fm_lines[j])
            if not item:
                break
            value = (item.group(1) or item.group(2) or "").strip()
            if value:
                values.append(value)
            j += 1
        return i, j, key, "block", values
    return None


def apply_frontmatter_tags(content: str, add_tags, remove_tags):
    """Removes remove_tags (tags Constellation added previously) from the
    frontmatter `tags:` list, then adds add_tags that the user does not already
    have. Returns (new_content, added) where added are the tags WE contributed --
    the cache tracks them so a later run removes only those, never user tags.

    Creates the frontmatter if needed; removes the `tags:` key / the whole
    frontmatter again when it ends up empty, so add-then-remove is lossless.
    """
    remove_norm = {_tag_norm(t) for t in (remove_tags or []) if _tag_norm(t)}
    wanted = []
    for t in add_tags or []:
        clean = str(t).strip().lstrip("#").strip()
        if clean and _tag_norm(clean) not in {_tag_norm(w) for w in wanted}:
            wanted.append(clean)

    fm, body = _split_frontmatter(content)
    if fm is None:
        if not wanted:
            return content, []
        fm, body = [], content

    found = _parse_fm_tags(fm)
    if found:
        start, end, key, style, values = found
    else:
        start = end = len(fm)
        key, style, values = "tags", "block", []

    kept = [v for v in values if _tag_norm(v) not in remove_norm]
    removed_any = len(kept) != len(values)
    present = {_tag_norm(v) for v in kept}
    added = []
    for tag in wanted:
        if _tag_norm(tag) not in present:
            kept.append(tag)
            present.add(_tag_norm(tag))
            added.append(tag)

    if not added and not removed_any:
        return content, []

    if not kept:
        tag_lines = []
    elif style == "inline":
        tag_lines = [f"{key}: [{', '.join(kept)}]"]
    else:
        tag_lines = [f"{key}:"] + [f"  - {v}" for v in kept]

    new_fm = fm[:start] + tag_lines + fm[end:]
    if not any(line.strip() for line in new_fm):
        return body, added
    return "---\n" + "\n".join(new_fm) + "\n---\n" + body, added


def iter_notes(directory: Path, excluded_dirs=None):
    """Yields every user-authored .md file in the vault, filtering out internal and excluded folders."""
    directory = Path(directory)
    if excluded_dirs is None:
        excluded_dirs = config.EXCLUDED_DIRS

    excluded_patterns = {d.strip().lower().rstrip("/\\") for d in excluded_dirs if d and d.strip()}
    ignored_patterns = {d.lower() for d in IGNORED_DIRS}

    # rglob order depends on the filesystem; sorting keeps duplicate-filename
    # disambiguation (first one wins the bare title) stable between runs.
    for file in sorted(directory.rglob("*.md")):
        rel_parts = [p.lower() for p in file.relative_to(directory).parts[:-1]]
        if any(part in ignored_patterns for part in rel_parts):
            continue
        if any(part in excluded_patterns for part in rel_parts):
            continue
        rel_posix = file.relative_to(directory).parent.as_posix().lower()
        if any(rel_posix == pat or rel_posix.startswith(pat + "/") for pat in excluded_patterns):
            continue
        yield file


def iter_titled_notes(directory: Path):
    """Yields (title, file). Titles are file names; a later duplicate name gets
    'parent/name' so every note has a unique key in the cache."""
    directory = Path(directory)
    seen = set()
    for file in iter_notes(directory):
        title = file.name
        if title in seen:
            rel_str = str(file.relative_to(directory).as_posix())
            log(f"Warning: duplicate filename '{file.name}' found at '{rel_str}'. Disambiguating note title as '{file.parent.name}/{file.name}'.")
            title = f"{file.parent.name}/{file.name}"
        seen.add(title)
        yield title, file


def count_notes(directory, excluded_dirs=None) -> int:
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(1 for _ in iter_notes(directory, excluded_dirs=excluded_dirs))


def reset_vault_state(vault_path: str) -> str:
    """Deletes the cache and FAISS index so the next run re-processes everything."""
    import shutil
    vault = Path(vault_path).expanduser()
    removed = []
    cache = vault / CACHE_FILENAME
    if cache.exists():
        cache.unlink()
        removed.append(CACHE_FILENAME)
    index = vault / ".linker_faiss_index"
    if index.exists():
        shutil.rmtree(index, ignore_errors=True)
        removed.append(".linker_faiss_index")
    return ", ".join(removed) if removed else "nothing to reset"


def _load_cache(vault: Path) -> dict:
    cache_path = vault / CACHE_FILENAME
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log(f"Warning: could not read cache ({e}).")
        return {}


def strip_vault(vault_path, dry_run: bool = False) -> dict:
    """Removes everything Constellation ever wrote into the vault's notes.

    Every note loses its generated ``## Tags`` / ``## Related Links`` sections
    and the frontmatter tags Constellation added (user tags stay), then the cache and index are deleted so the vault is back to a pre-tool
    state. Notes are backed up first, so even this is undoable. With dry_run
    nothing is touched and notes_cleaned is the number that WOULD change.
    Returns {"notes_cleaned": int, "backup_dir": str | None}.
    """
    vault = Path(vault_path).expanduser()
    if not vault.is_dir():
        raise ValueError(f"The path '{vault_path}' is not a valid directory.")

    with vault_lock(vault):
        # Frontmatter tags we added are only identifiable through the cache.
        fm_tags = _load_cache(vault).get("frontmatter_tags") or {}
        changes = []
        for title, file in iter_titled_notes(vault):
            try:
                content = file.read_text(encoding="utf-8")
            except Exception as e:
                log(f"Warning: skipping '{title}' ({e})")
                continue
            cleaned = strip_auto_sections(apply_frontmatter_tags(content, [], fm_tags.get(title))[0])
            if cleaned.rstrip() == content.rstrip():
                continue
            changes.append((file, (cleaned + "\n") if cleaned else ""))

        if dry_run:
            log(f"Preview: would clean {len(changes)} note(s).")
            return {"notes_cleaned": len(changes), "backup_dir": None}

        backup_dir = None
        if changes:
            backup_dir = str(create_backup(vault, [f for f, _ in changes]))
            log(f"Backed up {len(changes)} note(s) to {backup_dir}")

        cleaned_count = 0
        for file, new_content in changes:
            try:
                _write_atomic(file, new_content)
                cleaned_count += 1
            except Exception as e:
                log(f"Failed to clean {file.name}: {e}")

        reset_vault_state(str(vault))
        log(f"Stripped Constellation output from {cleaned_count} note(s); cache and index deleted.")
        return {"notes_cleaned": cleaned_count, "backup_dir": backup_dir}


def vault_reader(state: AgentState):
    directory_path = (state.get("dir") or "").strip().strip("'\"").strip()
    if not directory_path:
        raise ValueError("No Obsidian vault directory was provided.")

    directory = Path(directory_path).expanduser()
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"The path '{directory_path}' is not a valid directory.")

    cache_path = directory / CACHE_FILENAME
    cache = _load_cache(directory)

    cached_files = cache.get("files") or {}
    all_cached_concepts = cache.get("concepts") or []
    all_cached_links = cache.get("links") or []
    cached_tags = cache.get("tags") or {}
    cached_fm_tags = cache.get("frontmatter_tags") or {}
    pending = set(cache.get("pending") or [])

    notes = []
    new_notes = []
    current_hashes = {}
    current_titles = set()

    for title, file in iter_titled_notes(directory):
        try:
            content = file.read_text(encoding="utf-8")
        except Exception as e:
            log(f"Warning: skipping '{title}' ({e})")
            continue

        current_titles.add(title)
        content_hash = get_core_hash(content, cached_fm_tags.get(title))
        current_hashes[title] = content_hash

        note_obj = {"title": title, "path": str(file.resolve()), "content": content}
        notes.append(note_obj)

        if title not in cached_files or cached_files[title].get("hash") != content_hash:
            new_notes.append(note_obj)

    reprocessed = {n["title"] for n in new_notes}
    valid_titles = current_titles - reprocessed
    # Notes a crashed run had already extracted (checkpoint) but never linked:
    # their concepts are reused, but they still need relationship extraction.
    resumed = sorted(t for t in valid_titles if t in pending)
    if resumed:
        log(f"Resuming {len(resumed)} note(s) checkpointed by an interrupted run.")

    retained_concepts = [c for c in all_cached_concepts if c.get("source_note") in valid_titles]
    # Only links FROM re-processed notes are regenerated. A link A->B where B was
    # edited is still valid as long as A is unchanged and B exists, so keep it.
    retained_links = [
        l for l in all_cached_links
        if l.get("from_note") in valid_titles
        and l.get("from_note") not in pending
        and l.get("to_note") in current_titles
    ]
    retained_tags = {k: v for k, v in cached_tags.items() if k in valid_titles}

    previous_rendered = cache.get("rendered")
    if not isinstance(previous_rendered, dict):
        # Caches written before "rendered" existed: reconstruct what was most
        # likely written from the cached links.
        previous_rendered = {
            src: [item["target"] for item in items]
            for src, items in build_effective_links(all_cached_links).items()
        }

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
        "raw_tags_by_note": {},
        "failed_notes": [],
        "skipped_due_to_edit": [],
        "previous_links": list(all_cached_links),
        "previous_tags": dict(cached_tags),
        "previous_rendered": previous_rendered,
        "previous_fm_tags": dict(cached_fm_tags),
        # Missing for caches written before settings were tracked: assume
        # nothing changed rather than rewriting the whole vault.
        "previous_settings": cache.get("settings") or render_settings(),
        "resumed_notes": resumed,
    }


def render_settings() -> dict:
    """Settings that change how notes are rendered without changing which links
    exist. If any differs from the last run, every linked/tagged note is redrawn."""
    return {
        "annotate": bool(config.ANNOTATE_LINKS),
        "tags_in_frontmatter": bool(config.TAGS_IN_FRONTMATTER),
    }


def _wikilink_name(title: str) -> str:
    """Obsidian wikilinks reference the note name without the .md suffix."""
    return title[:-3] if title.lower().endswith(".md") else title


def link_score(link: dict) -> float:
    """Score of a link; caches written before scoring existed default to 0."""
    try:
        return float(link.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rank_and_cap(items, max_links: int = None, key: str = "target"):
    """Keeps the max_links highest-scoring items, then orders them alphabetically.

    Ranking happens BEFORE the cap so the strongest links survive it; the
    alphabetical order afterwards is only for stable, readable output. Ties on
    score are broken alphabetically so the result is deterministic.
    """
    if max_links is None:
        max_links = config.MAX_LINKS_PER_NOTE
    ranked = sorted(items, key=lambda x: (-link_score(x), str(x.get(key, ""))))
    if max_links and max_links > 0:
        ranked = ranked[:max_links]
    return sorted(ranked, key=lambda x: str(x.get(key, "")))


def build_effective_links(links, existing_titles=None, backlinks: bool = None, max_links: int = None):
    """Returns {source_title: [{"target", "evidence", "score"}, ...]} -- exactly the
    links that would be rendered into each note.

    Everything that decides what ends up on disk (dedupe, backlinks, score
    ranking, cap) lives here so link_writer and the cache diff agree on it.
    When existing_titles is given, links touching a missing note are dropped.
    """
    if backlinks is None:
        backlinks = getattr(config, "BACKLINKS", False)
    candidates = {}

    def add(src, dst, evidence, score):
        if not src or not dst or src == dst:
            return
        if existing_titles is not None and (src not in existing_titles or dst not in existing_titles):
            return
        evidence = (evidence or "").strip().replace("\n", " ")
        current = candidates.get((src, dst))
        if current is None:
            candidates[(src, dst)] = {"target": dst, "evidence": evidence, "score": score}
            return
        current["score"] = max(current["score"], score)
        if not current["evidence"] and evidence:
            current["evidence"] = evidence

    for link in links:
        src, dst = link.get("from_note"), link.get("to_note")
        score = link_score(link)
        evidence = link.get("evidence", "")
        add(src, dst, evidence, score)
        if backlinks:
            add(dst, src, evidence, score)

    by_source = {}
    for (src, _dst), item in candidates.items():
        by_source.setdefault(src, []).append(item)
    return {src: rank_and_cap(items, max_links) for src, items in by_source.items()}


def render_note(
    base_content: str,
    tags=None,
    targets=None,
    max_links: int = None,
    annotate: bool = None,
) -> str:
    """Rebuilds a note's text with the auto-generated Tags / Related Links sections."""
    content = base_content.rstrip()
    prefix = "\n\n" if content else ""
    if tags:
        content += prefix + "## Tags\n" + " ".join(tags)
        prefix = "\n\n"
    if targets:
        if annotate is None:
            annotate = config.ANNOTATE_LINKS

        # Normalize targets into dicts of (target, evidence, score)
        target_items = []
        for t in targets:
            if isinstance(t, str):
                target_items.append({"target": t, "evidence": "", "score": 0.0})
            elif isinstance(t, dict):
                target_items.append({
                    "target": t.get("target", ""),
                    "evidence": t.get("evidence", ""),
                    "score": link_score(t),
                })
            elif isinstance(t, (tuple, list)):
                target_items.append({"target": t[0], "evidence": t[1] if len(t) > 1 else "", "score": 0.0})

        # Deduplicate targets, keeping the best score and any evidence
        seen_targets = {}
        for item in target_items:
            if not item["target"]:
                continue
            item["evidence"] = (item["evidence"] or "").strip().replace("\n", " ")
            current = seen_targets.get(item["target"])
            if current is None:
                seen_targets[item["target"]] = item
                continue
            current["score"] = max(current["score"], item["score"])
            if not current["evidence"] and item["evidence"]:
                current["evidence"] = item["evidence"]

        ranked = rank_and_cap(list(seen_targets.values()), max_links)
        sorted_targets = [(item["target"], item["evidence"]) for item in ranked]

        lines = []
        for target_title, evidence in sorted_targets:
            if annotate and evidence:
                lines.append(f"- [[{target_title}]] — {evidence}")
            else:
                lines.append(f"- [[{target_title}]]")

        if lines:
            content += prefix + "## Related Links\n" + "\n".join(lines)
    return (content + "\n") if content else ""


def _carry_forward_failed(state: AgentState, titles: set):
    """Links and tags for notes whose processing failed this run.

    vault_reader dropped them because the note was due for re-processing; if the
    LLM call then failed we would otherwise strip perfectly good links from disk
    (and from neighbours' backlinks) over a transient error.
    """
    failed = set(state.get("failed_notes") or [])
    links = list(state.get("links") or [])
    tags_by_note = dict(state.get("tags_by_note") or {})
    if not failed:
        return links, tags_by_note
    present = {(l.get("from_note"), l.get("to_note")) for l in links}
    for link in state.get("previous_links") or []:
        pair = (link.get("from_note"), link.get("to_note"))
        if pair[0] in failed and pair[1] in titles and pair not in present:
            links.append(link)
            present.add(pair)
    for title in failed:
        prev = (state.get("previous_tags") or {}).get(title)
        if prev and title not in tags_by_note:
            tags_by_note[title] = prev
    return links, tags_by_note


def _write_atomic(file_path: Path, content: str) -> None:
    """Temp-file swap so a crash mid-write never leaves a half-written note."""
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(file_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def content_hash(content: str) -> str:
    """Hash of the full file content (unlike get_core_hash, nothing is ignored).
    Watch mode uses it to recognise files the tool itself just wrote."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def link_writer(state: AgentState):
    log("\n--- [Link Writer] ---")
    dry_run = state.get("dry_run", False)
    if dry_run:
        log("Preview mode: no files will be modified.")

    notes = state.get("notes") or []
    titles = {n["title"] for n in notes}
    note_to_path = {n["title"]: n["path"] for n in notes}
    note_to_content = {n["title"]: n["content"] for n in notes}
    failed = set(state.get("failed_notes") or [])
    fm_mode = bool(config.TAGS_IN_FRONTMATTER)
    previous_fm_tags = state.get("previous_fm_tags") or {}

    links, tags_by_note = _carry_forward_failed(state, titles)
    effective = build_effective_links(links, existing_titles=titles)
    current_rendered = {src: [i["target"] for i in items] for src, items in effective.items()}
    previous_rendered = state.get("previous_rendered") or {}

    # Rewrite every note that was (successfully) re-processed, plus any note
    # whose rendered link set changed -- that covers notes that gained a link
    # AND notes that lost one because its target was deleted or renamed.
    processed = ({n["title"] for n in state.get("new_notes") or []}
                 | set(state.get("resumed_notes") or [])) - failed
    changed = {
        t for t in titles
        if set(current_rendered.get(t, [])) != set(previous_rendered.get(t, []))
    }
    notes_to_update = processed | changed
    if (state.get("previous_settings") or render_settings()) != render_settings():
        # Annotation or tag placement changed: redraw everything we ever wrote.
        log("Rendering settings changed since the last run; refreshing all linked/tagged notes.")
        notes_to_update |= set(current_rendered) | set(tags_by_note) | set(previous_fm_tags)
    notes_to_update = (notes_to_update & titles) - failed

    planned = []
    fm_tags_by_note = {t: list(v) for t, v in previous_fm_tags.items() if t in titles}
    planned_fm = {}
    for note_title in sorted(notes_to_update):
        file_path_str = note_to_path.get(note_title)
        original_content = note_to_content.get(note_title)
        if not file_path_str or original_content is None:
            continue

        note_tags = tags_by_note.get(note_title)
        targets = [
            {"target": _wikilink_name(i["target"]), "evidence": i["evidence"], "score": i["score"]}
            for i in effective.get(note_title, [])
        ]
        new_content = render_note(
            strip_auto_sections(original_content), None if fm_mode else note_tags, targets
        )
        # Always strip the frontmatter tags we added last time (the user may have
        # switched modes); re-add them only in frontmatter mode.
        new_content, added_fm = apply_frontmatter_tags(
            new_content, (note_tags or []) if fm_mode else [], previous_fm_tags.get(note_title)
        )
        # Trailing-whitespace-only differences are not worth touching a file for.
        if new_content.rstrip() == original_content.rstrip():
            continue

        updates = [f"{len(targets)} links"] if targets else []
        if note_tags:
            updates.append(f"{len(note_tags)} tags")
        summary = " and ".join(updates) or "removed stale links"
        planned.append((note_title, Path(file_path_str), original_content, new_content, summary))
        planned_fm[note_title] = added_fm

    backup_dir = None
    write_failed = set()
    if planned and not dry_run:
        # Snapshot exactly the files we are about to touch, before touching any.
        try:
            backup_dir = str(create_backup(state.get("dir", ""), [p[1] for p in planned]))
            log(f"Backed up {len(planned)} note(s) to {backup_dir}")
        except Exception as e:
            log(f"Could not create a backup ({e}); not writing any notes.")
            write_failed = {p[0] for p in planned}
            planned = []

    written = 0
    skipped_due_to_edit = []
    preview = {}
    written_hashes = {}
    for note_title, file_path, original_content, new_content, summary in planned:
        if dry_run:
            log(f"Would write {summary} to {note_title}")
            preview[note_title] = {"before": original_content, "after": new_content}
            written += 1
            continue

        # The pipeline can run for minutes; if the user saved the note in the
        # meantime, writing our stale copy would silently discard their edit.
        try:
            on_disk = file_path.read_text(encoding="utf-8")
        except Exception as e:
            log(f"Failed to re-read {note_title} before writing: {e}")
            write_failed.add(note_title)
            continue
        if on_disk != original_content:
            log(f"Skipped {note_title}: it was edited while Constellation was running. "
                f"It will be re-processed next run.")
            skipped_due_to_edit.append(note_title)
            continue

        try:
            _write_atomic(file_path, new_content)
            written += 1
            written_hashes[str(file_path)] = content_hash(new_content)
            if planned_fm.get(note_title):
                fm_tags_by_note[note_title] = planned_fm[note_title]
            else:
                fm_tags_by_note.pop(note_title, None)
            log(f"Wrote {summary} to {note_title}")
        except Exception as e:
            write_failed.add(note_title)
            log(f"Failed to write to {note_title}: {e}")

    if not planned:
        log("No notes needed updating.")

    # Record what is actually on disk now: notes we could not write keep their
    # old link set, so the next run still sees the difference and retries.
    rendered_out = dict(current_rendered)
    for title in failed | set(skipped_due_to_edit) | write_failed:
        if previous_rendered.get(title):
            rendered_out[title] = list(previous_rendered[title])
        else:
            rendered_out.pop(title, None)

    return {
        "notes_written": written,
        "links": links,
        "tags_by_note": tags_by_note,
        "skipped_due_to_edit": skipped_due_to_edit,
        "backup_dir": backup_dir,
        "rendered_links": {k: v for k, v in rendered_out.items() if v},
        "fm_tags_by_note": fm_tags_by_note,
        "preview": preview,
        "written_hashes": written_hashes,
    }


def _write_cache(cache_path, cache_data: dict) -> bool:
    try:
        tmp = Path(str(cache_path) + ".tmp")
        tmp.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
        tmp.replace(cache_path)
        return True
    except Exception as e:
        log(f"Warning: failed to save cache: {e}")
        return False


def save_checkpoint(state: AgentState, verified_titles) -> None:
    """Persists the extraction results right after concept verification.

    Extraction is the expensive part of a run; if relationship extraction or
    writing crashes afterwards, the next run reuses these concepts instead of
    paying for them again. Everything that describes what is ON DISK (links,
    rendered sets, frontmatter tags) is left as it was, and the notes are marked
    "pending" so the next run still links them. Real runs only.
    """
    if state.get("dry_run") or not state.get("cache_path"):
        return
    verified_titles = [t for t in verified_titles if t in (state.get("file_hashes") or {})]
    if not verified_titles:
        return
    cache = _load_cache(Path(state["cache_path"]).parent)
    files = dict(cache.get("files") or {})
    for title in verified_titles:
        files[title] = {"hash": state["file_hashes"][title]}
    cache["files"] = files
    cache["concepts"] = state.get("concepts") or []
    cache["tags"] = {**(cache.get("tags") or {}), **(state.get("tags_by_note") or {})}
    cache["pending"] = sorted(set(cache.get("pending") or []) | set(verified_titles))
    if _write_cache(state["cache_path"], cache):
        log(f"Checkpoint saved: {len(verified_titles)} note(s) extracted.")


def _iso_now() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_run_report(state: AgentState) -> dict:
    """A machine-readable summary of the run (also shown in the CLI / UI)."""
    from datetime import datetime

    from constellation.core import usage

    started_at = state.get("started_at") or _iso_now()
    try:
        duration = (datetime.now().astimezone() - datetime.fromisoformat(started_at)).total_seconds()
    except (TypeError, ValueError):
        duration = 0.0

    previous_pairs = {(l.get("from_note"), l.get("to_note")) for l in state.get("previous_links") or []}
    current_pairs = {(l.get("from_note"), l.get("to_note")) for l in state.get("links") or []}
    failed = sorted(state.get("failed_notes") or [])
    counters = usage.snapshot()
    return {
        "started_at": started_at,
        "duration_s": round(max(duration, 0.0), 2),
        "notes_total": len(state.get("notes") or []),
        "notes_processed": len({n["title"] for n in state.get("new_notes") or []} - set(failed)),
        "notes_failed": failed,
        "notes_written": int(state.get("notes_written") or 0),
        "links_total": len(current_pairs),
        "links_new": len(current_pairs - previous_pairs),
        "links_removed": len(previous_pairs - current_pairs),
        "llm_calls": counters["llm_calls"],
        "input_tokens": counters["input_tokens"],
        "output_tokens": counters["output_tokens"],
        "embedding_calls": counters["embedding_calls"],
        "skipped_due_to_edit": sorted(state.get("skipped_due_to_edit") or []),
        "backup_dir": state.get("backup_dir"),
        "dry_run": bool(state.get("dry_run", False)),
    }


def format_run_report(report: dict) -> str:
    """Compact human-readable version of build_run_report's output."""
    lines = [
        f"Notes: {report['notes_total']} total, {report['notes_processed']} processed, "
        f"{len(report['notes_failed'])} failed, {report['notes_written']} "
        f"{'would be written' if report['dry_run'] else 'written'}",
        f"Links: {report['links_total']} total (+{report['links_new']} / -{report['links_removed']})",
        f"LLM: {report['llm_calls']} calls, {report['input_tokens']} in / {report['output_tokens']} out tokens | "
        f"Embedding requests: {report['embedding_calls']}",
        f"Duration: {report['duration_s']}s",
    ]
    if report["notes_failed"]:
        lines.append(f"Failed (retried next run): {', '.join(report['notes_failed'])}")
    if report["skipped_due_to_edit"]:
        lines.append(f"Skipped (edited during run): {', '.join(report['skipped_due_to_edit'])}")
    if report["backup_dir"]:
        lines.append(f"Backup: {report['backup_dir']}")
    return "\n".join(lines)


def summary_reporter(state: AgentState):
    log("\n--- [Summary Reporter] ---")
    links = state.get("links") or []
    log(f"Done. {len(links)} total links, {len(state.get('new_links') or [])} created this run.")

    failed = list(state.get("failed_notes") or [])
    skipped = list(state.get("skipped_due_to_edit") or [])
    if failed:
        log(f"{len(failed)} note(s) failed and will be retried next run: {', '.join(sorted(failed))}")
    if skipped:
        log(f"{len(skipped)} note(s) were edited during the run and will be re-processed next run.")

    report = build_run_report(state)

    if state.get("dry_run", False):
        log("Preview mode: cache not written, so the next real run will redo this work.")
        return {"run_report": report}

    cache_path = state.get("cache_path")
    if cache_path:
        # A saved hash marks a note as done. Failed or skipped notes must not get
        # one, otherwise they would never be retried.
        not_done = set(failed) | set(skipped)
        cache_data = {
            "files": {
                title: {"hash": h}
                for title, h in (state.get("file_hashes") or {}).items()
                if title not in not_done
            },
            "concepts": state.get("concepts") or [],
            "links": links,
            "tags": state.get("tags_by_note") or {},
            "rendered": state.get("rendered_links") or {},
            "frontmatter_tags": state.get("fm_tags_by_note") or {},
            "settings": render_settings(),
        }
        if _write_cache(cache_path, cache_data):
            log(f"Saved cache to {cache_path}")

        report_path = Path(cache_path).parent / REPORT_FILENAME
        try:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as e:
            log(f"Warning: failed to save run report: {e}")

    return {"run_report": report}

