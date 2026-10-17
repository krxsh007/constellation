import asyncio
import json
import re

from langchain_core.documents import Document

from constellation.core import config, vectorstore
from constellation.core.log import log
from constellation.core.prompts import prompt_concept_extraction, prompt_concept_verification
from constellation.core.state import AgentState, ConceptExtractionOutput
from constellation.core.utils import abatch_invoke_with_retry, minify_concepts, prepare_note_text
from constellation.nodes.io import save_checkpoint


def _mark_failed(failed_notes: list, title: str) -> None:
    if title not in failed_notes:
        failed_notes.append(title)


async def concept_extractor(state: AgentState):
    log("\n--- [Concept Extractor] ---")
    all_extracted_concepts = []
    raw_tags_by_note = dict(state.get("raw_tags_by_note") or {})
    failed_notes = list(state.get("failed_notes") or [])

    new_notes = state.get("new_notes") or []
    if not new_notes:
        log("No new notes to extract concepts from.")
        return {"raw_concepts": [], "raw_tags_by_note": raw_tags_by_note, "failed_notes": failed_notes}

    log(f"Extracting raw concepts from {len(new_notes)} notes concurrently...")
    inputs_list = [{"text": prepare_note_text(note["content"])} for note in new_notes]

    results = await abatch_invoke_with_retry(
        prompt_concept_extraction, config.LLM_EXTRACTION, ConceptExtractionOutput, inputs_list
    )

    for note, result in zip(new_notes, results):
        if isinstance(result, Exception) or result is None:
            log(f"Failed to extract concepts from '{note['title']}': {str(result)[:200]}")
            _mark_failed(failed_notes, note["title"])
            continue

        for concept in result.concepts:
            concept_dict = concept.model_dump()
            concept_dict["source_note"] = note["title"]
            all_extracted_concepts.append(concept_dict)

        if result.tags:
            raw_tags_by_note[note["title"]] = result.tags

    log(f"Total raw concepts extracted: {len(all_extracted_concepts)}")
    return {
        "raw_concepts": all_extracted_concepts,
        "raw_tags_by_note": raw_tags_by_note,
        "failed_notes": failed_notes,
    }


async def concept_verifier(state: AgentState):
    log("\n--- [Concept Verifier] ---")
    all_verified_concepts = []
    tags_by_note = dict(state.get("tags_by_note") or {})
    failed_notes = list(state.get("failed_notes") or [])

    # Concepts of notes an interrupted run checkpointed but never linked count
    # as "new" again so relationship extraction picks them up.
    resumed = set(state.get("resumed_notes") or [])
    resumed_concepts = [c for c in state.get("concepts") or [] if c.get("source_note") in resumed]

    new_notes = state.get("new_notes") or []
    if not new_notes:
        return {
            "new_concepts": resumed_concepts,
            "concepts": state.get("concepts") or [],
            "tags_by_note": tags_by_note,
            "failed_notes": failed_notes,
        }

    notes_with_concepts = []
    inputs_list = []

    for note in new_notes:
        note_raw_concepts = [c for c in state.get("raw_concepts") or [] if c.get("source_note") == note["title"]]
        if not note_raw_concepts:
            continue

        notes_with_concepts.append(note)
        inputs_list.append({
            "text": prepare_note_text(note["content"]),
            "concepts": json.dumps(minify_concepts(note_raw_concepts)),
            "tags": json.dumps((state.get("raw_tags_by_note") or {}).get(note["title"], []))
        })

    results = []
    if inputs_list:
        log(f"Verifying concepts for {len(inputs_list)} notes concurrently...")
        results = await abatch_invoke_with_retry(
            prompt_concept_verification, config.LLM_VERIFICATION, ConceptExtractionOutput, inputs_list
        )
    else:
        log("No raw concepts found to verify.")

    for note, result in zip(notes_with_concepts, results):
        if isinstance(result, Exception) or result is None:
            log(f"Failed to verify concepts from '{note['title']}': {str(result)[:200]}")
            _mark_failed(failed_notes, note["title"])
            continue

        for concept in result.concepts:
            concept_dict = concept.model_dump()
            concept_dict["source_note"] = note["title"]
            all_verified_concepts.append(concept_dict)

        if result.tags:
            tags_by_note[note["title"]] = _normalise_tags(result.tags)

    # Embedding is blocking network I/O (with retry sleeps); run it in a worker
    # thread so the event loop -- and the UI driving it -- stays responsive.
    await asyncio.to_thread(_index_concepts, state, all_verified_concepts)

    log(f"Total verified concepts from new notes: {len(all_verified_concepts)}")
    update = {
        "new_concepts": all_verified_concepts + resumed_concepts,
        "concepts": (state.get("concepts") or []) + all_verified_concepts,
        "tags_by_note": tags_by_note,
        "failed_notes": failed_notes,
    }
    # Checkpoint the extraction spend before the (crash-prone) later stages.
    verified_titles = [n["title"] for n in new_notes if n["title"] not in failed_notes]
    save_checkpoint({**state, **update}, verified_titles)
    return update


def _normalise_tags(tags):
    """Obsidian tags: '#' prefixed, no spaces, no duplicates."""
    cleaned = []
    for tag in tags:
        tag = str(tag).strip().lstrip("#").strip()
        if not tag:
            continue
        # Remove trailing/leading punctuation often emitted by LLMs (e.g., commas, semicolons)
        tag = re.sub(r"^[^\w/]+|[^\w/]+$", "", tag)
        if not tag:
            continue
        tag = "#" + "-".join(tag.split()).lower()
        if tag not in cleaned:
            cleaned.append(tag)
    return cleaned


def _persist_index(state: AgentState, store, vault_dir: str) -> None:
    """Saves the index into the vault -- or, in preview mode, keeps it in memory
    for the relationship extractor so a dry run writes nothing to the vault."""
    if state.get("dry_run"):
        vectorstore.hold_preview_index(vault_dir, store)
    else:
        vectorstore.save_index(store, vault_dir, config.embeddings)


def _index_concepts(state: AgentState, verified_concepts):
    """Refreshes the vault-local FAISS index for the notes touched in this run."""
    vault_dir = state.get("dir", "")
    if not vault_dir:
        return

    store = vectorstore.load_index(vault_dir, config.embeddings)

    # Drop vectors for notes we just re-processed, plus any note that has since
    # been deleted or renamed in the vault -- otherwise the index keeps
    # proposing links to files that no longer exist.
    current_titles = {n["title"] for n in state.get("notes") or []}
    stale_titles = {n["title"] for n in state.get("new_notes") or []}
    stale_titles |= {t for t in vectorstore.indexed_notes(store) if t and t not in current_titles}

    removed = vectorstore.drop_notes(store, stale_titles)
    if removed:
        log(f"Removed {removed} stale vectors from the local index.")

    # Cached concepts whose vectors are missing -- the index was dropped because
    # the embedding model changed, it was a legacy pickle index, or it was
    # deleted. Re-embedding them is cheap compared to re-running the LLM.
    indexed = vectorstore.indexed_notes(store)
    missing = [
        c for c in state.get("concepts") or []
        if c.get("source_note") in current_titles
        and c.get("source_note") not in indexed
        and c.get("source_note") not in stale_titles
    ]
    if missing:
        log(f"Re-embedding {len(missing)} cached concepts missing from the index.")
    verified_concepts = list(verified_concepts) + missing

    if not verified_concepts:
        if removed:
            try:
                _persist_index(state, store, vault_dir)
            except Exception as e:
                log(f"Failed to save FAISS index: {e}")
        elif state.get("dry_run"):
            vectorstore.hold_preview_index(vault_dir, store)
        return

    docs = [
        Document(
            page_content=f"{c['concept_name']}: {c['explanation']}",
            metadata={"name": c["concept_name"], "note": c["source_note"]}
        )
        for c in verified_concepts
    ]

    log(f"Embedding and indexing {len(docs)} concepts into the local FAISS index...")
    try:
        store = vectorstore.add_documents(store, docs, config.embeddings)
        _persist_index(state, store, vault_dir)
        log(f"Index now holds {vectorstore.count(store)} concept vectors.")
    except Exception as e:
        log(f"Failed to index concepts into FAISS: {e}")
