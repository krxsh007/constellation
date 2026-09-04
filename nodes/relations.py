import json

from core import config, vectorstore
from core.log import log
from core.prompts import prompt_relation, prompt_relation_verify
from core.state import AgentState, RelationshipExtractionOutput
from core.utils import invoke_with_retry, minify_concepts


def relationship_extractor(state: AgentState):
    log("\n--- [Relationship Extractor] ---")
    new_concepts = state.get("new_concepts", [])
    all_concepts = state.get("concepts", [])

    # Smart resumption: if nothing new was extracted this run (e.g. we are resuming after
    # an error) but we have cached concepts and no links yet, treat the cache as new.
    if not new_concepts:
        cached_links = state.get("links", [])
        if all_concepts and not cached_links:
            log("No new concepts, but cached concepts exist with 0 links. Re-linking all concepts...")
            new_concepts = all_concepts
        else:
            log("No new concepts extracted. Skipping relationship extraction.")
            return {"raw_links": []}

    store = vectorstore.load_index(state.get("dir", ""), config.embeddings)
    if store is None:
        log("No FAISS index available -- cannot search for related concepts.")
        return {"raw_links": []}

    try:
        # Group new concepts by source note so we reason note-by-note.
        concepts_by_note = {}
        for c in new_concepts:
            note = c.get("source_note")
            if note:
                concepts_by_note.setdefault(note, []).append(c)

        all_relations = []
        log(f"Extracting relationships for {len(concepts_by_note)} notes...")

        for note_title, note_concepts in concepts_by_note.items():
            retrieved_concepts = []
            retrieved_names = set()

            for c in note_concepts:
                query = f"{c['concept_name']}: {c['explanation']}"
                try:
                    results = store.similarity_search(query, k=10)
                except Exception as e:
                    log(f"  Similarity search failed for '{c['concept_name']}': {str(e)[:160]}")
                    continue

                for doc in results:
                    name = doc.metadata.get("name")
                    note = doc.metadata.get("note")

                    # Exclude concepts from the same note.
                    if note == note_title:
                        continue

                    key = (name, note)
                    if key in retrieved_names:
                        continue
                    retrieved_names.add(key)

                    match = next(
                        (x for x in all_concepts
                         if x["concept_name"] == name and x["source_note"] == note),
                        None
                    )
                    if match:
                        retrieved_concepts.append(match)

            # Top 15 most similar keeps the prompt compact and precise.
            retrieved_concepts = retrieved_concepts[:15]

            if not retrieved_concepts:
                continue

            log(f"  '{note_title}': {len(note_concepts)} concepts vs {len(retrieved_concepts)} related concepts...")

            extracted_output = invoke_with_retry(
                prompt_relation, config.LLM_RELATIONSHIP, RelationshipExtractionOutput,
                {
                    "new_concepts": json.dumps(minify_concepts(note_concepts)),
                    "existing_concepts": json.dumps(minify_concepts(retrieved_concepts))
                }
            )

            if extracted_output and extracted_output.links:
                all_relations.extend([link.model_dump() for link in extracted_output.links])

        log(f"Total raw cross-note relationships extracted: {len(all_relations)}")
        return {"raw_links": all_relations}
    except Exception as e:
        log(f"Failed to extract cross-note relationships: {e}")
        return {"raw_links": []}


def relationship_verifier(state: AgentState):
    log("\n--- [Relationship Verifier] ---")
    raw_links = state.get("raw_links", [])
    concepts = state.get("concepts", [])

    if not raw_links or not concepts:
        log("No raw links or concepts to verify.")
        return {"links": state.get("links", []), "new_links": []}

    try:
        # Verify in batches of 20 to stay inside Groq's context/token limits (413).
        batch_size = 20
        all_verified_links = []

        log(f"Verifying {len(raw_links)} cross-note relationships in batches of {batch_size}...")

        concept_lookup = {c["concept_name"]: c for c in concepts}

        for i in range(0, len(raw_links), batch_size):
            batch_links = raw_links[i:i + batch_size]

            concepts_in_batch = set()
            for link in batch_links:
                if link.get("source"):
                    concepts_in_batch.add(link["source"])
                if link.get("target"):
                    concepts_in_batch.add(link["target"])

            relevant_concepts = [concept_lookup[n] for n in concepts_in_batch if n in concept_lookup]
            minified_concepts = minify_concepts(relevant_concepts)

            batch_no = i // batch_size + 1
            total_batches = (len(raw_links) - 1) // batch_size + 1
            log(f"  Batch {batch_no}/{total_batches} ({len(batch_links)} links, {len(minified_concepts)} concepts)...")

            verified_output = invoke_with_retry(
                prompt_relation_verify, config.LLM_VERIFICATION, RelationshipExtractionOutput,
                {
                    "concepts": json.dumps(minified_concepts),
                    "relationships": json.dumps(batch_links)
                }
            )

            if verified_output and verified_output.links:
                all_verified_links.extend([link.model_dump() for link in verified_output.links])

        # --- Structural filtering: no self-links, no links to non-existent notes ---
        note_titles = {note["title"] for note in state.get("notes", [])}
        title_map = {t.lower(): t for t in note_titles}
        concept_to_note = {c["concept_name"]: c["source_note"] for c in concepts}
        existing_pairs = {
            (l.get("from_note"), l.get("to_note")) for l in state.get("links", [])
        }
        valid_links = []
        seen_pairs = set()
        prevented = 0

        for link in all_verified_links:
            from_note = link.get("from_note") or concept_to_note.get(link.get("source"))
            to_note = link.get("to_note") or concept_to_note.get(link.get("target"))

            from_note = _resolve_title(from_note, title_map)
            to_note = _resolve_title(to_note, title_map)

            if not from_note or not to_note:
                prevented += 1
                continue
            if from_note not in note_titles or to_note not in note_titles:
                prevented += 1
                continue
            if from_note == to_note:
                prevented += 1
                continue
            if (from_note, to_note) in seen_pairs or (from_note, to_note) in existing_pairs:
                prevented += 1
                continue

            seen_pairs.add((from_note, to_note))
            link["from_note"] = from_note
            link["to_note"] = to_note
            valid_links.append(link)

        log(f"Total verified cross-note relationships: {len(valid_links)} (rejected {prevented})")
        return {
            "links": state.get("links", []) + valid_links,
            "new_links": valid_links
        }
    except Exception as e:
        log(f"Failed to verify relationships: {e}")
        return {"links": state.get("links", []), "new_links": []}


def _resolve_title(note: str, title_map: dict) -> str:
    """Resolves a model-supplied note name to a real vault filename, tolerating
    a missing '.md' suffix and case differences."""
    if not note:
        return note
    candidate = note if note.lower().endswith(".md") else f"{note}.md"
    return title_map.get(candidate.lower(), note)
