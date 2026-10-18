# Design notes

Constellation reads an Obsidian vault, asks an LLM what each note is about, finds notes that are genuinely related, and writes `## Tags` and `## Related Links` sections back into the markdown. This document explains why the pipeline is shaped the way it is, what was traded away, and where it is still weak. File references are to the Python app unless they say `obsidian-plugin/`.

## The pipeline

```
vault_reader → concept_extractor → concept_verifier → relationship_extractor → relationship_verifier → link_writer → summary_reporter
```

A LangGraph `StateGraph` over one `AgentState` dict (`constellation/core/state.py`). It is a straight line: no branches, no cycles. LangGraph earns its place through the typed state channels, async node execution and the stage log, not through control flow. The nodes are plain functions that return partial state updates, so each one is unit-tested by calling it with a hand-built dict.

## Two verification passes

The pipeline makes two pairs of LLM calls per note: **extract → verify** for concepts and tags, then **propose → verify** for cross-note relationships (`core/prompts.py`).

**Why.** The extraction prompt has to be generous: it is asked for 5–8 concepts and 3–5 tags, and models fill the quota with things mentioned in passing. Asking the same call to also be strict produces either noise or empty results, depending on the model. A second call with a narrower job ("remove anything not supported by the text, merge duplicates, fix formatting") is easier for a model to get right than one call that does both. The same goes for links. The relationship prompt sees a note's concepts next to up to 10 retrieved concepts from other notes and will happily find a "similar_to" between any two of them. The verifier sees only the proposed links and the concepts they mention, and its only instruction is to reject weak ones.

**Cost.** Roughly double the LLM calls of a single pass. Three things keep that affordable:
- the content-hash cache means a re-run only pays for edited notes;
- the verifier gets minified concepts (name, note, a 240-character explanation; `minify_concepts`) and link batches of 10;
- extraction and verification can use different models (`LLM_EXTRACTION`, `LLM_VERIFICATION`), so the verifier can be the cheaper one.

**Not trusted blindly.** After the LLM verifier, `filter_and_score_links` (`nodes/relations.py`) applies rules a model cannot be relied on for: links must connect two *different* notes that *exist*, model-supplied note names are resolved case-insensitively (with or without `.md`, with or without `[[ ]]`), and duplicates and already-known pairs are dropped.

Whether the second pass actually improves precision enough to justify its cost is a measurable question; the evaluation harness (see the README's Evaluation section) compares it against single-pass and retrieval-only baselines.

## Retrieval: FAISS on disk, not a database

Every verified concept is embedded as `"<name>: <explanation>"` and stored in a FAISS index inside the vault at `.linker_faiss_index/` (`core/vectorstore.py`). The relationship stage queries it with each new concept (k = 10) and only shows the LLM concepts from *other* notes.

**Why FAISS files in the vault.** It keeps the tool a local app with no service to run, no account and no port: copy the vault and the index comes with it. Vaults are personal-scale (hundreds to tens of thousands of concepts), where an exact flat index answers in milliseconds, so a vector database's features (ANN tuning, filtering, replication, concurrent writers) would be cost without benefit.

**What the index stores.** `index.faiss` (vectors), `docstore.json` (concept text and `{name, note}` metadata, in index order) and `manifest.json` (embedding provider, model, dimension). The manifest is checked on load: vectors from a different model live in a different space, and comparing them returns confident-looking garbage. A mismatch discards the index, and it is rebuilt from the cached concepts. That costs embedding calls only, no LLM calls.

**Keeping it in sync.** Before indexing, vectors of re-processed notes and of notes that no longer exist are dropped (`_index_concepts` in `nodes/concepts.py`), so the index never proposes links to deleted or renamed notes. Cached concepts whose vectors are missing are re-embedded.

**Preview runs** keep the index in memory (`hold_preview_index` / `take_preview_index`) so a dry run writes nothing at all into the vault, not even the index.

## The pickle-loading fix

LangChain's `FAISS.save_local` writes `index.pkl`, and `load_local` refuses to read it unless you pass `allow_dangerous_deserialization=True`. The earlier version did exactly that. The index lives *inside the vault*, and vaults get synced through iCloud, Dropbox, Syncthing and git, and shared. Unpickling a file from such a folder is arbitrary code execution for anyone who can drop a file into it.

The fix (`core/vectorstore.py`) persists the raw index with `faiss.write_index` and the docstore as JSON, then rebuilds the LangChain `FAISS` object from those two on load. A legacy `index.pkl` is never opened: it is logged, ignored, deleted on the next save, and the index is rebuilt from the cache. `tests/test_vectorstore.py` round-trips an index with no pickle file present and plants a malicious `index.pkl` to check that it is ignored.

## Incremental runs: cache invalidation by content hash

`.linker_cache.json` stores, per note, a hash of its content together with the concepts, tags, scored links and exactly what was rendered last time. `vault_reader` re-processes a note only when its hash changes.

**What is hashed.** `get_core_hash` (`nodes/io.py`) is the MD5 of the note *after* removing the sections Constellation itself wrote and any frontmatter tags it added. Without that, every write would invalidate the note it wrote to and each run would redo the whole vault. MD5 is used for change detection, not security.

**Why not mtimes.** Sync clients, editors and `git checkout` touch modification times without changing content, and a copied vault has fresh mtimes everywhere. Hashing content makes "unchanged" mean unchanged.

**Failure semantics.** A saved hash means "this note is done". So:
- notes whose extraction or verification raised are not given a hash, and they are retried on the next run instead of being silently skipped;
- notes skipped because the user edited them mid-run keep their old hash;
- after the concept verifier, extraction results are **checkpointed** to the cache and the notes are marked `pending`. If the run crashes in the relationship stages, the next run resumes from the checkpoint instead of paying for extraction again.

Changing a rendering setting (annotations, tag placement) is detected by comparing `render_settings()` to the previous run's and redraws every note the tool ever wrote to, without new LLM calls.

## Link scoring and capping

Each note gets at most `MAX_LINKS_PER_NOTE` links (5 by default). The earlier version capped by sorting targets alphabetically, so which links survived depended on note names. Now every link carries a score (`score_links` in `nodes/relations.py`):

```
score = best similarity between any concept pair of the two notes
      + 0.05 × (number of distinct supporting concept pairs − 1)
```

FAISS returns L2 distances, mapped to `1 / (1 + d)` so that higher is closer. The support bonus prefers note pairs connected by several independent concept relations over a single coincidental match, which is why a score can exceed 1. `rank_and_cap` keeps the top-scoring links and then sorts the survivors alphabetically, for stable diffs. Backlinks (optional) inherit the forward link's score and go through the same cap.

Links are also **dropped** when they go stale: if a target note is deleted or renamed, every note whose rendered link set changed is rewritten, not only the notes processed in this run.

## Edit safety and concurrency

The tool rewrites files people are actively editing, so most of the defensive code is about never losing their text.

- **Only its own lines.** `strip_auto_sections` removes a `## Tags` or `## Related Links` section only while it is made of lines the tool generates (a run of hashtags, `- [[link]]` bullets with or without an annotation). The first line that isn't one of those (prose, a heading, a code fence) ends the strip, and everything after it is kept. Frontmatter is edited line by line, only the top-level `tags:` key, and only the tags the tool added are ever removed. A full YAML round-trip would reformat the user's frontmatter on every run.
- **Edited mid-run.** A run can take minutes. Just before writing, `link_writer` re-reads the file. If it differs from what the pipeline read, the note is skipped and re-processed next run.
- **Atomic writes.** New content goes to `note.md.tmp` and is swapped in with `os.replace`, so a crash never leaves a half-written note. The cache is written the same way.
- **Backups and undo.** Before the first write of a run, exactly the files about to change are copied to `.constellation_backups/<timestamp>/` (`core/backup.py`). If the backup fails, nothing is written. `--restore` and the UI's History tab put a snapshot back, and `--strip` removes everything the tool ever wrote, after taking a backup of its own. The newest `MAX_BACKUPS` snapshots are kept.
- **One run per vault.** `.linker.lock` holds the owner's pid and start time (`core/lock.py`). A second run (CLI and UI at once, or two watch loops) gets `VaultLockedError` instead of racing on the cache. A lock whose process is dead or that is older than 6 hours is treated as stale, so a crash never blocks a vault forever.
- **Watch mode** snapshots mtimes *before* a run and ignores files whose content hash equals what the run itself just wrote, so its own writes never retrigger it, while user edits made during a run still do.

**Concurrency inside a run.** LLM calls run concurrently through `asyncio.gather` behind a semaphore (`LLM_CONCURRENCY`, default 2: Groq's free tier rate-limits long before throughput matters). One failed call returns its exception in place (`return_exceptions=True`) rather than failing the batch. Embedding is blocking network I/O with retry sleeps, so it runs in `asyncio.to_thread` to keep the event loop, and the Streamlit stage tracker driven by it, responsive.

## API key rotation and retries

Free-tier keys hit rate limits quickly. `async_invoke_with_retry` (`core/utils.py`) handles failures by type:

- **Rate limit:** rotate to the next key and retry immediately. Rotation goes through an `asyncio.Lock` and is compare-and-swap on the key index, so when several concurrent calls hit the same limit, the key advances once instead of once per call.
- **Request too large (413 / TPM):** halve the note text (down to 4,000 characters) and retry.
- **Malformed structured output:** after a few failures, switch from tool-calling to a raw-JSON prompt with a tolerant parser that repairs truncated JSON, normalises field names and drops invalid items while keeping the valid ones (`_fallback_parse`).
- **Anything else:** exponential backoff.

Google embeddings rotate the same way. Retries scale with the number of keys (`max(len(keys), 5)`).

## Testing offline

Every test runs without a network or API keys (`python3 -m unittest discover tests`). The two network boundaries, `abatch_invoke_with_retry` and the embedding model, are replaced by fakes that answer by prompt type, and the **real** LangGraph graph runs end to end over temporary vaults. That covers incremental re-runs, hand-written text surviving, excluded folders staying untouched, strip → restore round-trips, edits made mid-run, crash checkpoints and the pickle refusal.

`constellation demo` uses the same seam with richer canned answers (`constellation/demo.py`, `examples/demo-responses.json`). The canned extraction output deliberately includes junk concepts and tags, and the relationship output includes weak links, so the verification passes visibly remove them.

## The Obsidian plugin port

`obsidian-plugin/` is a TypeScript re-implementation, not a wrapper around the Python code, so it installs like any plugin with no Python, no subprocess and no native modules. What changed and why:

| Concern | Python app | Plugin | Why it differs |
|---|---|---|---|
| HTTP | LangChain clients | Obsidian `requestUrl` behind a small transport interface (`src/providers.ts`) | `fetch` from the Obsidian renderer hits CORS on provider APIs; `requestUrl` doesn't. The interface keeps the pipeline testable with a fake transport. |
| Vector search | FAISS flat L2 index | JSON list of L2-normalised `Float32Array`s, brute-force cosine, top-k (`src/vectors.ts`) | FAISS is a native library and can't ship in a plugin. At vault scale, a linear scan over normalised vectors is fast enough, and on normalised vectors cosine and L2 rank neighbours identically. |
| Index file | `index.faiss` + JSON | `index.json` with base64 Float32 vectors and the same manifest check | One portable file, nothing executable. |
| Change detection | MD5 of the note minus generated sections | cyrb53 of the note *body* minus generated sections | Synchronous and dependency-free. Frontmatter edits (often made by other plugins) no longer trigger re-extraction. |
| Writes | read, compare, temp-file swap | `vault.process()` with a compare inside the callback; frontmatter via `processFrontMatter` | Obsidian's own atomic APIs keep the editor, the metadata cache and other plugins consistent. |
| Undo | `.constellation_backups/` in the vault | snapshots in the plugin folder | Keeps plugin state out of the user's notes. |

What stayed identical: prompts, both verification passes, the scoring formula and support bonus, the cap, and the section-ownership rules. The rules are pinned by a parity fixture generated from the Python implementation (`obsidian-plugin/scripts/gen_parity_fixture.py` → `tests/fixtures/python-parity.json`), so either tool can clean up after the other.

## Alternatives considered and rejected

- **Single-pass extraction and linking.** Half the calls, but noticeably noisier tags and many generic "similar_to" links. Kept as an ablation in the evaluation harness rather than as the default.
- **Links from embedding similarity alone.** No LLM cost for linking, but no relationship type and no evidence, and topical neighbours aren't always useful links (two notes that both mention databases). Also kept as a baseline in the harness.
- **A vector database (Chroma, Qdrant, pgvector, sqlite-vss).** Adds a service or a heavier dependency for features a single-user vault doesn't need, and breaks "copy the vault, keep the index".
- **LangChain `save_local` / pickle.** Rejected for the security reason above.
- **mtime-based change detection.** Unreliable across sync tools and copies (see above).
- **Full YAML parse and dump of frontmatter.** Reformats quoting, ordering and comments on every write.
- **Inline links in the note body.** More "natural", but impossible to update or remove safely later. A dedicated, owned section can be rewritten idempotently.
- **Git for undo.** Most vaults aren't repositories, and auto-committing to one that is would be intrusive.

## Known limitations

- **Retrieval budget per note.** Each new concept retrieves k = 10 neighbours, but only the first 10 distinct retrieved concepts per *note* reach the prompt. A note's first concept can use up the whole budget, so neighbours of its later concepts are never considered. The demo vault shows this: Kubernetes' etcd concept crowds out its pod-scheduling neighbours. Splitting the budget across concepts would fix it.
- **Concept identity is the name.** Lookups in the relationship stages map concept names to notes, so two notes with an identically named concept can be confused there (the structural filter still prevents links to non-existent notes).
- **Score scale depends on the embedding model.** `1 / (1 + L2)` isn't calibrated, so scores aren't comparable across embedding models, and there is no absolute minimum-score threshold, only the per-note cap.
- **Self-verification.** By default the verifier is the same model as the extractor, so it shares the extractor's blind spots.
- **Edit check window.** Between re-reading a note and swapping in the new content there is a millisecond-scale window in which a save could be overwritten. The plugin narrows this with `vault.process()`.
- **The lock is local.** A pid-file lock doesn't protect a vault synced between two machines running the tool at the same time.
- **Demo mode patches module globals.** It is fine for the CLI, the tests and a single-user local Streamlit app, but not for a shared server. The demo also needs a source checkout, because `examples/` isn't in the wheel.
- **Legacy formatting.** Older modules predate `ruff format` and are listed in `[tool.ruff.format] exclude`.

## Next steps

1. Use the evaluation harness to decide per-stage model choices and whether both verification passes pay for themselves.
2. Split the retrieval budget across a note's concepts, and add an optional minimum score.
3. Key concepts by `(note, name)` throughout the relationship stages.
4. Switch to an HNSW or IVF index automatically above a size threshold.
5. Reformat the legacy modules in one commit and drop the format exclude list.
6. Publish the plugin to the Obsidian community catalogue.
