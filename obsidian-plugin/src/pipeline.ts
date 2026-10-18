// The seven-stage pipeline (vault reader → concept extractor → concept verifier
// → relationship extractor → relationship verifier → link writer → summary
// reporter), ported from constellation/nodes/*. It talks to Obsidian only
// through VaultIO / PluginStorage so it can be exercised offline in tests.
//
// A run is split in two: plan() does every LLM/embedding call and computes the
// exact per-note changes without touching notes; commit() snapshots, writes
// and persists. "Preview changes" shows a plan and commits it on Apply.

import { CONCEPT_SCHEMA, RELATIONSHIP_SCHEMA } from "./json";
import { buildEffectiveLinks, Concept, filterAndScoreLinks, minifyConcepts, pairKey, Relationship } from "./links";
import { CONCEPT_EXTRACTION, CONCEPT_VERIFICATION, RELATION, RELATION_VERIFY } from "./prompts";
import { createEmbedder, Http, LLMClient, mapPool } from "./providers";
import {
	coreHash,
	normaliseTags,
	prepareNoteText,
	renderNote,
	stripAutoSections,
	tagNorm,
} from "./sections";
import {
	Cache,
	ConstellationSettings,
	emptyCache,
	ProgressEvent,
	RenderSettings,
	RunReport,
	STAGES,
	Usage,
} from "./types";
import { VectorIndex } from "./vectors";

export interface VaultIO {
	/** Every note to consider (excluded folders already filtered out), as vault paths. */
	listNotes(): string[];
	read(path: string): Promise<string>;
	/** Atomically replace the content if it still equals `expected`; false if the user edited it. */
	writeIfUnchanged(path: string, expected: string, next: string): Promise<boolean>;
	/** Unconditional write (undo). */
	write(path: string, content: string): Promise<void>;
	exists(path: string): boolean;
	/** Tags currently in the note's frontmatter. */
	frontmatterTags(path: string): string[];
	/** Remove `remove`, add `add` in the frontmatter tags list. */
	updateFrontmatterTags(path: string, add: string[], remove: string[]): Promise<void>;
	/** Text to put inside [[...]] when `sourcePath` links to `targetPath`. */
	linkText(targetPath: string, sourcePath: string): string;
}

export interface Backup {
	id: string;
	createdAt: string;
	files: Record<string, string>;
	cache: Cache;
	/** Notes whose vectors should be re-synced after an undo. */
	processed: string[];
	label: string;
}

export interface PluginStorage {
	loadCache(): Promise<Cache>;
	saveCache(cache: Cache): Promise<void>;
	loadIndex(): Promise<VectorIndex>;
	saveIndex(index: VectorIndex): Promise<void>;
	deleteIndex(): Promise<void>;
	saveBackup(backup: Backup, keep: number): Promise<void>;
	latestBackup(): Promise<Backup | null>;
	deleteBackup(id: string): Promise<void>;
}

export interface PipelineDeps {
	io: VaultIO;
	storage: PluginStorage;
	http: Http;
	settings: ConstellationSettings;
	log: (msg: string) => void;
	progress: (e: ProgressEvent) => void;
	isCancelled?: () => boolean;
}

export interface PlannedChange {
	path: string;
	before: string;
	after: string;
	fmAdd: string[];
	fmRemove: string[];
	/** Our frontmatter tags once this change is applied. */
	fmOurs: string[];
	summary: string;
}

export interface RunPlan {
	dryRun: boolean;
	scope: "vault" | "note";
	changes: PlannedChange[];
	/** Cache as it will be once every change is written (commit adjusts for failures). */
	nextCache: Cache;
	previousRendered: Record<string, string[]>;
	index: VectorIndex;
	indexDirty: boolean;
	failed: string[];
	processed: string[];
	notesTotal: number;
	previousLinks: Relationship[];
	usage: Usage;
	startedAt: number;
}

export class CancelledError extends Error {
	constructor() {
		super("Run cancelled.");
	}
}

export function renderSettings(s: ConstellationSettings): RenderSettings {
	return {
		annotate: !!s.annotateLinks,
		tagsInFrontmatter: !!s.tagsInFrontmatter,
		maxLinks: s.maxLinksPerNote,
		backlinks: !!s.backlinks,
	};
}

const sameSet = (a: string[] = [], b: string[] = []) => {
	if (a.length !== b.length) return false;
	const s = new Set(a);
	return b.every((x) => s.has(x));
};

/**
 * Computes frontmatter tag changes. `ours` are tags Constellation added last
 * time; user tags are never removed, and a wanted tag the user already has is
 * left as theirs.
 */
export function planFrontmatter(present: string[], ours: string[], wanted: string[]) {
	const oursNorm = new Set(ours.map(tagNorm));
	const presentNorm = new Set(present.map(tagNorm));
	const userNorm = new Set(present.map(tagNorm).filter((t) => !oursNorm.has(t)));
	const next: string[] = [];
	const nextNorm = new Set<string>();
	for (const w of wanted) {
		const clean = String(w).trim().replace(/^#+/, "").trim();
		const n = tagNorm(clean);
		if (!n || userNorm.has(n) || nextNorm.has(n)) continue;
		next.push(clean);
		nextNorm.add(n);
	}
	const remove = present.filter((t) => oursNorm.has(tagNorm(t)) && !nextNorm.has(tagNorm(t)));
	const add = next.filter((t) => !presentNorm.has(tagNorm(t)));
	return { add, remove, next };
}

export async function plan(deps: PipelineDeps, opts: { dryRun: boolean; scopePath?: string }): Promise<RunPlan> {
	const { io, storage, settings, log, progress } = deps;
	const startedAt = Date.now();
	const usage: Usage = { llmCalls: 0, inputTokens: 0, outputTokens: 0, embeddingCalls: 0 };
	const llm = new LLMClient(deps.http, settings, usage, log);
	const concurrency = Math.max(1, Math.floor(settings.concurrency) || 1);
	const checkCancel = () => {
		if (deps.isCancelled?.()) throw new CancelledError();
	};
	let stage = 0;
	const begin = (i: number, detail?: string) => {
		stage = i;
		log(`\n--- [${STAGES[i]}] ---`);
		progress({ stage: i, status: "running", detail });
	};
	const end = (detail?: string, status: "done" | "skipped" = "done") => progress({ stage, status, detail });

	// --- 1. Vault reader ---------------------------------------------------
	begin(0);
	const cache = await storage.loadCache();
	const paths = io.listNotes();
	const titles = new Set(paths);
	const contents = new Map<string, string>();
	await mapPool(paths, 16, async (p) => contents.set(p, await io.read(p)));
	for (const p of paths) if (!contents.has(p)) titles.delete(p);

	const pending = new Set(cache.pending || []);
	const scope = opts.scopePath && titles.has(opts.scopePath) ? opts.scopePath : undefined;
	const fileHashes: Record<string, string> = {};
	const newNotes: string[] = [];
	for (const p of titles) {
		const h = coreHash(contents.get(p)!);
		const cached = cache.files[p]?.hash;
		const changed = cached !== h;
		if (scope && p !== scope) {
			// Out of scope: leave it exactly as the cache knows it, so a later
			// full run still sees (and pays for) any edit.
			if (cached) fileHashes[p] = cached;
			continue;
		}
		fileHashes[p] = h;
		if (changed) newNotes.push(p);
	}
	const reprocessed = new Set(newNotes);
	const valid = new Set([...titles].filter((t) => !reprocessed.has(t)));
	const resumed = [...valid].filter((t) => pending.has(t) && (!scope || t === scope));
	if (scope && !reprocessed.has(scope) && !resumed.includes(scope)) resumed.push(scope);
	const resumedSet = new Set(resumed);

	let concepts = (cache.concepts || []).filter((c) => valid.has(c.source_note!));
	// Only links FROM re-processed notes are regenerated; A→B stays valid when
	// B was edited as long as A is unchanged and B still exists.
	const retainedLinks = (cache.links || []).filter(
		(l) => valid.has(l.from_note) && !resumedSet.has(l.from_note) && titles.has(l.to_note),
	);
	const tagsByNote: Record<string, string[]> = {};
	for (const [k, v] of Object.entries(cache.tags || {})) if (valid.has(k)) tagsByNote[k] = v;
	log(`Found ${titles.size} notes. Needing processing: ${newNotes.length}${resumed.length ? `, relinking ${resumed.length}` : ""}.`);
	end(`${titles.size} notes, ${newNotes.length} changed`);
	checkCancel();

	const failed = new Set<string>();
	const noteText = (p: string, limit: number) => prepareNoteText(contents.get(p)!, limit);

	// --- 2. Concept extractor ----------------------------------------------
	begin(1);
	const rawConcepts = new Map<string, Concept[]>();
	const rawTags = new Map<string, string[]>();
	if (newNotes.length) {
		progress({ stage: 1, status: "running", done: 0, total: newNotes.length });
		const results = await mapPool(
			newNotes,
			concurrency,
			(p) => {
				checkCancel();
				return llm.invoke("extraction", CONCEPT_EXTRACTION, CONCEPT_SCHEMA, (limit) => ({ text: noteText(p, limit) }), settings.maxNoteChars);
			},
			(done) => progress({ stage: 1, status: "running", done, total: newNotes.length }),
		);
		checkCancel();
		results.forEach((r, i) => {
			const p = newNotes[i];
			if (r instanceof Error) {
				log(`Failed to extract concepts from '${p}': ${r.message.slice(0, 200)}`);
				failed.add(p);
				return;
			}
			rawConcepts.set(p, r.items.map((c) => ({ ...c, source_note: p })));
			if (r.tags.length) rawTags.set(p, r.tags);
		});
		const total = [...rawConcepts.values()].reduce((n, c) => n + c.length, 0);
		end(`${total} raw concepts`);
	} else end("nothing new", "skipped");

	// --- 3. Concept verifier (+ embedding) -----------------------------------
	begin(2);
	const verified: Concept[] = [];
	const toVerify = newNotes.filter((p) => (rawConcepts.get(p) || []).length);
	if (toVerify.length) {
		progress({ stage: 2, status: "running", done: 0, total: toVerify.length });
		const results = await mapPool(
			toVerify,
			concurrency,
			(p) => {
				checkCancel();
				return llm.invoke(
					"verification",
					CONCEPT_VERIFICATION,
					CONCEPT_SCHEMA,
					(limit) => ({
						text: noteText(p, limit),
						concepts: JSON.stringify(minifyConcepts(rawConcepts.get(p)!)),
						tags: JSON.stringify(rawTags.get(p) || []),
					}),
					settings.maxNoteChars,
				);
			},
			(done) => progress({ stage: 2, status: "running", done, total: toVerify.length }),
		);
		checkCancel();
		results.forEach((r, i) => {
			const p = toVerify[i];
			if (r instanceof Error) {
				log(`Failed to verify concepts from '${p}': ${r.message.slice(0, 200)}`);
				failed.add(p);
				return;
			}
			verified.push(...r.items.map((c) => ({ ...c, source_note: p })));
			if (r.tags.length) tagsByNote[p] = normaliseTags(r.tags);
		});
	}
	const resumedConcepts = concepts.filter((c) => resumedSet.has(c.source_note!));
	concepts = concepts.concat(verified);

	// Checkpoint the extraction spend before the later, failure-prone stages.
	if (!opts.dryRun && verified.length) {
		const ok = newNotes.filter((p) => !failed.has(p));
		const files = { ...cache.files };
		for (const p of ok) files[p] = { hash: fileHashes[p] };
		await storage.saveCache({
			...cache,
			files,
			concepts: (cache.concepts || []).filter((c) => !reprocessed.has(c.source_note!)).concat(verified),
			tags: { ...cache.tags, ...tagsByNote },
			pending: [...new Set([...(cache.pending || []), ...ok])].sort(),
		});
	}

	// Refresh the vector index: drop re-processed and deleted notes, embed new
	// concepts plus any cached concept whose vectors are missing (e.g. the
	// embedding model changed).
	const embedder = createEmbedder(deps.http, settings, usage, log);
	let index = await storage.loadIndex();
	let indexDirty = false;
	if (!index.matches(embedder.provider, embedder.model)) {
		if (index.entries.length) log(`Embedding model changed to ${embedder.provider}/${embedder.model}; re-embedding all concepts.`);
		index = new VectorIndex();
		index.manifest = { provider: embedder.provider, model: embedder.model, dimension: 0 };
		indexDirty = true;
	}
	const stale = new Set([...reprocessed, ...[...index.notes()].filter((n) => !titles.has(n))]);
	if (index.dropNotes(stale)) indexDirty = true;
	const indexed = index.notes();
	const missing = concepts.filter((c) => !indexed.has(c.source_note!) && !reprocessed.has(c.source_note!));
	const toEmbed = verified.concat(missing);
	const embedFailed = new Set<string>();
	if (toEmbed.length) {
		progress({ stage: 2, status: "running", detail: `embedding ${toEmbed.length} concepts` });
		try {
			const vectors = await embedder.embed(toEmbed.map((c) => `${c.concept_name}: ${c.explanation}`));
			toEmbed.forEach((c, i) => index.add(c.source_note!, c.concept_name, vectors[i]));
			indexDirty = true;
			log(`Embedded ${toEmbed.length} concepts; index holds ${index.entries.length} vectors.`);
		} catch (e) {
			log(`Failed to embed concepts: ${(e as Error).message}`);
			progress({ stage: 2, status: "error", detail: "embedding failed" });
			// Keep the (paid-for) concepts but relink these notes next run.
			for (const c of toEmbed) embedFailed.add(c.source_note!);
		}
	}
	end(`${verified.length} verified concepts`);
	checkCancel();

	// --- 4. Relationship extractor -----------------------------------------
	begin(3);
	let newConcepts = verified.concat(resumedConcepts).filter((c) => !embedFailed.has(c.source_note!));
	if (!newConcepts.length && !scope && concepts.length && !retainedLinks.length) {
		log("No new concepts, but cached concepts exist with 0 links. Re-linking all concepts...");
		newConcepts = concepts.filter((c) => !embedFailed.has(c.source_note!));
	}
	const pairSimilarity: Record<string, number> = {};
	const rawLinks: Relationship[] = [];
	if (newConcepts.length && index.entries.length) {
		const byNote = new Map<string, Concept[]>();
		for (const c of newConcepts) {
			if (!byNote.has(c.source_note!)) byNote.set(c.source_note!, []);
			byNote.get(c.source_note!)!.push(c);
		}
		const conceptIndex = new Map(concepts.map((c) => [`${c.concept_name}\x1f${c.source_note}`, c]));
		const inputs: { new_concepts: string; existing_concepts: string }[] = [];
		for (const [note, noteConcepts] of byNote) {
			const retrieved: Concept[] = [];
			const seen = new Set<string>();
			for (const c of noteConcepts) {
				const vector = index.get(note, c.concept_name);
				if (!vector) continue;
				for (const hit of index.search(vector, 10, note)) {
					const key = pairKey(note, hit.note);
					const sim = Math.max(0, hit.score);
					if (sim > (pairSimilarity[key] ?? 0)) pairSimilarity[key] = sim;
					const id = `${hit.name}\x1f${hit.note}`;
					if (seen.has(id)) continue;
					seen.add(id);
					const match = conceptIndex.get(id);
					if (match) retrieved.push(match);
				}
			}
			// Top 10 keeps the prompt compact and within free-tier TPM limits.
			if (!retrieved.length) continue;
			inputs.push({
				new_concepts: JSON.stringify(minifyConcepts(noteConcepts)),
				existing_concepts: JSON.stringify(minifyConcepts(retrieved.slice(0, 10))),
			});
		}
		progress({ stage: 3, status: "running", done: 0, total: inputs.length });
		const results = await mapPool(
			inputs,
			concurrency,
			(inp) => {
				checkCancel();
				return llm.invoke("relationship", RELATION, RELATIONSHIP_SCHEMA, () => inp, settings.maxNoteChars);
			},
			(done) => progress({ stage: 3, status: "running", done, total: inputs.length }),
		);
		checkCancel();
		for (const r of results) {
			if (r instanceof Error) log(`Relationship extraction call failed: ${r.message.slice(0, 200)}`);
			else rawLinks.push(...r.items);
		}
		end(`${rawLinks.length} candidate links`);
	} else end(index.entries.length ? "no new concepts" : "index empty", "skipped");

	// --- 5. Relationship verifier ------------------------------------------
	begin(4);
	let links = retainedLinks;
	if (rawLinks.length && concepts.length) {
		const lookup = new Map(concepts.map((c) => [c.concept_name, c]));
		const batches: { concepts: string; relationships: string }[] = [];
		for (let i = 0; i < rawLinks.length; i += 10) {
			const batch = rawLinks.slice(i, i + 10);
			const names = new Set(batch.flatMap((l) => [l.source, l.target]).filter(Boolean));
			const relevant = [...names].map((n) => lookup.get(n)).filter((c): c is Concept => !!c);
			batches.push({ concepts: JSON.stringify(minifyConcepts(relevant)), relationships: JSON.stringify(batch) });
		}
		progress({ stage: 4, status: "running", done: 0, total: batches.length });
		const results = await mapPool(
			batches,
			concurrency,
			(inp) => {
				checkCancel();
				return llm.invoke("verification", RELATION_VERIFY, RELATIONSHIP_SCHEMA, () => inp, settings.maxNoteChars);
			},
			(done) => progress({ stage: 4, status: "running", done, total: batches.length }),
		);
		checkCancel();
		const verifiedLinks: Relationship[] = [];
		for (const r of results) {
			if (r instanceof Error) log(`Relationship verification call failed: ${r.message.slice(0, 200)}`);
			else verifiedLinks.push(...r.items);
		}
		const { valid: validLinks, rejected } = filterAndScoreLinks(verifiedLinks, {
			titles,
			concepts,
			existingLinks: retainedLinks,
			pairSimilarity,
		});
		log(`Verified cross-note links: ${validLinks.length} (rejected ${rejected}).`);
		links = retainedLinks.concat(validLinks);
		end(`${validLinks.length} new links`);
	} else end("nothing to verify", "skipped");

	// --- 6. Link writer (planning half) ------------------------------------
	begin(5);
	// Links/tags of notes that failed this run are carried forward so a
	// transient API error never strips perfectly good links from disk.
	const present = new Set(links.map((l) => `${l.from_note}\x1f${l.to_note}`));
	for (const l of cache.links || []) {
		const k = `${l.from_note}\x1f${l.to_note}`;
		if (failed.has(l.from_note) && titles.has(l.to_note) && !present.has(k)) {
			links.push(l);
			present.add(k);
		}
	}
	for (const t of failed) if (cache.tags?.[t] && !tagsByNote[t]) tagsByNote[t] = cache.tags[t];

	const effective = buildEffectiveLinks(links, {
		existingTitles: titles,
		backlinks: settings.backlinks,
		maxLinks: settings.maxLinksPerNote,
	});
	const currentRendered: Record<string, string[]> = {};
	for (const [src, items] of Object.entries(effective)) currentRendered[src] = items.map((i) => i.target);
	const previousRendered = cache.rendered || {};
	const previousFm = cache.frontmatterTags || {};

	const processed = new Set([...newNotes, ...resumed].filter((t) => !failed.has(t)));
	const toUpdate = new Set(processed);
	for (const t of titles) if (!sameSet(currentRendered[t], previousRendered[t])) toUpdate.add(t);
	const rs = renderSettings(settings);
	if (cache.settings && JSON.stringify(cache.settings) !== JSON.stringify(rs)) {
		log("Rendering settings changed since the last run; refreshing all linked/tagged notes.");
		for (const t of [...Object.keys(currentRendered), ...Object.keys(tagsByNote), ...Object.keys(previousFm)]) toUpdate.add(t);
	}

	const changes: PlannedChange[] = [];
	const fmOursByNote: Record<string, string[]> = {};
	for (const [k, v] of Object.entries(previousFm)) if (titles.has(k)) fmOursByNote[k] = [...v];
	for (const p of [...toUpdate].sort()) {
		if (!titles.has(p) || failed.has(p)) continue;
		const before = contents.get(p)!;
		const noteTags = tagsByNote[p] || [];
		const targets = (effective[p] || []).map((i) => ({ ...i, target: io.linkText(i.target, p) }));
		const after = renderNote(stripAutoSections(before), settings.tagsInFrontmatter ? null : noteTags, targets, {
			maxLinks: settings.maxLinksPerNote,
			annotate: settings.annotateLinks,
		});
		const fm = planFrontmatter(io.frontmatterTags(p), previousFm[p] || [], settings.tagsInFrontmatter ? noteTags : []);
		const bodyChanged = after.replace(/\s+$/, "") !== before.replace(/\s+$/, "");
		if (!bodyChanged && !fm.add.length && !fm.remove.length) {
			if (fm.next.length) fmOursByNote[p] = fm.next;
			continue;
		}
		const parts = [];
		if (targets.length) parts.push(`${targets.length} link${targets.length === 1 ? "" : "s"}`);
		if (noteTags.length) parts.push(`${noteTags.length} tag${noteTags.length === 1 ? "" : "s"}`);
		changes.push({
			path: p,
			before,
			after: bodyChanged ? after : before,
			fmAdd: fm.add,
			fmRemove: fm.remove,
			fmOurs: fm.next,
			summary: parts.join(" and ") || "removed stale links",
		});
		if (fm.next.length) fmOursByNote[p] = fm.next;
		else delete fmOursByNote[p];
	}
	log(changes.length ? `${changes.length} note(s) ${opts.dryRun ? "would change" : "to write"}.` : "No notes need updating.");
	end(`${changes.length} note(s) to update`);

	const notDone = new Set([...failed]);
	const files: Record<string, { hash: string }> = {};
	for (const [t, h] of Object.entries(fileHashes)) if (!notDone.has(t)) files[t] = { hash: h };
	const nextPending = [...new Set([...[...pending].filter((t) => titles.has(t) && !processed.has(t)), ...embedFailed])]
		.filter((t) => !failed.has(t))
		.sort();
	const renderedOut: Record<string, string[]> = { ...currentRendered };
	for (const t of failed) {
		if (previousRendered[t]?.length) renderedOut[t] = [...previousRendered[t]];
		else delete renderedOut[t];
	}

	return {
		dryRun: opts.dryRun,
		scope: scope ? "note" : "vault",
		changes,
		nextCache: {
			files,
			concepts,
			links,
			tags: tagsByNote,
			rendered: renderedOut,
			frontmatterTags: fmOursByNote,
			settings: rs,
			pending: nextPending,
			lastReport: cache.lastReport,
		},
		previousRendered,
		index,
		indexDirty,
		failed: [...failed].sort(),
		processed: [...processed].sort(),
		notesTotal: titles.size,
		previousLinks: cache.links || [],
		usage,
		startedAt,
	};
}

/** Snapshots, writes the planned changes and persists cache + index. */
export async function commit(deps: PipelineDeps, run: RunPlan, previousCache: Cache): Promise<RunReport> {
	const { io, storage, log, progress, settings } = deps;
	const cache = run.nextCache;
	let backupId: string | null = null;
	const skipped: string[] = [];
	const writeFailed: string[] = [];
	let written = 0;

	if (run.changes.length) {
		const backup: Backup = {
			id: backupStamp(),
			createdAt: new Date().toISOString(),
			files: Object.fromEntries(run.changes.map((c) => [c.path, c.before])),
			cache: previousCache,
			processed: run.processed,
			label: run.scope === "note" ? "Link current note" : "Link vault",
		};
		try {
			await storage.saveBackup(backup, settings.maxBackups);
			backupId = backup.id;
			log(`Backed up ${run.changes.length} note(s) (${backup.id}).`);
		} catch (e) {
			log(`Could not create a backup (${(e as Error).message}); not writing any notes.`);
			writeFailed.push(...run.changes.map((c) => c.path));
		}
	}

	if (backupId) {
		progress({ stage: 5, status: "running", done: 0, total: run.changes.length });
		for (const change of run.changes) {
			try {
				// vault.process() under the hood: if the user saved the note since
				// we read it, writing our stale copy would discard their edit.
				const ok =
					change.after === change.before ? (await io.read(change.path)) === change.before
						: await io.writeIfUnchanged(change.path, change.before, change.after);
				if (!ok) {
					log(`Skipped ${change.path}: it was edited while Constellation was running. It will be re-processed next run.`);
					skipped.push(change.path);
					continue;
				}
				if (change.fmAdd.length || change.fmRemove.length) {
					await io.updateFrontmatterTags(change.path, change.fmAdd, change.fmRemove);
				}
				written++;
				log(`Wrote ${change.summary} to ${change.path}`);
			} catch (e) {
				writeFailed.push(change.path);
				log(`Failed to write ${change.path}: ${(e as Error).message}`);
			}
			progress({ stage: 5, status: "running", done: written + skipped.length + writeFailed.length, total: run.changes.length });
		}
	}
	progress({ stage: 5, status: "done", detail: `${written} written` });

	// --- 7. Summary reporter -------------------------------------------------
	progress({ stage: 6, status: "running" });
	// Record what is actually on disk: notes we could not write keep their old
	// state so the next run sees the difference and retries.
	for (const t of [...skipped, ...writeFailed]) {
		const prev = run.previousRendered[t];
		if (prev?.length) cache.rendered[t] = [...prev];
		else delete cache.rendered[t];
		const prevFm = previousCache.frontmatterTags?.[t];
		if (prevFm?.length) cache.frontmatterTags[t] = prevFm;
		else delete cache.frontmatterTags[t];
		if (skipped.includes(t)) delete cache.files[t];
	}
	for (const k of Object.keys(cache.rendered)) if (!cache.rendered[k].length) delete cache.rendered[k];

	const prevPairs = new Set(run.previousLinks.map((l) => `${l.from_note}\x1f${l.to_note}`));
	const curPairs = new Set(cache.links.map((l) => `${l.from_note}\x1f${l.to_note}`));
	const report: RunReport = {
		startedAt: new Date(run.startedAt).toISOString(),
		durationS: Math.round((Date.now() - run.startedAt) / 10) / 100,
		dryRun: false,
		scope: run.scope,
		notesTotal: run.notesTotal,
		notesProcessed: run.processed.length,
		notesFailed: run.failed,
		notesWritten: written,
		linksTotal: curPairs.size,
		linksNew: [...curPairs].filter((p) => !prevPairs.has(p)).length,
		linksRemoved: [...prevPairs].filter((p) => !curPairs.has(p)).length,
		...usageReport(run),
		skippedDueToEdit: skipped,
		backupId,
	};
	cache.lastReport = report;
	await storage.saveCache(cache);
	if (run.indexDirty) await storage.saveIndex(run.index);
	log(`Done. ${report.linksTotal} links total (+${report.linksNew} / -${report.linksRemoved}), ${written} note(s) written.`);
	progress({ stage: 6, status: "done", detail: `${written} written` });
	return report;
}

function usageReport(run: RunPlan) {
	return {
		llmCalls: run.usage.llmCalls,
		inputTokens: run.usage.inputTokens,
		outputTokens: run.usage.outputTokens,
		embeddingCalls: run.usage.embeddingCalls,
	};
}

/** Report for a preview that was not applied. */
export function previewReport(run: RunPlan): RunReport {
	return {
		startedAt: new Date(run.startedAt).toISOString(),
		durationS: Math.round((Date.now() - run.startedAt) / 10) / 100,
		dryRun: true,
		scope: run.scope,
		notesTotal: run.notesTotal,
		notesProcessed: run.processed.length,
		notesFailed: run.failed,
		notesWritten: 0,
		linksTotal: new Set(run.nextCache.links.map((l) => `${l.from_note}\x1f${l.to_note}`)).size,
		linksNew: 0,
		linksRemoved: 0,
		...usageReport(run),
		skippedDueToEdit: [],
		backupId: null,
	};
}

function backupStamp(): string {
	const d = new Date();
	const pad = (n: number, w = 2) => String(n).padStart(w, "0");
	return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}-${pad(d.getMilliseconds(), 3)}`;
}

/** Restores the newest snapshot (notes + cache). Returns how many notes were restored. */
export async function undoLastRun(deps: PipelineDeps): Promise<{ restored: number; backup: Backup | null }> {
	const { io, storage, log } = deps;
	const backup = await storage.latestBackup();
	if (!backup) return { restored: 0, backup: null };
	let restored = 0;
	for (const [path, content] of Object.entries(backup.files)) {
		try {
			await io.write(path, content);
			restored++;
		} catch (e) {
			log(`Could not restore ${path}: ${(e as Error).message}`);
		}
	}
	await storage.saveCache(backup.cache ?? emptyCache());
	// Vectors of notes that run re-extracted no longer match the restored cache.
	const index = await storage.loadIndex();
	if (index.dropNotes(new Set(backup.processed || []))) await storage.saveIndex(index);
	await storage.deleteBackup(backup.id);
	log(`Restored ${restored} note(s) from ${backup.id}.`);
	return { restored, backup };
}

/**
 * Removes everything Constellation wrote: generated sections and the
 * frontmatter tags it added (user tags stay). Snapshotted first, so undoable.
 * With dryRun nothing is touched and the count is what WOULD change.
 */
export async function stripVault(deps: PipelineDeps, dryRun: boolean): Promise<{ cleaned: number; backupId: string | null }> {
	const { io, storage, log, settings } = deps;
	const cache = await storage.loadCache();
	const changes: { path: string; before: string; after: string; fmRemove: string[] }[] = [];
	for (const path of io.listNotes()) {
		const before = await io.read(path);
		const stripped = stripAutoSections(before);
		const after = stripped ? stripped + "\n" : "";
		const fm = planFrontmatter(io.frontmatterTags(path), cache.frontmatterTags?.[path] || [], []);
		if (after.replace(/\s+$/, "") === before.replace(/\s+$/, "") && !fm.remove.length) continue;
		changes.push({ path, before, after, fmRemove: fm.remove });
	}
	if (dryRun) return { cleaned: changes.length, backupId: null };

	let backupId: string | null = null;
	if (changes.length) {
		const backup: Backup = {
			id: backupStamp(),
			createdAt: new Date().toISOString(),
			files: Object.fromEntries(changes.map((c) => [c.path, c.before])),
			cache,
			processed: [],
			label: "Remove all output",
		};
		await storage.saveBackup(backup, settings.maxBackups);
		backupId = backup.id;
	}
	let cleaned = 0;
	for (const c of changes) {
		try {
			const ok = c.after === c.before || (await io.writeIfUnchanged(c.path, c.before, c.after));
			if (!ok) {
				log(`Skipped ${c.path}: edited during the cleanup.`);
				continue;
			}
			if (c.fmRemove.length) await io.updateFrontmatterTags(c.path, [], c.fmRemove);
			cleaned++;
		} catch (e) {
			log(`Failed to clean ${c.path}: ${(e as Error).message}`);
		}
	}
	await storage.saveCache(emptyCache());
	await storage.deleteIndex();
	log(`Removed Constellation output from ${cleaned} note(s); cache and index reset.`);
	return { cleaned, backupId };
}
