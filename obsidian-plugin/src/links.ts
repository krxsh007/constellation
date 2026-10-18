// Port of link scoring / filtering from constellation/nodes/relations.py and
// build_effective_links from constellation/nodes/io.py. Pure logic.

import { cmp, LinkTarget, linkScore, rankAndCap } from "./sections";

export interface Concept {
	concept_name: string;
	category: string;
	explanation: string;
	important_keywords: string[];
	related_concepts: string[];
	importance_score: number;
	source_note?: string;
}

export interface Relationship {
	source: string;
	target: string;
	relationship: string;
	evidence: string;
	from_note: string;
	to_note: string;
	score?: number;
	support?: number;
}

// Each additional verified concept pair between the same two notes adds this
// much on top of their best similarity: two notes linked by three independent
// concept relations are a stronger link than two linked by one.
export const SUPPORT_BONUS = 0.05;

/** Order-independent key for a note pair. */
export function pairKey(a: string, b: string): string {
	const [x, y] = [String(a), String(b)].sort(cmp);
	return `${x}\x1f${y}`;
}

/**
 * Resolves a model-supplied note name to a real vault path, tolerating a
 * missing '.md' suffix, wikilink brackets, quotes, case differences and a bare
 * basename when it is unambiguous.
 */
export function makeTitleResolver(titles: Iterable<string>): (note: string | undefined) => string | undefined {
	const map = new Map<string, string>();
	const basenames = new Map<string, string | null>();
	for (const t of titles) {
		map.set(t.toLowerCase(), t);
		const base = t.split("/").pop()!.toLowerCase();
		basenames.set(base, basenames.has(base) ? null : t);
	}
	return (note) => {
		if (!note) return note;
		let cleaned = String(note).trim().replace(/^['"]+|['"]+$/g, "");
		if (cleaned.startsWith("[[") && cleaned.endsWith("]]")) cleaned = cleaned.slice(2, -2).trim();
		cleaned = cleaned.split("|")[0].trim();
		const candidate = (cleaned.toLowerCase().endsWith(".md") ? cleaned : `${cleaned}.md`).toLowerCase();
		return map.get(candidate) ?? basenames.get(candidate) ?? note;
	};
}

/**
 * Structural filtering (no self-links, no duplicates, no links to notes that
 * do not exist) followed by scoring: best similarity between the two notes
 * plus a small bonus per extra supporting concept pair.
 */
export function filterAndScoreLinks(
	verified: Relationship[],
	ctx: {
		titles: Set<string>;
		concepts: Concept[];
		existingLinks: Relationship[];
		pairSimilarity: Record<string, number>;
	},
): { valid: Relationship[]; rejected: number } {
	const resolve = makeTitleResolver(ctx.titles);
	const conceptToNote = new Map<string, string>();
	for (const c of ctx.concepts) if (c.source_note) conceptToNote.set(c.concept_name, c.source_note);
	const existing = new Set(ctx.existingLinks.map((l) => `${l.from_note}\x1f${l.to_note}`));
	const seen = new Set<string>();
	const support = new Map<string, Set<string>>();
	const valid: Relationship[] = [];
	let rejected = 0;

	for (const link of verified) {
		const fromNote = resolve(link.from_note || conceptToNote.get(link.source));
		const toNote = resolve(link.to_note || conceptToNote.get(link.target));
		if (!fromNote || !toNote || !ctx.titles.has(fromNote) || !ctx.titles.has(toNote) || fromNote === toNote) {
			rejected++;
			continue;
		}
		// Every distinct concept pair backing this note pair counts as support,
		// even when the note pair itself is a duplicate we will not keep.
		const key = pairKey(fromNote, toNote);
		if (!support.has(key)) support.set(key, new Set());
		support.get(key)!.add([String(link.source), String(link.target)].sort(cmp).join("\x1f"));

		const directed = `${fromNote}\x1f${toNote}`;
		if (seen.has(directed) || existing.has(directed)) {
			rejected++;
			continue;
		}
		seen.add(directed);
		valid.push({ ...link, from_note: fromNote, to_note: toNote });
	}

	for (const link of valid) {
		const key = pairKey(link.from_note, link.to_note);
		const n = Math.max(support.get(key)?.size ?? 0, 1);
		const sim = ctx.pairSimilarity[key] ?? 0;
		link.score = Math.round((sim + SUPPORT_BONUS * (n - 1)) * 1e4) / 1e4;
		link.support = n;
	}
	return { valid, rejected };
}

/**
 * Returns {source: [{target, evidence, score}]} -- exactly the links that
 * would be rendered into each note (dedupe, optional backlinks, ranking, cap).
 */
export function buildEffectiveLinks(
	links: Relationship[],
	opts: { existingTitles?: Set<string>; backlinks: boolean; maxLinks: number },
): Record<string, LinkTarget[]> {
	const candidates = new Map<string, Map<string, LinkTarget>>();
	const add = (src: string, dst: string, evidence: string, score: number) => {
		if (!src || !dst || src === dst) return;
		if (opts.existingTitles && (!opts.existingTitles.has(src) || !opts.existingTitles.has(dst))) return;
		evidence = (evidence || "").trim().replace(/\n/g, " ");
		if (!candidates.has(src)) candidates.set(src, new Map());
		const bySrc = candidates.get(src)!;
		const current = bySrc.get(dst);
		if (!current) {
			bySrc.set(dst, { target: dst, evidence, score });
			return;
		}
		current.score = Math.max(current.score, score);
		if (!current.evidence && evidence) current.evidence = evidence;
	};
	for (const link of links) {
		const score = linkScore(link);
		add(link.from_note, link.to_note, link.evidence, score);
		if (opts.backlinks) add(link.to_note, link.from_note, link.evidence, score);
	}
	const out: Record<string, LinkTarget[]> = {};
	for (const [src, items] of candidates) {
		out[src] = rankAndCap([...items.values()], opts.maxLinks, (i) => i.target);
	}
	return out;
}

/** Strips bulky fields (keywords, related concepts) to cut prompt tokens. */
export function minifyConcepts(concepts: Concept[]) {
	return concepts.map((c) => ({
		concept_name: c.concept_name,
		source_note: c.source_note,
		explanation: truncate(c.explanation, 240),
	}));
}

function truncate(text: string, max: number): string {
	text = (text || "").trim();
	return text.length <= max ? text : text.slice(0, max).replace(/\s+$/, "") + "...";
}
