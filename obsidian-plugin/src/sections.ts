// Port of the section-writing rules in constellation/nodes/io.py.
// Pure string functions: no Obsidian imports, so they are unit-testable.

export interface LinkTarget {
	target: string;
	evidence: string;
	score: number;
}

const TAG_LINE = /^#[^\s#]+(?:\s+#[^\s#]+)*$/;
const LINK_LINE = /^-\s*\[\[[^\]]+\]\](?:\s*(?:[—–-]|:)\s*.*)?$/;

type Section = "tags" | "links" | null;

/** True only for lines this tool itself would have written. */
function isAutoLine(section: Section, stripped: string): boolean {
	if (section === "tags") return TAG_LINE.test(stripped);
	return LINK_LINE.test(stripped);
}

/**
 * Extracts the hand-written part of a note by removing the ## Tags and
 * ## Related Links sections.
 *
 * Only lines this tool would have generated (a run of hashtags, or `- [[link]]`
 * bullets) are dropped. The moment something else shows up -- prose, a heading,
 * a code fence -- the section is considered over and everything from there on is
 * kept, so text the user adds at the bottom of a note survives the next run.
 */
export function stripAutoSections(content: string): string {
	let inCodeBlock = false;
	const kept: string[] = [];
	let section: Section = null;

	// Closes an auto-section, leaving exactly one blank line where it was so
	// repeated runs don't accumulate whitespace.
	const endSection = () => {
		section = null;
		while (kept.length && !kept[kept.length - 1].trim()) kept.pop();
		if (kept.length) kept.push("");
	};

	for (const line of splitLines(content)) {
		const stripped = line.trim();
		const isFence = stripped.startsWith("```");
		if (isFence) inCodeBlock = !inCodeBlock;

		// Never interpret markdown structure inside a fenced code block.
		if (inCodeBlock || isFence) {
			if (section) endSection();
			kept.push(line);
			continue;
		}

		if (stripped === "## Tags") {
			section = "tags";
			continue;
		}
		if (stripped === "## Related Links") {
			section = "links";
			continue;
		}

		if (section) {
			if (!stripped) continue;
			if (isAutoLine(section, stripped)) continue;
			// Anything else is the user's own content -- stop stripping.
			endSection();
		}

		kept.push(line);
	}

	return kept.join("\n").replace(/\s+$/, "");
}

/** Python's str.splitlines() for the line endings notes actually contain. */
function splitLines(content: string): string[] {
	const lines = content.split(/\r\n|\r|\n/);
	if (lines.length && lines[lines.length - 1] === "") lines.pop();
	return lines;
}

export function linkScore(item: { score?: unknown }): number {
	const n = Number(item?.score ?? 0);
	return Number.isFinite(n) ? n : 0;
}

/**
 * Keeps the maxLinks highest-scoring items, then orders them alphabetically.
 * Ranking happens BEFORE the cap so the strongest links survive it; ties on
 * score are broken alphabetically so the result is deterministic.
 */
export function rankAndCap<T extends { score?: unknown }>(
	items: T[],
	maxLinks: number,
	key: (item: T) => string,
): T[] {
	let ranked = [...items].sort(
		(a, b) => linkScore(b) - linkScore(a) || cmp(key(a), key(b)),
	);
	if (maxLinks && maxLinks > 0) ranked = ranked.slice(0, maxLinks);
	return ranked.sort((a, b) => cmp(key(a), key(b)));
}

/** Python-style code-point string comparison (localeCompare would differ). */
export function cmp(a: string, b: string): number {
	return a < b ? -1 : a > b ? 1 : 0;
}

/** Rebuilds a note's text with the auto-generated Tags / Related Links sections. */
export function renderNote(
	baseContent: string,
	tags: string[] | null | undefined,
	targets: LinkTarget[] | null | undefined,
	opts: { maxLinks: number; annotate: boolean },
): string {
	let content = baseContent.replace(/\s+$/, "");
	let prefix = content ? "\n\n" : "";
	if (tags && tags.length) {
		content += prefix + "## Tags\n" + tags.join(" ");
		prefix = "\n\n";
	}
	if (targets && targets.length) {
		// Deduplicate targets, keeping the best score and any evidence.
		const seen = new Map<string, LinkTarget>();
		for (const t of targets) {
			if (!t.target) continue;
			const item = {
				target: t.target,
				evidence: (t.evidence || "").trim().replace(/\n/g, " "),
				score: linkScore(t),
			};
			const current = seen.get(item.target);
			if (!current) {
				seen.set(item.target, item);
				continue;
			}
			current.score = Math.max(current.score, item.score);
			if (!current.evidence && item.evidence) current.evidence = item.evidence;
		}
		const ranked = rankAndCap([...seen.values()], opts.maxLinks, (i) => i.target);
		const lines = ranked.map((i) =>
			opts.annotate && i.evidence ? `- [[${i.target}]] — ${i.evidence}` : `- [[${i.target}]]`,
		);
		if (lines.length) content += prefix + "## Related Links\n" + lines.join("\n");
	}
	return content ? content + "\n" : "";
}

/** Obsidian tags: '#' prefixed, no spaces, no duplicates. */
export function normaliseTags(tags: unknown[]): string[] {
	const cleaned: string[] = [];
	for (const raw of tags || []) {
		let tag = String(raw).trim().replace(/^#+/, "").trim();
		if (!tag) continue;
		// Remove leading/trailing punctuation LLMs often emit (commas, semicolons).
		tag = tag.replace(/^[^\p{L}\p{N}_/]+|[^\p{L}\p{N}_/]+$/gu, "");
		if (!tag) continue;
		tag = "#" + tag.split(/\s+/).join("-").toLowerCase();
		if (!cleaned.includes(tag)) cleaned.push(tag);
	}
	return cleaned;
}

/** Normalised form used to compare frontmatter tags. */
export function tagNorm(tag: unknown): string {
	return String(tag).trim().replace(/^['"]+|['"]+$/g, "").replace(/^#+/, "").trim().toLowerCase();
}

/**
 * Splits YAML frontmatter off a note: [frontmatter text or null, body].
 * The body is what gets hashed and sent to the LLM, so Obsidian rewriting the
 * frontmatter (processFrontMatter reformats YAML) never looks like an edit.
 */
export function splitFrontmatter(content: string): [string | null, string] {
	if (!content.startsWith("---")) return [null, content];
	const lines = content.split("\n");
	if (lines[0].trim() !== "---") return [null, content];
	for (let i = 1; i < lines.length; i++) {
		const l = lines[i].trim();
		if (l === "---" || l === "...") {
			return [lines.slice(1, i).join("\n"), lines.slice(i + 1).join("\n")];
		}
	}
	return [null, content];
}

/**
 * Frontmatter-tag update applied inside processFrontMatter: drops the tags
 * Constellation added last time, then adds wanted tags the user lacks.
 * Returns the tags WE contributed so a later run removes only those.
 */
export function updateFrontmatterTags(
	fm: Record<string, unknown>,
	add: string[],
	remove: string[],
): { added: string[]; changed: boolean } {
	const key = "tags" in fm ? "tags" : "tag" in fm ? "tag" : "tags";
	const raw = fm[key];
	let values: string[] = [];
	if (Array.isArray(raw)) values = raw.map(String);
	else if (typeof raw === "string" && raw.trim()) values = raw.split(/[,\s]+/).filter(Boolean);

	const removeNorm = new Set((remove || []).map(tagNorm).filter(Boolean));
	const kept = values.filter((v) => !removeNorm.has(tagNorm(v)));
	let changed = kept.length !== values.length;
	const present = new Set(kept.map(tagNorm));
	const added: string[] = [];
	for (const t of add || []) {
		const clean = String(t).trim().replace(/^#+/, "").trim();
		if (!clean || present.has(tagNorm(clean))) continue;
		kept.push(clean);
		present.add(tagNorm(clean));
		added.push(clean);
		changed = true;
	}
	if (changed) {
		if (kept.length) fm[key] = kept;
		else delete fm[key];
	}
	return { added, changed };
}

/** Wikilinks reference the note without its .md suffix. */
export function wikilinkName(title: string): string {
	return title.toLowerCase().endsWith(".md") ? title.slice(0, -3) : title;
}

/** Text the LLM sees and the hash covers: body minus auto-sections, capped. */
export function prepareNoteText(content: string, maxChars: number): string {
	const text = stripAutoSections(splitFrontmatter(content)[1]).trim();
	if (text.length <= maxChars) return text;
	return text.slice(0, maxChars).replace(/\s+$/, "") + "\n\n[... note truncated for token limit ...]";
}

/** Hash of the hand-written body, ignoring auto-sections and frontmatter. */
export function coreHash(content: string): string {
	return hashString(stripAutoSections(splitFrontmatter(content)[1]));
}

/** cyrb53: fast, synchronous, collision-resistant enough for change detection. */
export function hashString(str: string, seed = 0): string {
	let h1 = 0xdeadbeef ^ seed;
	let h2 = 0x41c6ce57 ^ seed;
	for (let i = 0; i < str.length; i++) {
		const ch = str.charCodeAt(i);
		h1 = Math.imul(h1 ^ ch, 2654435761);
		h2 = Math.imul(h2 ^ ch, 1597334677);
	}
	h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
	h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
	return (h2 >>> 0).toString(16).padStart(8, "0") + (h1 >>> 0).toString(16).padStart(8, "0");
}
