// Tolerant parsing of LLM JSON output -- port of the repair ideas in
// constellation/core/utils.py, plus a small hand-written schema check in place
// of the pydantic models in constellation/core/state.py.

import type { Concept, Relationship } from "./links";

export class LLMResponseError extends Error {
	constructor(message: string) {
		// Same marker as the Python code so retry logic classifies it as a parse error.
		super(`Model response unusable: ${message}`);
		this.name = "LLMResponseError";
	}
}

/** Pull the first JSON object or array out of an LLM response string. */
export function extractJsonText(raw: string): string {
	const fenced = /```(?:json)?\s*\n?([\s\S]*?)```/.exec(raw);
	if (fenced) return fenced[1].trim();
	// Start at whichever bracket comes first, so a bare top-level array is not
	// mistaken for its first element.
	const pairs = [["{", "}"], ["[", "]"]]
		.map(([open, close]) => ({ open, close, start: raw.indexOf(open) }))
		.filter((p) => p.start !== -1)
		.sort((a, b) => a.start - b.start);
	for (const { open, close, start } of pairs) {
		let depth = 0;
		for (let i = start; i < raw.length; i++) {
			if (raw[i] === open) depth++;
			else if (raw[i] === close) {
				depth--;
				if (depth === 0) return raw.slice(start, i + 1);
			}
		}
		// Unbalanced (truncated) -- hand back the tail so the repair pass can close it.
		return raw.slice(start).trim();
	}
	return raw.trim();
}

function tryParse(s: string): { ok: true; value: unknown } | { ok: false; error: Error } {
	try {
		return { ok: true, value: JSON.parse(s) };
	} catch (e) {
		return { ok: false, error: e as Error };
	}
}

/** Unclosed bracket stack for s, or null if s ends inside a string literal. */
function bracketStack(s: string): string[] | null {
	const stack: string[] = [];
	let inString = false;
	let escape = false;
	for (const ch of s) {
		if (escape) {
			escape = false;
			continue;
		}
		if (inString) {
			if (ch === "\\") escape = true;
			else if (ch === '"') inString = false;
			continue;
		}
		if (ch === '"') inString = true;
		else if (ch === "[" || ch === "{") stack.push(ch);
		else if ((ch === "]" || ch === "}") && stack.length) stack.pop();
	}
	return inString || escape ? null : stack;
}

function close(s: string, stack: string[]): string {
	return s + [...stack].reverse().map((c) => (c === "[" ? "]" : "}")).join("");
}

/** Cut back to the last complete element or key/value pair, then close up. */
function dropIncompleteTail(s: string): string {
	const cuts: number[] = [];
	let inString = false;
	let escape = false;
	for (let i = 0; i < s.length; i++) {
		const ch = s[i];
		if (escape) {
			escape = false;
			continue;
		}
		if (inString) {
			if (ch === "\\") escape = true;
			else if (ch === '"') inString = false;
			continue;
		}
		if (ch === '"') inString = true;
		else if (ch === "}" || ch === "]" || ch === ",") cuts.push(i);
	}
	for (let k = cuts.length - 1; k >= 0; k--) {
		const idx = cuts[k];
		// Keep a closing bracket, drop a trailing comma.
		const candidate = s[idx] === "," ? s.slice(0, idx) : s.slice(0, idx + 1);
		const stack = bracketStack(candidate);
		if (stack === null) continue;
		const closed = close(candidate, stack);
		if (tryParse(closed).ok) return closed;
	}
	return s;
}

/** Best-effort repair of JSON truncated by a max_tokens cutoff. */
export function repairTruncatedJson(input: string): string {
	const s = input.replace(/\s+$/, "");
	if (!s || tryParse(s).ok) return s;

	const stack: string[] = [];
	let inString = false;
	let escape = false;
	for (const ch of s) {
		if (escape) {
			escape = false;
			continue;
		}
		if (inString) {
			if (ch === "\\") escape = true;
			else if (ch === '"') inString = false;
			continue;
		}
		if (ch === '"') inString = true;
		else if (ch === "[" || ch === "{") stack.push(ch);
		else if ((ch === "]" || ch === "}") && stack.length) stack.pop();
	}

	let repaired = escape ? s.slice(0, -1) : s;
	if (inString) repaired += '"';
	repaired = repaired.replace(/,\s*$/, "");
	// A dangling "key": with no value cannot be closed -- drop it.
	repaired = repaired.replace(/,?\s*"[^"]*"\s*:\s*$/, "");
	repaired = close(repaired, stack);
	return tryParse(repaired).ok ? repaired : dropIncompleteTail(s);
}

// Field-name variants smaller models produce. Only applied when the key is not
// already a real field of the schema ('source' is genuine on relationships).
const FIELD_ALIASES: Record<string, string> = {
	concept: "concept_name",
	name: "concept_name",
	title: "concept_name",
	keywords: "important_keywords",
	key_words: "important_keywords",
	score: "importance_score",
	importance: "importance_score",
	related: "related_concepts",
	source: "source_note",
	description: "explanation",
	type: "relationship",
	relation: "relationship",
};

export function normaliseFieldNames(data: unknown, validFields: Set<string>): unknown {
	if (Array.isArray(data)) return data.map((d) => normaliseFieldNames(d, validFields));
	if (data && typeof data === "object") {
		const obj = data as Record<string, unknown>;
		const out: Record<string, unknown> = {};
		for (const [key, value] of Object.entries(obj)) {
			let canonical = key;
			if (!validFields.has(key)) {
				const alias = FIELD_ALIASES[key];
				if (alias && validFields.has(alias) && !(alias in obj)) canonical = alias;
			}
			out[canonical] = normaliseFieldNames(value, validFields);
		}
		return out;
	}
	return data;
}

// --- Schemas ---------------------------------------------------------------

export interface ItemSchema<T> {
	name: string;
	listKey: string;
	fields: string[];
	/** Returns the coerced item, or null when it is unusable. */
	coerce(item: unknown): T | null;
}

export interface ParsedOutput<T> {
	items: T[];
	tags: string[];
	dropped: number;
}

const str = (v: unknown): string | null =>
	typeof v === "string" ? v : typeof v === "number" ? String(v) : null;
const strList = (v: unknown): string[] =>
	Array.isArray(v) ? v.map(str).filter((s): s is string => !!s) : typeof v === "string" && v ? [v] : [];

export const CONCEPT_SCHEMA: ItemSchema<Concept> = {
	name: "ConceptExtractionOutput",
	listKey: "concepts",
	fields: [
		"concepts", "tags", "concept_name", "category", "explanation",
		"important_keywords", "related_concepts", "importance_score", "source_note",
	],
	coerce(item) {
		if (!item || typeof item !== "object") return null;
		const o = item as Record<string, unknown>;
		const name = str(o.concept_name)?.trim();
		const explanation = str(o.explanation)?.trim();
		if (!name || explanation === undefined || explanation === null) return null;
		const score = Number(o.importance_score);
		return {
			concept_name: name,
			category: str(o.category) ?? "",
			explanation,
			important_keywords: strList(o.important_keywords),
			related_concepts: strList(o.related_concepts),
			importance_score: Number.isFinite(score) ? Math.round(score) : 5,
		};
	},
};

export const RELATIONSHIP_SCHEMA: ItemSchema<Relationship> = {
	name: "RelationshipExtractionOutput",
	listKey: "links",
	fields: ["links", "source", "target", "relationship", "evidence", "from_note", "to_note"],
	coerce(item) {
		if (!item || typeof item !== "object") return null;
		const o = item as Record<string, unknown>;
		const source = str(o.source);
		const target = str(o.target);
		if (!source || !target) return null;
		return {
			source,
			target,
			relationship: str(o.relationship) ?? "",
			evidence: str(o.evidence) ?? "",
			from_note: str(o.from_note) ?? "",
			to_note: str(o.to_note) ?? "",
		};
	},
};

/** JSON Schema text appended to every prompt (Groq JSON mode needs the word "json"). */
export function schemaHint(schema: ItemSchema<unknown>): string {
	const shape =
		schema === (CONCEPT_SCHEMA as ItemSchema<unknown>)
			? {
				title: "ConceptExtractionOutput",
				type: "object",
				required: ["concepts"],
				properties: {
					concepts: {
						type: "array",
						items: {
							type: "object",
							required: [
								"concept_name", "category", "explanation",
								"important_keywords", "related_concepts", "importance_score",
							],
							properties: {
								concept_name: { type: "string", description: "The name of the concept." },
								category: { type: "string", description: "The broad category this concept belongs to." },
								explanation: { type: "string", description: "A concise explanation of the concept based on the text." },
								important_keywords: { type: "array", items: { type: "string" } },
								related_concepts: { type: "array", items: { type: "string" } },
								importance_score: { type: "integer", description: "An importance score from 1 to 10." },
							},
						},
					},
					tags: {
						type: "array",
						items: { type: "string" },
						description: "3-5 global Obsidian-style tags representing the entire note (e.g., '#machine-learning', '#python').",
					},
				},
			}
			: {
				title: "RelationshipExtractionOutput",
				type: "object",
				required: ["links"],
				properties: {
					links: {
						type: "array",
						items: {
							type: "object",
							required: ["source", "target", "relationship", "evidence", "from_note", "to_note"],
							properties: {
								source: { type: "string", description: "The name of the source concept." },
								target: { type: "string", description: "The name of the target concept from a DIFFERENT note." },
								relationship: { type: "string", description: "uses, depends_on, extends, similar_to, part_of, implemented_with, alternative_to, improves, causes" },
								evidence: { type: "string", description: "Brief explanation of why these concepts are related." },
								from_note: { type: "string", description: "The source_note of the source concept." },
								to_note: { type: "string", description: "The source_note of the target concept." },
							},
						},
					},
				},
			};
	return `You MUST respond with valid JSON matching this exact schema:\n${JSON.stringify(shape, null, 2)}`;
}

/**
 * Parses raw LLM text into the schema, tolerating fences, truncation, field
 * aliases, a bare top-level array, and individually-invalid items (dropped
 * rather than losing the whole response).
 */
export function parseLLMOutput<T>(raw: string, schema: ItemSchema<T>): ParsedOutput<T> {
	const jsonText = extractJsonText(raw || "");
	const candidates = [jsonText];
	const repaired = repairTruncatedJson(jsonText);
	if (repaired !== jsonText) candidates.push(repaired);

	const valid = new Set(schema.fields);
	let lastError = "no JSON found";
	for (const candidate of candidates) {
		const parsed = tryParse(candidate);
		if (!parsed.ok) {
			lastError = parsed.error.message;
			continue;
		}
		let data = normaliseFieldNames(parsed.value, valid);
		if (Array.isArray(data)) data = { [schema.listKey]: data };
		if (!data || typeof data !== "object") {
			lastError = "top-level value is not an object";
			continue;
		}
		const obj = data as Record<string, unknown>;
		const list = obj[schema.listKey];
		if (!Array.isArray(list)) {
			lastError = `missing '${schema.listKey}' array`;
			continue;
		}
		const items: T[] = [];
		let dropped = 0;
		for (const item of list) {
			const coerced = schema.coerce(item);
			if (coerced) items.push(coerced);
			else dropped++;
		}
		return { items, tags: strList(obj.tags), dropped };
	}
	throw new LLMResponseError(
		`could not parse as ${schema.name} (${lastError}). Raw output began: ${JSON.stringify((raw || "").slice(0, 300))}`,
	);
}
