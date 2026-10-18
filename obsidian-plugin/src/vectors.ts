// In-memory replacement for the vault-local FAISS index
// (constellation/core/vectorstore.py). Vectors are L2-normalised on insert so
// cosine similarity is a dot product; persisted as JSON with base64 Float32.

export interface VectorEntry {
	note: string;
	name: string;
	vector: Float32Array;
}

export interface IndexManifest {
	provider: string;
	model: string;
	dimension: number;
}

export interface SerializedIndex {
	manifest: IndexManifest | null;
	entries: { note: string; name: string; v: string }[];
}

export interface SearchHit {
	note: string;
	name: string;
	score: number;
}

export function normalise(v: ArrayLike<number>): Float32Array {
	const out = new Float32Array(v.length);
	let norm = 0;
	for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
	norm = Math.sqrt(norm) || 1;
	for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
	return out;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
	if (a.length !== b.length) throw new Error(`vector dimension mismatch (${a.length} vs ${b.length})`);
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	if (!na || !nb) return 0;
	return dot / Math.sqrt(na * nb);
}

/** Indices of the k largest scores, best first (ties keep insertion order). */
export function topK(scores: ArrayLike<number>, k: number): number[] {
	const idx: number[] = [];
	for (let i = 0; i < scores.length; i++) idx.push(i);
	idx.sort((a, b) => scores[b] - scores[a] || a - b);
	return idx.slice(0, Math.max(0, k));
}

export class VectorIndex {
	entries: VectorEntry[] = [];
	manifest: IndexManifest | null = null;

	get dimension(): number {
		return this.entries[0]?.vector.length ?? this.manifest?.dimension ?? 0;
	}

	notes(): Set<string> {
		return new Set(this.entries.map((e) => e.note));
	}

	add(note: string, name: string, vector: ArrayLike<number>): void {
		if (this.entries.length && vector.length !== this.dimension) {
			throw new Error(`vector dimension mismatch (${vector.length} vs ${this.dimension})`);
		}
		this.entries.push({ note, name, vector: normalise(vector) });
	}

	/** Drops every vector of the given notes; returns how many were removed. */
	dropNotes(notes: Set<string>): number {
		const before = this.entries.length;
		this.entries = this.entries.filter((e) => !notes.has(e.note));
		return before - this.entries.length;
	}

	/** Top-k nearest concepts to `query`, optionally skipping one note. */
	search(query: ArrayLike<number>, k: number, excludeNote?: string): SearchHit[] {
		const q = normalise(query);
		const pool = excludeNote === undefined ? this.entries : this.entries.filter((e) => e.note !== excludeNote);
		const scores = new Float32Array(pool.length);
		for (let i = 0; i < pool.length; i++) {
			const v = pool[i].vector;
			if (v.length !== q.length) continue;
			let dot = 0;
			for (let j = 0; j < v.length; j++) dot += v[j] * q[j];
			scores[i] = dot;
		}
		return topK(scores, k).map((i) => ({ note: pool[i].note, name: pool[i].name, score: scores[i] }));
	}

	/** Vector stored for a concept, if any. */
	get(note: string, name: string): Float32Array | undefined {
		return this.entries.find((e) => e.note === note && e.name === name)?.vector;
	}

	toJSON(): SerializedIndex {
		return {
			manifest: this.manifest ? { ...this.manifest, dimension: this.dimension } : null,
			entries: this.entries.map((e) => ({ note: e.note, name: e.name, v: encodeVector(e.vector) })),
		};
	}

	static fromJSON(data: SerializedIndex | null | undefined): VectorIndex {
		const index = new VectorIndex();
		if (!data || !Array.isArray(data.entries)) return index;
		index.manifest = data.manifest ?? null;
		for (const e of data.entries) {
			const vector = decodeVector(e.v);
			if (index.entries.length && vector.length !== index.dimension) continue;
			index.entries.push({ note: e.note, name: e.name, vector });
		}
		if (index.manifest?.dimension && index.entries.length && index.dimension !== index.manifest.dimension) {
			// Corrupt or hand-edited: let the caller rebuild.
			return new VectorIndex();
		}
		return index;
	}

	/** True when the stored vectors came from this provider/model. */
	matches(provider: string, model: string): boolean {
		return !!this.manifest && this.manifest.provider === provider && this.manifest.model === model;
	}
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

export function encodeVector(v: Float32Array): string {
	const bytes = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
	let out = "";
	for (let i = 0; i < bytes.length; i += 3) {
		const n = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
		out += B64[(n >> 18) & 63] + B64[(n >> 12) & 63];
		out += i + 1 < bytes.length ? B64[(n >> 6) & 63] : "=";
		out += i + 2 < bytes.length ? B64[n & 63] : "=";
	}
	return out;
}

export function decodeVector(s: string): Float32Array {
	const clean = s.replace(/=+$/, "");
	const bytes = new Uint8Array(Math.floor((clean.length * 3) / 4));
	let p = 0;
	for (let i = 0; i < clean.length; i += 4) {
		const n =
			(B64.indexOf(clean[i]) << 18) |
			(B64.indexOf(clean[i + 1]) << 12) |
			((i + 2 < clean.length ? B64.indexOf(clean[i + 2]) : 0) << 6) |
			(i + 3 < clean.length ? B64.indexOf(clean[i + 3]) : 0);
		if (p < bytes.length) bytes[p++] = (n >> 16) & 255;
		if (p < bytes.length) bytes[p++] = (n >> 8) & 255;
		if (p < bytes.length) bytes[p++] = n & 255;
	}
	return new Float32Array(bytes.buffer, 0, Math.floor(bytes.length / 4));
}
