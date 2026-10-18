import { describe, expect, it } from "vitest";
import { cosine, decodeVector, encodeVector, topK, VectorIndex } from "../src/vectors";

describe("cosine / topK", () => {
	it("computes cosine similarity", () => {
		expect(cosine([1, 0], [1, 0])).toBeCloseTo(1);
		expect(cosine([1, 0], [0, 1])).toBeCloseTo(0);
		expect(cosine([1, 1], [-1, -1])).toBeCloseTo(-1);
		expect(cosine([0, 0], [1, 1])).toBe(0);
		expect(() => cosine([1], [1, 2])).toThrow(/dimension/);
	});

	it("returns the k best indices, best first, stable on ties", () => {
		expect(topK([0.1, 0.9, 0.5, 0.9], 3)).toEqual([1, 3, 2]);
		expect(topK([1, 2], 10)).toEqual([1, 0]);
		expect(topK([1, 2], 0)).toEqual([]);
	});
});

describe("VectorIndex", () => {
	const build = () => {
		const idx = new VectorIndex();
		idx.manifest = { provider: "test", model: "m", dimension: 3 };
		idx.add("a.md", "Docker", [1, 0, 0]);
		idx.add("a.md", "Compose", [0.9, 0.1, 0]);
		idx.add("b.md", "Kubernetes", [0.8, 0.2, 0]);
		idx.add("c.md", "Baking", [0, 0, 1]);
		return idx;
	};

	it("searches by cosine, excluding the query's own note", () => {
		const hits = build().search([1, 0, 0], 2, "a.md");
		expect(hits.map((h) => h.name)).toEqual(["Kubernetes", "Baking"]);
		expect(hits[0].score).toBeGreaterThan(0.9);
	});

	it("drops notes and rejects mismatched dimensions", () => {
		const idx = build();
		expect(idx.dropNotes(new Set(["a.md"]))).toBe(2);
		expect([...idx.notes()].sort()).toEqual(["b.md", "c.md"]);
		expect(() => idx.add("d.md", "x", [1, 2])).toThrow(/dimension/);
	});

	it("round-trips through JSON (base64 float32) and tracks the embedding model", () => {
		const idx = build();
		const back = VectorIndex.fromJSON(JSON.parse(JSON.stringify(idx.toJSON())));
		expect(back.entries).toHaveLength(4);
		expect(back.matches("test", "m")).toBe(true);
		expect(back.matches("test", "other")).toBe(false);
		expect(back.search([0, 0, 1], 1)[0].name).toBe("Baking");
		const v = new Float32Array([0.25, -1.5, 3.125, 7, 1e-7]);
		expect(Array.from(decodeVector(encodeVector(v)))).toEqual(Array.from(v));
	});
});
