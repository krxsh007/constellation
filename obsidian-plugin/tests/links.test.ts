import { describe, expect, it } from "vitest";
import fixture from "./fixtures/python-parity.json";
import { buildEffectiveLinks, filterAndScoreLinks, makeTitleResolver, pairKey, Relationship, SUPPORT_BONUS } from "../src/links";

describe("buildEffectiveLinks", () => {
	it.each(fixture.effective.map((c) => [`backlinks=${c.backlinks} max=${c.max}`, c]))(
		"matches the Python implementation (%s)",
		(_n, c) => {
			const e = c as (typeof fixture.effective)[number];
			expect(buildEffectiveLinks(e.links as Relationship[], { backlinks: e.backlinks, maxLinks: e.max })).toEqual(e.output);
		},
	);

	it("drops links touching notes that no longer exist", () => {
		const links = [{ from_note: "a.md", to_note: "gone.md", evidence: "", score: 1 }] as Relationship[];
		expect(buildEffectiveLinks(links, { existingTitles: new Set(["a.md"]), backlinks: false, maxLinks: 5 })).toEqual({});
	});
});

describe("title resolution", () => {
	it("tolerates missing suffix, brackets, quotes, case, aliases and bare basenames", () => {
		const resolve = makeTitleResolver(["Docker.md", "Infra/Kubernetes.md", "a/Same.md", "b/Same.md"]);
		expect(resolve("docker")).toBe("Docker.md");
		expect(resolve("[[Docker]]")).toBe("Docker.md");
		expect(resolve('"Kubernetes"')).toBe("Infra/Kubernetes.md");
		expect(resolve("infra/kubernetes.md")).toBe("Infra/Kubernetes.md");
		expect(resolve("[[Docker|the docker note]]")).toBe("Docker.md");
		expect(resolve("Same")).toBe("Same"); // ambiguous basename stays unresolved
		expect(resolve("unknown")).toBe("unknown");
	});
});

describe("filterAndScoreLinks", () => {
	const titles = new Set(["a.md", "b.md", "c.md"]);
	const concepts = [
		{ concept_name: "Docker", source_note: "a.md" },
		{ concept_name: "K8s", source_note: "b.md" },
		{ concept_name: "Helm", source_note: "b.md" },
	] as any[];
	const link = (over: Partial<Relationship>): Relationship => ({
		source: "Docker", target: "K8s", relationship: "uses", evidence: "e", from_note: "a.md", to_note: "b.md", ...over,
	});

	it("rejects self links, unknown notes and duplicates; scores by similarity plus support", () => {
		const { valid, rejected } = filterAndScoreLinks(
			[
				link({}),
				link({ target: "Helm" }), // second concept pair for the same notes: support, not a new link
				link({ to_note: "a.md" }),
				link({ to_note: "nope.md" }),
				link({ from_note: "", to_note: "", source: "K8s", target: "Docker" }), // resolved through concepts
			],
			{ titles, concepts, existingLinks: [], pairSimilarity: { [pairKey("a.md", "b.md")]: 0.7 } },
		);
		expect(valid.map((l) => `${l.from_note}->${l.to_note}`)).toEqual(["a.md->b.md", "b.md->a.md"]);
		expect(rejected).toBe(3);
		expect(valid[0].support).toBe(2);
		expect(valid[0].score).toBeCloseTo(0.7 + SUPPORT_BONUS, 6);
	});

	it("skips pairs that already exist", () => {
		const { valid } = filterAndScoreLinks([link({})], {
			titles, concepts, existingLinks: [link({})], pairSimilarity: {},
		});
		expect(valid).toEqual([]);
	});
});
