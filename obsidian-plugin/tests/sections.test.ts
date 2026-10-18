import { describe, expect, it } from "vitest";
import fixture from "./fixtures/python-parity.json";
import {
	coreHash,
	normaliseTags,
	prepareNoteText,
	rankAndCap,
	renderNote,
	splitFrontmatter,
	stripAutoSections,
	updateFrontmatterTags,
} from "../src/sections";

describe("stripAutoSections", () => {
	it.each(fixture.strip.map((c) => [JSON.stringify(c.input).slice(0, 60), c.input, c.output]))(
		"matches the Python implementation: %s",
		(_name, input, output) => {
			expect(stripAutoSections(input as string)).toBe(output);
		},
	);

	it("keeps prose appended below generated sections", () => {
		expect(stripAutoSections("# A\n\nBody.\n\n## Tags\n#x #y\n\nMore thoughts.\n")).toBe("# A\n\nBody.\n\nMore thoughts.");
	});
});

describe("renderNote", () => {
	it.each(fixture.render.map((c, i) => [i, c]))("matches the Python implementation (case %s)", (_i, c) => {
		const r = c as (typeof fixture.render)[number];
		expect(renderNote(r.base, r.tags, r.targets, { maxLinks: r.max, annotate: r.annotate })).toBe(r.output);
	});

	it("is stable across runs and round-trips through strip", () => {
		const base = "# A\n\nBody.\n\nAppended later by hand.";
		const opts = { maxLinks: 5, annotate: true };
		const targets = [{ target: "B", evidence: "because", score: 0.5 }];
		const first = renderNote(base, ["#x", "#y"], targets, opts);
		const second = renderNote(stripAutoSections(first), ["#x", "#y"], targets, opts);
		expect(stripAutoSections(first)).toBe(base);
		expect(second).toBe(first);
		expect(coreHash(first)).toBe(coreHash(base));
		expect(coreHash(first + "\nnew user text\n")).not.toBe(coreHash(first));
	});

	it("caps by score, then orders alphabetically", () => {
		const out = renderNote("# R", null, [
			{ target: "Z", evidence: "", score: 0.9 },
			{ target: "A", evidence: "", score: 0.1 },
			{ target: "M", evidence: "", score: 0.8 },
		], { maxLinks: 2, annotate: false });
		expect(out).toBe("# R\n\n## Related Links\n- [[M]]\n- [[Z]]\n");
	});
});

describe("rankAndCap", () => {
	it("breaks score ties alphabetically and keeps everything when max is 0", () => {
		const items = [{ t: "b", score: 1 }, { t: "a", score: 1 }, { t: "c", score: 2 }];
		expect(rankAndCap(items, 2, (i) => i.t).map((i) => i.t)).toEqual(["a", "c"]);
		expect(rankAndCap(items, 0, (i) => i.t).map((i) => i.t)).toEqual(["a", "b", "c"]);
	});
});

describe("tags", () => {
	it("normalises like the Python implementation", () => {
		expect(normaliseTags(fixture.tags.input)).toEqual(fixture.tags.output);
	});

	it("frontmatter: removes only our tags and never duplicates user tags", () => {
		const fm: Record<string, unknown> = { tags: ["mine", "ours-old"], title: "x" };
		const res = updateFrontmatterTags(fm, ["#Mine", "new"], ["ours-old"]);
		expect(fm.tags).toEqual(["mine", "new"]);
		expect(res.added).toEqual(["new"]);
		const empty: Record<string, unknown> = { tags: ["new"] };
		updateFrontmatterTags(empty, [], ["new"]);
		expect("tags" in empty).toBe(false);
	});
});

describe("frontmatter-aware hashing", () => {
	it("ignores frontmatter changes and the generated sections", () => {
		const a = "---\ntags: [a]\n---\n# Note\nBody";
		const b = "---\ntags:\n  - a\n  - b\n---\n# Note\nBody\n\n## Related Links\n- [[X]]\n";
		expect(coreHash(a)).toBe(coreHash(b));
		expect(splitFrontmatter(a)[1]).toBe("# Note\nBody");
		expect(prepareNoteText(b, 1000)).toBe("# Note\nBody");
		expect(prepareNoteText("x".repeat(50), 10)).toContain("[... note truncated");
	});
});
