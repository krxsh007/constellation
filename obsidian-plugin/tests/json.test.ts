import { describe, expect, it } from "vitest";
import fixture from "./fixtures/python-parity.json";
import { CONCEPT_SCHEMA, extractJsonText, LLMResponseError, parseLLMOutput, RELATIONSHIP_SCHEMA, repairTruncatedJson } from "../src/json";

const link = (source = "Docker", target = "Kubernetes", over: Record<string, string> = {}) => ({
	source, target, relationship: "uses", evidence: "K8s schedules containers.", from_note: "a.md", to_note: "b.md", ...over,
});
const concept = (name = "Docker") => ({
	concept_name: name, category: "tool", explanation: "Containers.", important_keywords: ["c"], related_concepts: [], importance_score: 8,
});

describe("repairTruncatedJson", () => {
	it.each(fixture.repair.map((c, i) => [i, c]))("matches the Python implementation (case %s)", (_i, c) => {
		const r = c as (typeof fixture.repair)[number];
		expect(repairTruncatedJson(r.input)).toBe(r.output);
		if (r.sources) {
			// The port is more tolerant (missing optional fields are defaulted), never less.
			const got = parseLLMOutput(r.input, RELATIONSHIP_SCHEMA).items.map((l) => l.source);
			for (const s of r.sources) expect(got).toContain(s);
		}
	});

	it("closes a value cut inside a string", () => {
		const cut = JSON.stringify({ links: [link("A", "B", { evidence: "a very long ev" })] }).slice(0, -14);
		expect(() => JSON.parse(repairTruncatedJson(cut))).not.toThrow();
	});
});

describe("parseLLMOutput", () => {
	it("handles fences and surrounding prose", () => {
		const raw = "Sure!\n```json\n" + JSON.stringify({ links: [link()] }) + "\n```\nHope that helps.";
		expect(parseLLMOutput(raw, RELATIONSHIP_SCHEMA).items).toHaveLength(1);
		expect(extractJsonText('blah {"a": {"b": 1}} trailing')).toBe('{"a": {"b": 1}}');
	});

	it("maps field-name aliases without clobbering real fields", () => {
		const out = parseLLMOutput(JSON.stringify({ links: [link()] }), RELATIONSHIP_SCHEMA);
		expect(out.items[0].source).toBe("Docker");
		const { concept_name: _drop, ...rest } = concept();
		const c = parseLLMOutput(JSON.stringify({ concepts: [{ ...rest, name: "Redis", description: "x" }], tags: ["#a"] }), CONCEPT_SCHEMA);
		expect(c.items[0].concept_name).toBe("Redis");
		expect(c.items[0].explanation).toBe("Containers.");
		expect(c.tags).toEqual(["#a"]);
	});

	it("drops individually invalid items instead of failing the whole response", () => {
		const out = parseLLMOutput(JSON.stringify({ concepts: [concept(), { concept_name: "Broken" }], tags: [] }), CONCEPT_SCHEMA);
		expect(out.items.map((c) => c.concept_name)).toEqual(["Docker"]);
		expect(out.dropped).toBe(1);
	});

	it("accepts a bare top-level array and coerces numeric strings", () => {
		const out = parseLLMOutput(JSON.stringify([{ ...concept(), importance_score: "7" }]), CONCEPT_SCHEMA);
		expect(out.items[0].importance_score).toBe(7);
	});

	it("raises an LLMResponseError that quotes the raw output", () => {
		expect(() => parseLLMOutput("I could not answer that.", CONCEPT_SCHEMA)).toThrow(LLMResponseError);
		expect(() => parseLLMOutput("I could not answer that.", CONCEPT_SCHEMA)).toThrow(/I could not answer/);
	});
});
