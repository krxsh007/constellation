import { describe, expect, it } from "vitest";
import { lineDiff, withContext } from "../src/diff";

describe("lineDiff", () => {
	it("marks added and removed lines", () => {
		const d = lineDiff("a\nb\nc\n", "a\nc\nd\n");
		expect(d).toEqual([
			{ op: " ", text: "a" },
			{ op: "-", text: "b" },
			{ op: " ", text: "c" },
			{ op: "+", text: "d" },
		]);
		const ctx = withContext(lineDiff("1\n2\n3\n4\n5\n6\n7\n", "1\n2\n3\n4\n5\n6\n7\nX\n"), 1);
		expect(ctx[0]).toBeNull();
		expect(ctx.slice(1)).toEqual([{ op: " ", text: "7" }, { op: "+", text: "X" }]);
	});
});
