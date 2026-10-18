// Minimal line diff for the preview modal.

export type DiffLine = { op: " " | "+" | "-"; text: string };

export function lineDiff(before: string, after: string): DiffLine[] {
	const a = before.replace(/\n$/, "").split("\n");
	const b = after.replace(/\n$/, "").split("\n");
	// Common prefix/suffix first: generated sections live at the end of a note,
	// so this usually leaves a handful of lines for the LCS table.
	let start = 0;
	while (start < a.length && start < b.length && a[start] === b[start]) start++;
	let endA = a.length;
	let endB = b.length;
	while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
		endA--;
		endB--;
	}
	const midA = a.slice(start, endA);
	const midB = b.slice(start, endB);
	const out: DiffLine[] = a.slice(0, start).map((text) => ({ op: " ", text }));

	if (midA.length * midB.length > 4_000_000) {
		out.push(...midA.map((text) => ({ op: "-" as const, text })), ...midB.map((text) => ({ op: "+" as const, text })));
	} else {
		const n = midA.length;
		const m = midB.length;
		const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
		for (let i = n - 1; i >= 0; i--)
			for (let j = m - 1; j >= 0; j--)
				lcs[i][j] = midA[i] === midB[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
		let i = 0;
		let j = 0;
		while (i < n && j < m) {
			if (midA[i] === midB[j]) {
				out.push({ op: " ", text: midA[i] });
				i++;
				j++;
			} else if (lcs[i + 1][j] >= lcs[i][j + 1]) out.push({ op: "-", text: midA[i++] });
			else out.push({ op: "+", text: midB[j++] });
		}
		while (i < n) out.push({ op: "-", text: midA[i++] });
		while (j < m) out.push({ op: "+", text: midB[j++] });
	}
	out.push(...a.slice(endA).map((text) => ({ op: " " as const, text })));
	return out;
}

/** Keeps changed lines plus `context` lines around them; null marks a gap. */
export function withContext(lines: DiffLine[], context = 2): (DiffLine | null)[] {
	const keep = new Array(lines.length).fill(false);
	lines.forEach((l, i) => {
		if (l.op === " ") return;
		for (let k = Math.max(0, i - context); k <= Math.min(lines.length - 1, i + context); k++) keep[k] = true;
	});
	const out: (DiffLine | null)[] = [];
	let gap = false;
	lines.forEach((l, i) => {
		if (keep[i]) {
			out.push(l);
			gap = false;
		} else if (!gap) {
			out.push(null);
			gap = true;
		}
	});
	return out;
}
