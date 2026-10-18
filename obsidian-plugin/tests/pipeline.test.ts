// Offline end-to-end runs of the pipeline: in-memory vault + storage, and a
// fake HTTP layer standing in for Groq and Gemini.
import { beforeEach, describe, expect, it } from "vitest";
import { Backup, commit, PipelineDeps, PluginStorage, plan, stripVault, undoLastRun, VaultIO } from "../src/pipeline";
import type { Http } from "../src/providers";
import { updateFrontmatterTags } from "../src/sections";
import { Cache, DEFAULT_SETTINGS, emptyCache } from "../src/types";
import { VectorIndex } from "../src/vectors";

class MemoryVault implements VaultIO {
	files = new Map<string, string>();
	fm = new Map<string, Record<string, unknown>>();
	/** Simulates the user saving a note between plan() and commit(). */
	onBeforeWrite?: (path: string) => void;

	listNotes() {
		return [...this.files.keys()].filter((p) => !p.startsWith("Templates/")).sort();
	}
	async read(p: string) {
		return this.files.get(p)!;
	}
	async writeIfUnchanged(p: string, expected: string, next: string) {
		this.onBeforeWrite?.(p);
		if (this.files.get(p) !== expected) return false;
		this.files.set(p, next);
		return true;
	}
	async write(p: string, c: string) {
		this.files.set(p, c);
	}
	exists(p: string) {
		return this.files.has(p);
	}
	frontmatterTags(p: string) {
		const t = this.fm.get(p)?.tags;
		return Array.isArray(t) ? t.map(String) : [];
	}
	async updateFrontmatterTags(p: string, add: string[], remove: string[]) {
		const fm = this.fm.get(p) ?? {};
		updateFrontmatterTags(fm, add, remove);
		this.fm.set(p, fm);
	}
	linkText(target: string) {
		return target.split("/").pop()!.replace(/\.md$/, "");
	}
}

class MemoryStorage implements PluginStorage {
	cache: Cache = emptyCache();
	index: unknown = null;
	backups: Backup[] = [];
	async loadCache() {
		return JSON.parse(JSON.stringify(this.cache));
	}
	async saveCache(c: Cache) {
		this.cache = JSON.parse(JSON.stringify(c));
	}
	async loadIndex() {
		return VectorIndex.fromJSON(this.index as any);
	}
	async saveIndex(i: VectorIndex) {
		this.index = JSON.parse(JSON.stringify(i.toJSON()));
	}
	async deleteIndex() {
		this.index = null;
	}
	async saveBackup(b: Backup, keep: number) {
		this.backups.push(JSON.parse(JSON.stringify(b)));
		this.backups = this.backups.slice(-keep);
	}
	async latestBackup() {
		return this.backups[this.backups.length - 1] ?? null;
	}
	async deleteBackup(id: string) {
		this.backups = this.backups.filter((b) => b.id !== id);
	}
}

const VECTORS: Record<string, number[]> = {
	Docker: [1, 0.1, 0],
	Kubernetes: [0.9, 0.25, 0],
	Sourdough: [0, 0, 1],
};

function fakeHttp(calls: { llm: string[]; embed: number }): Http {
	const section = (text: string, start: string, end?: string) => {
		const from = text.indexOf(start) + start.length;
		const to = end ? text.indexOf(end, from) : text.length;
		return text.slice(from, to).trim();
	};
	const ok = (body: unknown) => ({ status: 200, text: JSON.stringify(body), headers: {} });
	return async (req) => {
		const body = JSON.parse(req.body!);
		if (req.url.includes("batchEmbedContents")) {
			calls.embed++;
			return ok({
				embeddings: body.requests.map((r: any) => ({ values: VECTORS[r.content.parts[0].text.split(":")[0]] ?? [0.3, 0.3, 0.3] })),
			});
		}
		const prompt: string = body.messages[1].content;
		const reply = (content: unknown) =>
			ok({ choices: [{ message: { content: JSON.stringify(content) } }], usage: { prompt_tokens: 10, completion_tokens: 5 } });
		if (prompt.startsWith("Analyze the following text")) {
			calls.llm.push("extract");
			const text = section(prompt, "TEXT:");
			const concepts = Object.keys(VECTORS)
				.filter((n) => text.includes(n))
				.map((n) => ({ concept_name: n, category: "c", explanation: `${n} explained`, important_keywords: [], related_concepts: [], importance_score: 7 }));
			return reply({ concepts, tags: concepts.map((c) => `#${c.concept_name}`) });
		}
		if (prompt.startsWith("Analyze the provided concepts")) {
			calls.llm.push("verify");
			const concepts = JSON.parse(section(prompt, "CONCEPTS:\n", "\nTAGS:"));
			return reply({ concepts, tags: JSON.parse(section(prompt, "TAGS:\n")).map((t: string) => t.toLowerCase()) });
		}
		if (prompt.includes("NEW CONCEPTS:")) {
			calls.llm.push("relate");
			const all = [
				...JSON.parse(section(prompt, "NEW CONCEPTS:\n", "\n\nEXISTING CONCEPTS:")),
				...JSON.parse(section(prompt, "EXISTING CONCEPTS:\n")),
			];
			const d = all.find((c) => c.concept_name === "Docker");
			const k = all.find((c) => c.concept_name === "Kubernetes");
			const links = d && k
				? [{ source: "Kubernetes", target: "Docker", relationship: "uses", evidence: "K8s runs containers", from_note: k.source_note, to_note: d.source_note }]
				: [];
			return reply({ links });
		}
		if (prompt.startsWith("Below are proposed relationships")) {
			calls.llm.push("verify-links");
			return reply({ links: JSON.parse(section(prompt, "RELATIONSHIPS:\n")) });
		}
		return { status: 500, text: "unexpected", headers: {} };
	};
}

describe("pipeline end to end", () => {
	let vault: MemoryVault;
	let storage: MemoryStorage;
	let calls: { llm: string[]; embed: number };
	let settings: typeof DEFAULT_SETTINGS;

	const deps = (): PipelineDeps => ({
		io: vault,
		storage,
		http: fakeHttp(calls),
		settings,
		log: () => {},
		progress: () => {},
	});
	const run = async (scopePath?: string) => {
		const before = await storage.loadCache();
		const p = await plan(deps(), { dryRun: false, scopePath });
		return { plan: p, report: await commit(deps(), p, before) };
	};

	beforeEach(() => {
		vault = new MemoryVault();
		storage = new MemoryStorage();
		calls = { llm: [], embed: 0 };
		settings = { ...DEFAULT_SETTINGS, groqApiKeys: "k1", geminiApiKey: "g", concurrency: 2 };
		vault.files.set("Infra/Kubernetes.md", "# Kubernetes\n\nKubernetes schedules containers.\n");
		vault.files.set("Docker.md", "# Docker\n\nDocker builds images.\n\nMy own closing line.\n");
		vault.files.set("Sourdough.md", "# Sourdough\n\nSourdough needs a starter.\n");
		vault.files.set("Templates/Docker template.md", "Docker");
	});

	it("links related notes, tags them and leaves user text alone", async () => {
		const { report } = await run();
		expect(vault.files.get("Infra/Kubernetes.md")).toBe(
			"# Kubernetes\n\nKubernetes schedules containers.\n\n## Tags\n#kubernetes\n\n## Related Links\n- [[Docker]]\n",
		);
		expect(vault.files.get("Docker.md")).toBe("# Docker\n\nDocker builds images.\n\nMy own closing line.\n\n## Tags\n#docker\n");
		expect(vault.files.get("Templates/Docker template.md")).toBe("Docker");
		expect(report.linksTotal).toBe(1);
		expect(report.notesWritten).toBe(3);
		expect(storage.backups).toHaveLength(1);
		expect(storage.cache.pending).toEqual([]);
		expect(calls.llm.filter((c) => c === "extract")).toHaveLength(3);
	});

	it("costs nothing on an unchanged second run, and only re-extracts edited notes", async () => {
		await run();
		calls.llm = [];
		calls.embed = 0;
		const second = await run();
		expect(calls.llm).toEqual([]);
		expect(calls.embed).toBe(0);
		expect(second.report.notesWritten).toBe(0);

		vault.files.set("Sourdough.md", vault.files.get("Sourdough.md")! + "\nAdded a paragraph.\n");
		await run();
		expect(calls.llm.filter((c) => c === "extract")).toHaveLength(1);
		expect(vault.files.get("Infra/Kubernetes.md")).toContain("- [[Docker]]");
	});

	it("backlinks and annotations follow the settings", async () => {
		settings.backlinks = true;
		settings.annotateLinks = true;
		await run();
		expect(vault.files.get("Docker.md")).toContain("## Related Links\n- [[Kubernetes]] — K8s runs containers\n");
	});

	it("never overwrites a note the user edited mid-run, and retries it next run", async () => {
		const before = await storage.loadCache();
		const p = await plan(deps(), { dryRun: false });
		vault.onBeforeWrite = (path) => {
			if (path === "Infra/Kubernetes.md") vault.files.set(path, "# Kubernetes\n\nUser rewrote this.\n");
		};
		const report = await commit(deps(), p, before);
		vault.onBeforeWrite = undefined;
		expect(report.skippedDueToEdit).toEqual(["Infra/Kubernetes.md"]);
		expect(vault.files.get("Infra/Kubernetes.md")).toBe("# Kubernetes\n\nUser rewrote this.\n");
		expect(storage.cache.files["Infra/Kubernetes.md"]).toBeUndefined();
	});

	it("preview computes changes without touching notes or the cache", async () => {
		const p = await plan(deps(), { dryRun: true });
		expect(p.changes.map((c) => c.path).sort()).toEqual(["Docker.md", "Infra/Kubernetes.md", "Sourdough.md"]);
		expect(vault.files.get("Docker.md")).not.toContain("## Tags");
		expect(storage.cache).toEqual(emptyCache());
		expect(storage.index).toBeNull();
	});

	it("undo restores the notes and the cache", async () => {
		const original = new Map(vault.files);
		await run();
		const { restored } = await undoLastRun(deps());
		expect(restored).toBe(3);
		expect(vault.files).toEqual(original);
		expect(storage.cache.links).toEqual([]);
	});

	it("remove all output strips sections and only our frontmatter tags", async () => {
		settings.tagsInFrontmatter = true;
		vault.fm.set("Docker.md", { tags: ["mine"] });
		await run();
		expect(vault.fm.get("Docker.md")!.tags).toEqual(["mine", "docker"]);
		expect(vault.files.get("Docker.md")).not.toContain("## Tags");
		const { cleaned } = await stripVault(deps(), false);
		expect(cleaned).toBe(3);
		expect(vault.fm.get("Docker.md")!.tags).toEqual(["mine"]);
		expect(vault.files.get("Infra/Kubernetes.md")).toBe("# Kubernetes\n\nKubernetes schedules containers.\n");
		expect(storage.index).toBeNull();
	});

	it("changing the embedding model re-embeds without new LLM calls", async () => {
		await run();
		calls.llm = [];
		calls.embed = 0;
		settings.geminiEmbeddingDimension = 256;
		await run();
		expect(calls.llm).toEqual([]);
		expect(calls.embed).toBe(1);
		expect((storage.index as any).manifest.model).toBe("models/gemini-embedding-2@256");
	});

	it("link current note only extracts that note", async () => {
		await run("Infra/Kubernetes.md");
		expect(calls.llm.filter((c) => c === "extract")).toHaveLength(1);
		expect(storage.cache.files["Docker.md"]).toBeUndefined();
	});

	it("a failed extraction keeps the note's previous links and is retried", async () => {
		await run();
		vault.files.set("Infra/Kubernetes.md", vault.files.get("Infra/Kubernetes.md")!.replace("schedules", "orchestrates"));
		const good = fakeHttp(calls);
		const failing: Http = async (req) =>
			req.url.includes("groq") && JSON.parse(req.body!).messages[1].content.includes("orchestrates")
				? { status: 401, text: '{"error":{"message":"bad key"}}', headers: {} }
				: good(req);
		const d = { ...deps(), http: failing };
		const before = await storage.loadCache();
		const p = await plan(d, { dryRun: false });
		const report = await commit(d, p, before);
		expect(report.notesFailed).toEqual(["Infra/Kubernetes.md"]);
		expect(vault.files.get("Infra/Kubernetes.md")).toContain("- [[Docker]]");
		expect(storage.cache.files["Infra/Kubernetes.md"]).toBeUndefined();
	});
});
