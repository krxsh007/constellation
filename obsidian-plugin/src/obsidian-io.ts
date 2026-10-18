// Obsidian-backed implementations of VaultIO, PluginStorage and Http.

import { App, normalizePath, parseFrontMatterTags, requestUrl, TFile } from "obsidian";
import type { Backup, PluginStorage, VaultIO } from "./pipeline";
import type { Http } from "./providers";
import { updateFrontmatterTags } from "./sections";
import { Cache, emptyCache, parseList } from "./types";
import { SerializedIndex, VectorIndex } from "./vectors";

export const obsidianHttp: Http = async (req) => {
	const res = await requestUrl({
		url: req.url,
		method: req.method,
		headers: req.headers,
		body: req.body,
		contentType: req.headers?.["Content-Type"],
		throw: false,
	});
	return { status: res.status, text: res.text, headers: res.headers ?? {} };
};

/** True when `path` lies inside one of the excluded folders (case-insensitive). */
export function isExcluded(path: string, excluded: string[]): boolean {
	const lower = path.toLowerCase();
	return excluded.some((raw) => {
		const folder = raw.trim().replace(/^\/+|\/+$/g, "").toLowerCase();
		if (!folder) return false;
		if (lower.startsWith(folder + "/")) return true;
		// A bare folder name also matches that folder at any depth, like the Python app.
		return !folder.includes("/") && lower.split("/").slice(0, -1).includes(folder);
	});
}

export class ObsidianVaultIO implements VaultIO {
	constructor(private app: App, private excludedFolders: () => string) {}

	private file(path: string): TFile {
		const f = this.app.vault.getAbstractFileByPath(path);
		if (!(f instanceof TFile)) throw new Error(`Note not found: ${path}`);
		return f;
	}

	listNotes(): string[] {
		const excluded = [...parseList(this.excludedFolders()), ".trash"];
		return this.app.vault
			.getMarkdownFiles()
			.map((f) => f.path)
			.filter((p) => !isExcluded(p, excluded))
			.sort();
	}

	read(path: string): Promise<string> {
		return this.app.vault.read(this.file(path));
	}

	async writeIfUnchanged(path: string, expected: string, next: string): Promise<boolean> {
		let ok = false;
		await this.app.vault.process(this.file(path), (current) => {
			if (current !== expected) return current;
			ok = true;
			return next;
		});
		return ok;
	}

	async write(path: string, content: string): Promise<void> {
		const f = this.app.vault.getAbstractFileByPath(path);
		if (f instanceof TFile) {
			await this.app.vault.process(f, () => content);
			return;
		}
		const dir = path.split("/").slice(0, -1).join("/");
		if (dir && !this.app.vault.getAbstractFileByPath(dir)) await this.app.vault.createFolder(dir);
		await this.app.vault.create(path, content);
	}

	exists(path: string): boolean {
		return this.app.vault.getAbstractFileByPath(path) instanceof TFile;
	}

	frontmatterTags(path: string): string[] {
		const fm = this.app.metadataCache.getFileCache(this.file(path))?.frontmatter;
		return (parseFrontMatterTags(fm) || []).map((t) => t.replace(/^#/, ""));
	}

	async updateFrontmatterTags(path: string, add: string[], remove: string[]): Promise<void> {
		await this.app.fileManager.processFrontMatter(this.file(path), (fm: Record<string, unknown>) => {
			updateFrontmatterTags(fm, add, remove);
		});
	}

	linkText(targetPath: string, sourcePath: string): string {
		const f = this.app.vault.getAbstractFileByPath(targetPath);
		if (f instanceof TFile) return this.app.metadataCache.fileToLinktext(f, sourcePath, true);
		return targetPath.replace(/\.md$/i, "");
	}
}

/** cache.json, index.json and backups/ inside the plugin folder. */
export class ObsidianStorage implements PluginStorage {
	constructor(private app: App, private dir: string) {}

	private path(...parts: string[]) {
		return normalizePath([this.dir, ...parts].join("/"));
	}

	private async readJson<T>(path: string): Promise<T | null> {
		const a = this.app.vault.adapter;
		// A crash between writeJson's remove and rename leaves only the temp file.
		if (!(await a.exists(path))) path = path + ".tmp";
		if (!(await a.exists(path))) return null;
		try {
			return JSON.parse(await a.read(path)) as T;
		} catch (e) {
			console.warn(`Constellation: could not read ${path}`, e);
			return null;
		}
	}

	/** Temp-file swap so a crash mid-write never leaves a half-written file. */
	private async writeJson(path: string, data: unknown): Promise<void> {
		const a = this.app.vault.adapter;
		const dir = path.split("/").slice(0, -1).join("/");
		if (dir && !(await a.exists(dir))) await a.mkdir(dir);
		const tmp = path + ".tmp";
		await a.write(tmp, JSON.stringify(data));
		if (await a.exists(path)) await a.remove(path);
		await a.rename(tmp, path);
	}

	async loadCache(): Promise<Cache> {
		const data = await this.readJson<Partial<Cache>>(this.path("cache.json"));
		return { ...emptyCache(), ...(data || {}) };
	}

	saveCache(cache: Cache): Promise<void> {
		return this.writeJson(this.path("cache.json"), cache);
	}

	async loadIndex(): Promise<VectorIndex> {
		return VectorIndex.fromJSON(await this.readJson<SerializedIndex>(this.path("index.json")));
	}

	saveIndex(index: VectorIndex): Promise<void> {
		return this.writeJson(this.path("index.json"), index.toJSON());
	}

	async deleteIndex(): Promise<void> {
		const p = this.path("index.json");
		if (await this.app.vault.adapter.exists(p)) await this.app.vault.adapter.remove(p);
	}

	private async backupIds(): Promise<string[]> {
		const dir = this.path("backups");
		if (!(await this.app.vault.adapter.exists(dir))) return [];
		const listed = await this.app.vault.adapter.list(dir);
		return listed.files
			.filter((f) => f.endsWith(".json"))
			.map((f) => f.split("/").pop()!.replace(/\.json$/, ""))
			.sort();
	}

	async saveBackup(backup: Backup, keep: number): Promise<void> {
		await this.writeJson(this.path("backups", `${backup.id}.json`), backup);
		const ids = await this.backupIds();
		for (const id of ids.slice(0, Math.max(0, ids.length - Math.max(1, keep)))) await this.deleteBackup(id);
	}

	async latestBackup(): Promise<Backup | null> {
		const ids = await this.backupIds();
		if (!ids.length) return null;
		return this.readJson<Backup>(this.path("backups", `${ids[ids.length - 1]}.json`));
	}

	async deleteBackup(id: string): Promise<void> {
		const p = this.path("backups", `${id}.json`);
		if (await this.app.vault.adapter.exists(p)) await this.app.vault.adapter.remove(p);
	}
}
