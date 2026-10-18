import { Notice, Plugin, WorkspaceLeaf } from "obsidian";
import { ConfirmModal, PreviewModal } from "./modals";
import { ObsidianStorage, ObsidianVaultIO, obsidianHttp } from "./obsidian-io";
import { CancelledError, commit, PipelineDeps, plan, previewReport, RunPlan, stripVault, undoLastRun } from "./pipeline";
import { ConstellationSettingTab } from "./settings";
import { Cache, ConstellationSettings, DEFAULT_SETTINGS, ProgressEvent, STAGES } from "./types";
import { ConstellationView, VIEW_TYPE } from "./view";

export default class ConstellationPlugin extends Plugin {
	settings: ConstellationSettings = { ...DEFAULT_SETTINGS };
	cache: Cache | null = null;
	running = false;
	stageState: ProgressEvent[] = [];
	private cancelled = false;
	private statusEl!: HTMLElement;
	private storage!: ObsidianStorage;
	private io!: ObsidianVaultIO;
	private renderTimer: number | null = null;

	async onload() {
		await this.loadSettings();
		const dir = this.manifest.dir ?? `${this.app.vault.configDir}/plugins/${this.manifest.id}`;
		this.storage = new ObsidianStorage(this.app, dir);
		this.io = new ObsidianVaultIO(this.app, () => this.settings.excludedFolders);

		this.registerView(VIEW_TYPE, (leaf: WorkspaceLeaf) => new ConstellationView(leaf, this));
		this.addRibbonIcon("sparkles", "Constellation", () => this.activateView());
		this.statusEl = this.addStatusBarItem();
		this.statusEl.addClass("constellation-status");
		this.setStatus("");

		this.addCommand({ id: "link-vault", name: "Link entire vault", callback: () => this.runVault(false) });
		this.addCommand({
			id: "link-current-note",
			name: "Link current note",
			checkCallback: (checking) => {
				const file = this.app.workspace.getActiveFile();
				if (!file || file.extension !== "md") return false;
				if (!checking) this.runCurrentNote();
				return true;
			},
		});
		this.addCommand({ id: "preview-changes", name: "Preview changes", callback: () => this.runVault(true) });
		this.addCommand({ id: "undo-last-run", name: "Undo last run", callback: () => this.undo() });
		this.addCommand({ id: "remove-all-output", name: "Remove all Constellation output", callback: () => this.removeAll() });
		this.addCommand({ id: "open-panel", name: "Open panel", callback: () => this.activateView() });
		this.addCommand({ id: "cancel-run", name: "Cancel running job", checkCallback: (c) => {
			if (!this.running) return false;
			if (!c) this.cancel();
			return true;
		} });

		this.addSettingTab(new ConstellationSettingTab(this.app, this));

		this.app.workspace.onLayoutReady(async () => {
			this.cache = await this.storage.loadCache();
			this.refreshView();
		});
	}

	onunload() {
		this.cancelled = true;
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
		this.refreshView();
	}

	async activateView() {
		const { workspace } = this.app;
		let leaf = workspace.getLeavesOfType(VIEW_TYPE)[0];
		if (!leaf) {
			const right = workspace.getRightLeaf(false);
			if (!right) return;
			await right.setViewState({ type: VIEW_TYPE, active: true });
			leaf = right;
		}
		workspace.revealLeaf(leaf);
	}

	/** Re-render the panel, coalescing bursts of progress events. */
	refreshView() {
		if (this.renderTimer !== null) return;
		this.renderTimer = window.setTimeout(() => {
			this.renderTimer = null;
			for (const leaf of this.app.workspace.getLeavesOfType(VIEW_TYPE)) {
				if (leaf.view instanceof ConstellationView) leaf.view.render();
			}
		}, 100);
	}

	private setStatus(text: string) {
		this.statusEl.setText(text ? `✦ ${text}` : "✦ Constellation");
		this.statusEl.toggleClass("is-running", this.running);
	}

	private deps(): PipelineDeps {
		return {
			io: this.io,
			storage: this.storage,
			http: obsidianHttp,
			settings: { ...this.settings },
			log: (msg) => console.log(`[Constellation] ${msg.trim()}`),
			progress: (e) => this.onProgress(e),
			isCancelled: () => this.cancelled,
		};
	}

	private onProgress(e: ProgressEvent) {
		this.stageState[e.stage] = { ...this.stageState[e.stage], ...e };
		const counter = e.total !== undefined && e.status === "running" ? ` ${e.done ?? 0}/${e.total}` : "";
		this.setStatus(`${e.stage + 1}/${STAGES.length} ${STAGES[e.stage]}${counter}`);
		this.refreshView();
	}

	cancel() {
		if (!this.running) return;
		this.cancelled = true;
		new Notice("Constellation: cancelling after the current request…");
	}

	private async exclusive(label: string, job: () => Promise<void>) {
		if (this.running) {
			new Notice("Constellation is already running.");
			return;
		}
		this.running = true;
		this.cancelled = false;
		this.stageState = STAGES.map((_, i) => ({ stage: i, status: "pending" as const }));
		this.setStatus(label);
		this.refreshView();
		try {
			await job();
		} catch (e) {
			if (e instanceof CancelledError) {
				new Notice("Constellation: run cancelled. Nothing was written.");
			} else {
				console.error("[Constellation]", e);
				new Notice(`Constellation failed: ${(e as Error).message}`, 10000);
			}
			const running = this.stageState.findIndex((s) => s.status === "running");
			if (running >= 0) this.stageState[running] = { ...this.stageState[running], status: "error" };
		} finally {
			this.running = false;
			this.cancelled = false;
			this.setStatus("");
			this.cache = await this.storage.loadCache();
			this.refreshView();
		}
	}

	private missingKeys(): string | null {
		const s = this.settings;
		if (s.llmProvider === "groq" && !s.groqApiKeys.trim()) return "Add a Groq API key in Settings → Constellation (or switch to Ollama).";
		if (s.embeddingProvider === "gemini" && !s.geminiApiKey.trim()) return "Add a Gemini API key in Settings → Constellation (or switch to Ollama).";
		return null;
	}

	runVault(preview: boolean) {
		this.start(preview, undefined);
	}

	runCurrentNote() {
		const file = this.app.workspace.getActiveFile();
		if (!file || file.extension !== "md") {
			new Notice("Constellation: open a note first.");
			return;
		}
		this.start(false, file.path);
	}

	private start(preview: boolean, scopePath: string | undefined) {
		const missing = this.missingKeys();
		if (missing) {
			new Notice(`Constellation: ${missing}`, 8000);
			return;
		}
		void this.exclusive(preview ? "Previewing…" : "Linking…", async () => {
			const deps = this.deps();
			const before = await this.storage.loadCache();
			const run = await plan(deps, { dryRun: preview, scopePath });
			if (preview) {
				const apply = await new Promise<boolean>((resolve) => new PreviewModal(this.app, run, resolve).open());
				if (!apply) {
					await this.storage.saveCache({ ...before, lastReport: previewReport(run) });
					new Notice("Constellation: preview closed, nothing written.");
					return;
				}
			}
			await this.finish(deps, run, before);
		});
	}

	private async finish(deps: PipelineDeps, run: RunPlan, before: Cache) {
		const report = await commit(deps, run, before);
		const parts = [`${report.notesWritten} note(s) updated`, `${report.linksTotal} links (+${report.linksNew}/−${report.linksRemoved})`];
		if (report.notesFailed.length) parts.push(`${report.notesFailed.length} failed`);
		if (report.skippedDueToEdit.length) parts.push(`${report.skippedDueToEdit.length} skipped (edited)`);
		new Notice(`Constellation: ${parts.join(", ")}.`, 6000);
	}

	undo() {
		void this.exclusive("Undoing…", async () => {
			const backup = await this.storage.latestBackup();
			if (!backup) {
				new Notice("Constellation: nothing to undo.");
				return;
			}
			const count = Object.keys(backup.files).length;
			const ok = await new Promise<boolean>((resolve) =>
				new ConfirmModal(
					this.app,
					"Undo last run?",
					`Restores ${count} note(s) to how they were before “${backup.label}” (${new Date(backup.createdAt).toLocaleString()}). Edits you made to those notes since then are replaced.`,
					"Restore",
					resolve,
				).open(),
			);
			if (!ok) return;
			const { restored } = await undoLastRun(this.deps());
			new Notice(`Constellation: restored ${restored} note(s).`);
		});
	}

	removeAll() {
		void this.exclusive("Scanning…", async () => {
			const deps = this.deps();
			const { cleaned: count } = await stripVault(deps, true);
			const ok = await new Promise<boolean>((resolve) =>
				new ConfirmModal(
					this.app,
					"Remove all Constellation output?",
					`Removes the generated ## Tags / ## Related Links sections and the frontmatter tags Constellation added from ${count} note(s), then resets the cache and index. Your own text and tags stay. A snapshot is taken first, so “Undo last run” can bring it back.`,
					"Remove",
					resolve,
				).open(),
			);
			if (!ok) return;
			const { cleaned } = await stripVault(deps, false);
			new Notice(`Constellation: cleaned ${cleaned} note(s).`);
		});
	}
}
