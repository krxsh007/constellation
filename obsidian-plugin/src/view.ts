import { ItemView, setIcon, TFile, WorkspaceLeaf } from "obsidian";
import type ConstellationPlugin from "./main";
import { buildEffectiveLinks } from "./links";
import { ProgressEvent, STAGES, StageStatus } from "./types";

export const VIEW_TYPE = "constellation-view";

const STATUS_ICON: Record<StageStatus, string> = {
	pending: "circle",
	running: "loader",
	done: "check-circle-2",
	skipped: "minus-circle",
	error: "alert-circle",
};

/** Right-sidebar panel: pipeline stages, last-run stats, related notes. */
export class ConstellationView extends ItemView {
	constructor(leaf: WorkspaceLeaf, private plugin: ConstellationPlugin) {
		super(leaf);
	}

	getViewType() {
		return VIEW_TYPE;
	}

	getDisplayText() {
		return "Constellation";
	}

	getIcon() {
		return "sparkles";
	}

	async onOpen() {
		this.registerEvent(this.app.workspace.on("active-leaf-change", () => this.render()));
		this.render();
	}

	async onClose() {
		this.contentEl.empty();
	}

	render() {
		const el = this.contentEl;
		el.empty();
		el.addClass("constellation-view");
		const p = this.plugin;

		const actions = el.createDiv({ cls: "constellation-actions" });
		const btn = (icon: string, label: string, cb: () => void, cta = false) => {
			const b = actions.createEl("button", { cls: cta ? "mod-cta" : "", attr: { "aria-label": label } });
			const i = b.createSpan({ cls: "constellation-btn-icon" });
			setIcon(i, icon);
			b.createSpan({ text: label });
			b.disabled = p.running && icon !== "square";
			b.onclick = cb;
		};
		if (p.running) btn("square", "Cancel", () => p.cancel());
		else {
			btn("sparkles", "Link vault", () => p.runVault(false), true);
			btn("file-search", "Preview", () => p.runVault(true));
			btn("file-symlink", "Link note", () => p.runCurrentNote());
		}

		// Stages
		el.createEl("h6", { text: "Pipeline", cls: "constellation-heading" });
		const stages = el.createEl("ol", { cls: "constellation-stages" });
		STAGES.forEach((name, i) => {
			const st: ProgressEvent = p.stageState[i] ?? { stage: i, status: "pending" };
			const li = stages.createEl("li", { cls: `constellation-stage is-${st.status}` });
			const icon = li.createSpan({ cls: "constellation-stage-icon" });
			setIcon(icon, STATUS_ICON[st.status]);
			const body = li.createDiv({ cls: "constellation-stage-body" });
			body.createDiv({ cls: "constellation-stage-name", text: name });
			const detail =
				st.total !== undefined && st.status === "running" ? `${st.done ?? 0}/${st.total}` : (st.detail ?? "");
			if (detail) body.createDiv({ cls: "constellation-stage-detail", text: detail });
			if (st.status === "running" && st.total) {
				const bar = body.createDiv({ cls: "constellation-progress" });
				bar.createDiv({ cls: "constellation-progress-fill" }).style.width = `${Math.round(((st.done ?? 0) / st.total) * 100)}%`;
			}
		});

		// Last run
		const report = p.cache?.lastReport;
		el.createEl("h6", { text: "Last run", cls: "constellation-heading" });
		if (!report) {
			el.createDiv({ cls: "constellation-muted", text: "No run yet. Use “Link vault” to start." });
		} else {
			const grid = el.createDiv({ cls: "constellation-stats" });
			const stat = (label: string, value: string | number) => {
				const c = grid.createDiv({ cls: "constellation-stat" });
				c.createDiv({ cls: "constellation-stat-value", text: String(value) });
				c.createDiv({ cls: "constellation-stat-label", text: label });
			};
			stat("notes", report.notesTotal);
			stat("processed", report.notesProcessed);
			stat("written", report.notesWritten);
			stat("links", `${report.linksTotal}`);
			stat("new / removed", `+${report.linksNew} / −${report.linksRemoved}`);
			stat("tokens", report.inputTokens + report.outputTokens);
			const meta = el.createDiv({ cls: "constellation-muted" });
			meta.setText(
				`${new Date(report.startedAt).toLocaleString()} · ${report.durationS}s · ${report.llmCalls} LLM calls` +
					(report.dryRun ? " · preview" : ""),
			);
			if (report.notesFailed.length) {
				el.createDiv({ cls: "constellation-warning", text: `Failed (retried next run): ${report.notesFailed.join(", ")}` });
			}
			if (report.skippedDueToEdit.length) {
				el.createDiv({ cls: "constellation-warning", text: `Skipped (edited during run): ${report.skippedDueToEdit.join(", ")}` });
			}
		}

		// Related notes for the active file
		const file = this.app.workspace.getActiveFile();
		el.createEl("h6", { text: "Related notes", cls: "constellation-heading" });
		if (!file || file.extension !== "md") {
			el.createDiv({ cls: "constellation-muted", text: "Open a note to see its related notes." });
			return;
		}
		const links = p.cache?.links ?? [];
		const effective = buildEffectiveLinks(links, {
			backlinks: p.settings.backlinks,
			maxLinks: p.settings.maxLinksPerNote,
		});
		const outgoing = effective[file.path] ?? [];
		const incoming = links.filter((l) => l.to_note === file.path && !outgoing.some((o) => o.target === l.from_note));
		if (!outgoing.length && !incoming.length) {
			el.createDiv({ cls: "constellation-muted", text: "No related notes found for this note yet." });
			return;
		}
		const list = el.createDiv({ cls: "constellation-related" });
		const item = (path: string, evidence: string, score: number | undefined, dir: "out" | "in") => {
			const row = list.createDiv({ cls: "constellation-related-item" });
			const icon = row.createSpan({ cls: "constellation-related-icon" });
			setIcon(icon, dir === "out" ? "arrow-up-right" : "arrow-down-left");
			const body = row.createDiv({ cls: "constellation-related-body" });
			const name = body.createEl("a", { cls: "constellation-related-name", text: path.split("/").pop()!.replace(/\.md$/i, "") });
			name.onclick = (ev) => {
				ev.preventDefault();
				const target = this.app.vault.getAbstractFileByPath(path);
				if (target instanceof TFile) this.app.workspace.getLeaf(ev.ctrlKey || ev.metaKey).openFile(target);
			};
			if (evidence) body.createDiv({ cls: "constellation-related-evidence", text: evidence });
			if (score) row.createSpan({ cls: "constellation-related-score", text: score.toFixed(2) });
		};
		for (const o of outgoing) item(o.target, o.evidence, o.score, "out");
		for (const l of incoming) item(l.from_note, l.evidence, l.score, "in");
	}
}
