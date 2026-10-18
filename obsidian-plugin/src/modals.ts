import { App, Modal, Setting } from "obsidian";
import { lineDiff, withContext } from "./diff";
import type { RunPlan } from "./pipeline";

/** Lists every planned per-note change as a diff, with Apply / Cancel. */
export class PreviewModal extends Modal {
	private decided = false;

	constructor(app: App, private run: RunPlan, private onDecision: (apply: boolean) => void) {
		super(app);
	}

	onOpen(): void {
		const { contentEl } = this;
		this.modalEl.addClass("constellation-preview-modal");
		this.titleEl.setText("Constellation: preview changes");
		const changes = this.run.changes;
		const summary = contentEl.createDiv({ cls: "constellation-preview-summary" });
		summary.setText(
			changes.length
				? `${changes.length} note${changes.length === 1 ? "" : "s"} would change. Nothing has been written yet.`
				: "No notes would change.",
		);
		if (this.run.failed.length) {
			contentEl.createDiv({
				cls: "constellation-warning",
				text: `${this.run.failed.length} note(s) failed and will be retried next run: ${this.run.failed.join(", ")}`,
			});
		}

		const list = contentEl.createDiv({ cls: "constellation-preview-list" });
		for (const change of changes) {
			const details = list.createEl("details", { cls: "constellation-diff" });
			if (changes.length <= 3) details.open = true;
			const head = details.createEl("summary");
			head.createSpan({ cls: "constellation-diff-path", text: change.path });
			head.createSpan({ cls: "constellation-diff-meta", text: change.summary });
			const pre = details.createEl("pre", { cls: "constellation-diff-body" });
			for (const line of withContext(lineDiff(change.before, change.after))) {
				if (!line) {
					pre.createDiv({ cls: "constellation-diff-gap", text: "⋯" });
					continue;
				}
				const cls = line.op === "+" ? "is-added" : line.op === "-" ? "is-removed" : "is-context";
				pre.createDiv({ cls: `constellation-diff-line ${cls}`, text: `${line.op} ${line.text}` });
			}
			if (change.fmAdd.length || change.fmRemove.length) {
				const fm = details.createDiv({ cls: "constellation-diff-fm" });
				fm.setText(
					"Frontmatter tags: " +
						[...change.fmAdd.map((t) => `+${t}`), ...change.fmRemove.map((t) => `−${t}`)].join("  "),
				);
			}
		}

		new Setting(contentEl)
			.addButton((b) => b.setButtonText("Cancel").onClick(() => this.decide(false)))
			.addButton((b) =>
				b
					.setButtonText(changes.length ? `Apply ${changes.length} change${changes.length === 1 ? "" : "s"}` : "Close")
					.setCta()
					.onClick(() => this.decide(changes.length > 0)),
			);
	}

	private decide(apply: boolean) {
		this.decided = true;
		this.close();
		this.onDecision(apply);
	}

	onClose(): void {
		this.contentEl.empty();
		if (!this.decided) this.onDecision(false);
	}
}

export class ConfirmModal extends Modal {
	private done = false;

	constructor(
		app: App,
		private title: string,
		private message: string,
		private cta: string,
		private onResult: (ok: boolean) => void,
	) {
		super(app);
	}

	onOpen(): void {
		this.titleEl.setText(this.title);
		this.contentEl.createEl("p", { text: this.message });
		new Setting(this.contentEl)
			.addButton((b) => b.setButtonText("Cancel").onClick(() => this.finish(false)))
			.addButton((b) => b.setButtonText(this.cta).setWarning().onClick(() => this.finish(true)));
	}

	private finish(ok: boolean) {
		this.done = true;
		this.close();
		this.onResult(ok);
	}

	onClose(): void {
		this.contentEl.empty();
		if (!this.done) this.onResult(false);
	}
}
