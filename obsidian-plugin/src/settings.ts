import { App, PluginSettingTab, Setting } from "obsidian";
import type ConstellationPlugin from "./main";
import type { ConstellationSettings } from "./types";

type StringKey = { [K in keyof ConstellationSettings]: ConstellationSettings[K] extends string ? K : never }[keyof ConstellationSettings];
type NumberKey = { [K in keyof ConstellationSettings]: ConstellationSettings[K] extends number ? K : never }[keyof ConstellationSettings];
type BoolKey = { [K in keyof ConstellationSettings]: ConstellationSettings[K] extends boolean ? K : never }[keyof ConstellationSettings];

export class ConstellationSettingTab extends PluginSettingTab {
	constructor(app: App, private plugin: ConstellationPlugin) {
		super(app, plugin);
	}

	private get s() {
		return this.plugin.settings;
	}

	private text(el: HTMLElement, key: StringKey, name: string, desc: string, opts: { secret?: boolean; placeholder?: string } = {}) {
		return new Setting(el).setName(name).setDesc(desc).addText((t) => {
			t.setPlaceholder(opts.placeholder ?? "").setValue(String(this.s[key] ?? ""));
			if (opts.secret) t.inputEl.type = "password";
			t.onChange(async (v) => {
				(this.s[key] as string) = v.trim();
				await this.plugin.saveSettings();
			});
		});
	}

	private number(el: HTMLElement, key: NumberKey, name: string, desc: string, min: number, max: number) {
		return new Setting(el).setName(name).setDesc(desc).addText((t) => {
			t.inputEl.type = "number";
			t.inputEl.min = String(min);
			t.inputEl.max = String(max);
			t.setValue(String(this.s[key]));
			t.onChange(async (v) => {
				const n = Math.round(Number(v));
				if (!Number.isFinite(n)) return;
				(this.s[key] as number) = Math.min(max, Math.max(min, n));
				await this.plugin.saveSettings();
			});
		});
	}

	private toggle(el: HTMLElement, key: BoolKey, name: string, desc: string) {
		return new Setting(el).setName(name).setDesc(desc).addToggle((t) =>
			t.setValue(this.s[key]).onChange(async (v) => {
				(this.s[key] as boolean) = v;
				await this.plugin.saveSettings();
			}),
		);
	}

	display(): void {
		const el = this.containerEl;
		el.empty();

		new Setting(el).setName("Language model").setHeading();
		new Setting(el)
			.setName("Provider")
			.setDesc("Groq (hosted, free tier) or Ollama (local, nothing leaves your machine).")
			.addDropdown((d) =>
				d
					.addOption("groq", "Groq")
					.addOption("ollama", "Ollama")
					.setValue(this.s.llmProvider)
					.onChange(async (v) => {
						this.s.llmProvider = v as ConstellationSettings["llmProvider"];
						await this.plugin.saveSettings();
						this.display();
					}),
			);
		if (this.s.llmProvider === "groq") {
			new Setting(el)
				.setName("Groq API keys")
				.setDesc("One or more keys, one per line. Constellation rotates to the next key on rate limits. Stored in this vault's plugin folder.")
				.addTextArea((t) => {
					t.setPlaceholder("gsk_...").setValue(this.s.groqApiKeys);
					t.inputEl.rows = 2;
					t.inputEl.addClass("constellation-secret");
					t.onChange(async (v) => {
						this.s.groqApiKeys = v.trim();
						await this.plugin.saveSettings();
					});
				});
			this.text(el, "groqModelExtraction", "Extraction model", "Model used to extract concepts and tags.");
			this.text(el, "groqModelVerification", "Verification model", "Model used to verify concepts and links.");
			this.text(el, "groqModelRelationship", "Relationship model", "Model used to propose cross-note links.");
			new Setting(el)
				.setName("Reasoning effort")
				.setDesc("For reasoning models (gpt-oss, qwen3). Low keeps them from spending the output budget thinking.")
				.addDropdown((d) =>
					d
						.addOption("", "Model default")
						.addOption("low", "Low")
						.addOption("medium", "Medium")
						.addOption("high", "High")
						.setValue(this.s.reasoningEffort)
						.onChange(async (v) => {
							this.s.reasoningEffort = v as ConstellationSettings["reasoningEffort"];
							await this.plugin.saveSettings();
						}),
				);
		} else {
			this.text(el, "ollamaLlmModel", "Ollama chat model", "Used for every stage.", { placeholder: "llama3.1:8b" });
		}

		new Setting(el).setName("Embeddings").setHeading();
		new Setting(el)
			.setName("Provider")
			.setDesc("Changing the provider or model re-embeds every concept on the next run (no extra LLM calls).")
			.addDropdown((d) =>
				d
					.addOption("gemini", "Google Gemini")
					.addOption("ollama", "Ollama")
					.setValue(this.s.embeddingProvider)
					.onChange(async (v) => {
						this.s.embeddingProvider = v as ConstellationSettings["embeddingProvider"];
						await this.plugin.saveSettings();
						this.display();
					}),
			);
		if (this.s.embeddingProvider === "gemini") {
			this.text(el, "geminiApiKey", "Gemini API key", "From Google AI Studio. Stored in this vault's plugin folder.", { secret: true });
			this.text(el, "geminiEmbeddingModel", "Embedding model", "Gemini embedding model ID.", { placeholder: "models/gemini-embedding-2" });
			this.number(el, "geminiEmbeddingDimension", "Embedding dimension", "Output dimensionality (0 = model default). Smaller keeps the index file small.", 0, 4096);
		} else {
			this.text(el, "ollamaEmbeddingModel", "Ollama embedding model", "", { placeholder: "nomic-embed-text" });
		}
		if (this.s.llmProvider === "ollama" || this.s.embeddingProvider === "ollama") {
			this.text(el, "ollamaBaseUrl", "Ollama URL", "Where Ollama is listening.", { placeholder: "http://localhost:11434" });
		}

		new Setting(el).setName("Output").setHeading();
		this.number(el, "maxLinksPerNote", "Max links per note", "The highest-scoring links survive the cap (0 = no cap).", 0, 100);
		this.toggle(el, "annotateLinks", "Annotate links", "Append the model's one-line evidence after each link.");
		this.toggle(el, "backlinks", "Backlinks", "When A links to B, also list A under B's Related Links.");
		this.toggle(el, "tagsInFrontmatter", "Tags in frontmatter", "Write tags to the frontmatter `tags` list instead of a ## Tags section. Your own tags are never removed.");
		new Setting(el)
			.setName("Excluded folders")
			.setDesc("One folder per line (or comma separated). Notes inside are never read or written.")
			.addTextArea((t) => {
				t.setPlaceholder("Templates\nArchive/old").setValue(this.s.excludedFolders);
				t.onChange(async (v) => {
					this.s.excludedFolders = v;
					await this.plugin.saveSettings();
				});
			});

		new Setting(el).setName("Advanced").setHeading();
		this.number(el, "concurrency", "Concurrency", "LLM requests in flight at once. Keep low on free tiers.", 1, 16);
		this.number(el, "maxOutputTokens", "Max output tokens", "Per LLM call.", 256, 32768);
		this.number(el, "maxNoteChars", "Max note characters", "Longer notes are truncated before being sent.", 1000, 200000);
		this.number(el, "maxBackups", "Backups to keep", "Snapshots taken before each run, used by \"Undo last run\".", 1, 100);
	}
}
