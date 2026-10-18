import type { Concept, Relationship } from "./links";

export type LLMProvider = "groq" | "ollama";
export type EmbeddingProvider = "gemini" | "ollama";

export interface ConstellationSettings {
	llmProvider: LLMProvider;
	/** One or more Groq keys, separated by commas or new lines (rotated on rate limits). */
	groqApiKeys: string;
	groqModelExtraction: string;
	groqModelVerification: string;
	groqModelRelationship: string;
	reasoningEffort: "" | "low" | "medium" | "high";
	embeddingProvider: EmbeddingProvider;
	geminiApiKey: string;
	geminiEmbeddingModel: string;
	/** Gemini output dimensionality; 0 means the model default. */
	geminiEmbeddingDimension: number;
	ollamaBaseUrl: string;
	ollamaLlmModel: string;
	ollamaEmbeddingModel: string;
	maxLinksPerNote: number;
	annotateLinks: boolean;
	backlinks: boolean;
	tagsInFrontmatter: boolean;
	excludedFolders: string;
	concurrency: number;
	maxOutputTokens: number;
	maxNoteChars: number;
	maxBackups: number;
}

export const DEFAULT_SETTINGS: ConstellationSettings = {
	llmProvider: "groq",
	groqApiKeys: "",
	groqModelExtraction: "openai/gpt-oss-120b",
	groqModelVerification: "openai/gpt-oss-120b",
	groqModelRelationship: "openai/gpt-oss-120b",
	reasoningEffort: "low",
	embeddingProvider: "gemini",
	geminiApiKey: "",
	geminiEmbeddingModel: "models/gemini-embedding-2",
	geminiEmbeddingDimension: 768,
	ollamaBaseUrl: "http://localhost:11434",
	ollamaLlmModel: "llama3.1:8b",
	ollamaEmbeddingModel: "nomic-embed-text",
	maxLinksPerNote: 5,
	annotateLinks: false,
	backlinks: false,
	tagsInFrontmatter: false,
	excludedFolders: "",
	concurrency: 2,
	maxOutputTokens: 2048,
	maxNoteChars: 12000,
	maxBackups: 10,
};

/** Persisted between runs (plugin folder, cache.json). */
export interface Cache {
	files: Record<string, { hash: string }>;
	concepts: Concept[];
	links: Relationship[];
	tags: Record<string, string[]>;
	/** path -> target paths actually rendered into the note last time. */
	rendered: Record<string, string[]>;
	/** Frontmatter tags Constellation added; only these are ever removed. */
	frontmatterTags: Record<string, string[]>;
	settings: RenderSettings | null;
	/** Notes extracted by an interrupted run (or whose embedding failed) that still need linking. */
	pending: string[];
	lastReport: RunReport | null;
}

export interface RenderSettings {
	annotate: boolean;
	tagsInFrontmatter: boolean;
	maxLinks: number;
	backlinks: boolean;
}

export function emptyCache(): Cache {
	return {
		files: {},
		concepts: [],
		links: [],
		tags: {},
		rendered: {},
		frontmatterTags: {},
		settings: null,
		pending: [],
		lastReport: null,
	};
}

export interface RunReport {
	startedAt: string;
	durationS: number;
	dryRun: boolean;
	scope: "vault" | "note";
	notesTotal: number;
	notesProcessed: number;
	notesFailed: string[];
	notesWritten: number;
	linksTotal: number;
	linksNew: number;
	linksRemoved: number;
	llmCalls: number;
	inputTokens: number;
	outputTokens: number;
	embeddingCalls: number;
	skippedDueToEdit: string[];
	backupId: string | null;
}

export const STAGES = [
	"Vault reader",
	"Concept extractor",
	"Concept verifier",
	"Relationship extractor",
	"Relationship verifier",
	"Link writer",
	"Summary reporter",
] as const;

export type StageStatus = "pending" | "running" | "done" | "skipped" | "error";

export interface ProgressEvent {
	stage: number;
	status: StageStatus;
	detail?: string;
	done?: number;
	total?: number;
}

export interface Usage {
	llmCalls: number;
	inputTokens: number;
	outputTokens: number;
	embeddingCalls: number;
}

export function parseList(value: string): string[] {
	return (value || "")
		.split(/[\n,]/)
		.map((s) => s.trim())
		.filter(Boolean);
}
