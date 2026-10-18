// LLM + embedding clients. HTTP goes through an injected function (Obsidian's
// requestUrl in the plugin, which avoids CORS) so this file stays testable.
// Retry behaviour follows async_invoke_with_retry in constellation/core/utils.py.

import { ItemSchema, LLMResponseError, ParsedOutput, parseLLMOutput, schemaHint } from "./json";
import { fill, PromptTemplate } from "./prompts";
import { ConstellationSettings, parseList, Usage } from "./types";

export interface HttpRequest {
	url: string;
	method: "GET" | "POST";
	headers?: Record<string, string>;
	body?: string;
}

export interface HttpResponse {
	status: number;
	text: string;
	headers: Record<string, string>;
}

export type Http = (req: HttpRequest) => Promise<HttpResponse>;

export class HttpError extends Error {
	constructor(public status: number, message: string, public retryAfterS?: number) {
		super(message);
		this.name = "HttpError";
	}
}

export type Logger = (msg: string) => void;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Models whose reasoning tokens count against max_tokens; without an effort
// setting they can spend the whole budget thinking and return nothing.
const REASONING_MODEL_MARKERS = ["gpt-oss", "qwen3", "deepseek-r1"];

function retryAfter(headers: Record<string, string>): number | undefined {
	const raw = Object.entries(headers).find(([k]) => k.toLowerCase() === "retry-after")?.[1];
	const n = Number(raw);
	return Number.isFinite(n) && n > 0 ? n : undefined;
}

async function postJson(http: Http, url: string, body: unknown, headers: Record<string, string> = {}): Promise<any> {
	const res = await http({
		url,
		method: "POST",
		headers: { "Content-Type": "application/json", ...headers },
		body: JSON.stringify(body),
	});
	if (res.status < 200 || res.status >= 300) {
		let detail = res.text;
		try {
			const parsed = JSON.parse(res.text);
			detail = parsed?.error?.message ?? parsed?.error ?? res.text;
		} catch {
			/* keep raw text */
		}
		throw new HttpError(res.status, `HTTP ${res.status}: ${String(detail).slice(0, 400)}`, retryAfter(res.headers));
	}
	try {
		return JSON.parse(res.text);
	} catch {
		throw new LLMResponseError(`non-JSON HTTP body: ${res.text.slice(0, 200)}`);
	}
}

export type Stage = "extraction" | "verification" | "relationship";

export interface ChatMessage {
	role: "system" | "user";
	content: string;
}

const PARSE_MARKERS = ["model response unusable", "json_validate_failed", "tool_use_failed", "failed to generate json"];

/** Chat client with key rotation, 413 shrinking and raw-mode fallback. */
export class LLMClient {
	private keyIndex = 0;
	private keys: string[];

	constructor(
		private http: Http,
		private settings: ConstellationSettings,
		private usage: Usage,
		private log: Logger,
	) {
		this.keys = parseList(settings.groqApiKeys);
	}

	private model(stage: Stage): string {
		const s = this.settings;
		if (s.llmProvider === "ollama") return s.ollamaLlmModel;
		return stage === "extraction"
			? s.groqModelExtraction
			: stage === "verification"
				? s.groqModelVerification
				: s.groqModelRelationship;
	}

	/** One raw completion. jsonMode asks the provider to constrain output to JSON. */
	async complete(stage: Stage, messages: ChatMessage[], jsonMode: boolean, keyIndex: number): Promise<string> {
		const s = this.settings;
		const model = this.model(stage);
		this.usage.llmCalls++;
		if (s.llmProvider === "ollama") {
			const data = await postJson(this.http, `${s.ollamaBaseUrl.replace(/\/+$/, "")}/api/chat`, {
				model,
				messages,
				stream: false,
				...(jsonMode ? { format: "json" } : {}),
				options: { temperature: 0, num_predict: s.maxOutputTokens },
			});
			this.usage.inputTokens += Number(data?.prompt_eval_count) || 0;
			this.usage.outputTokens += Number(data?.eval_count) || 0;
			const text = String(data?.message?.content ?? "");
			if (!text.trim()) throw new LLMResponseError(`empty content (done_reason=${data?.done_reason})`);
			return text;
		}

		if (!this.keys.length) throw new Error("No Groq API key configured. Add one in Settings → Constellation.");
		const body: Record<string, unknown> = {
			model,
			messages,
			temperature: 0,
			max_tokens: s.maxOutputTokens,
		};
		if (jsonMode) body.response_format = { type: "json_object" };
		if (s.reasoningEffort && REASONING_MODEL_MARKERS.some((m) => model.toLowerCase().includes(m))) {
			body.reasoning_effort = s.reasoningEffort;
		}
		const data = await postJson(this.http, "https://api.groq.com/openai/v1/chat/completions", body, {
			Authorization: `Bearer ${this.keys[keyIndex % this.keys.length]}`,
		});
		this.usage.inputTokens += Number(data?.usage?.prompt_tokens) || 0;
		this.usage.outputTokens += Number(data?.usage?.completion_tokens) || 0;
		const choice = data?.choices?.[0];
		const text = String(choice?.message?.content ?? "");
		if (!text.trim()) {
			const hint =
				choice?.finish_reason === "length"
					? " The output budget was spent before any answer was emitted -- raise max output tokens or set reasoning effort to low."
					: "";
			throw new LLMResponseError(`empty content (finish_reason=${choice?.finish_reason}).${hint}`);
		}
		return text;
	}

	/**
	 * Fills a prompt, calls the model and parses the reply into `schema`,
	 * retrying with backoff. `shrink(limit)` rebuilds the inputs with a smaller
	 * note-size cap after a 413 (request too large).
	 */
	async invoke<T>(
		stage: Stage,
		prompt: PromptTemplate,
		schema: ItemSchema<T>,
		buildInputs: (noteCharLimit: number) => Record<string, string>,
		initialLimit: number,
	): Promise<ParsedOutput<T>> {
		const maxRetries = Math.max(this.keys.length, 5);
		let limit = initialLimit;
		let parseFailures = 0;
		let lastError: unknown;
		for (let attempt = 0; attempt < maxRetries; attempt++) {
			const keyUsed = this.keyIndex;
			// After repeated JSON-mode rejections, ask for plain text and repair it ourselves.
			const jsonMode = parseFailures < 3;
			try {
				const inputs = buildInputs(limit);
				const messages: ChatMessage[] = [
					{ role: "system", content: prompt.system },
					{ role: "user", content: fill(prompt.user, inputs) },
					{ role: "system", content: schemaHint(schema as ItemSchema<unknown>) },
				];
				const text = await this.complete(stage, messages, jsonMode, keyUsed);
				const parsed = parseLLMOutput(text, schema);
				if (parsed.dropped) this.log(`  Dropped ${parsed.dropped} malformed item(s) from the LLM response.`);
				return parsed;
			} catch (e) {
				lastError = e;
				const msg = String((e as Error)?.message ?? e).toLowerCase();
				const status = e instanceof HttpError ? e.status : 0;
				const last = attempt === maxRetries - 1;
				if (status === 429 || msg.includes("rate limit")) {
					if (this.keys.length > 1) {
						if (this.keyIndex === keyUsed) this.keyIndex = (this.keyIndex + 1) % this.keys.length;
						this.log(`  Rate limit hit; switching to key ${this.keyIndex + 1}.`);
						if (!last) continue;
					}
					if (!last) {
						const wait = (e as HttpError).retryAfterS ?? 2 * 2 ** attempt;
						this.log(`  Rate limited; retrying in ${Math.ceil(wait)}s...`);
						await sleep(Math.min(wait, 60) * 1000);
						continue;
					}
				} else if (status === 413 || msg.includes("too large")) {
					const next = Math.max(Math.floor(limit / 2), 4000);
					if (next < limit && !last) {
						this.log(`  Request too large; retrying with ${next} chars.`);
						limit = next;
						continue;
					}
				} else if (status === 400 || PARSE_MARKERS.some((m) => msg.includes(m))) {
					parseFailures++;
					this.log(`  Model output rejected (attempt ${attempt + 1}): ${msg.slice(0, 300)}`);
					if (this.keys.length > 1) this.keyIndex = (this.keyIndex + 1) % this.keys.length;
					if (!last) {
						await sleep(1000);
						continue;
					}
				} else if (status === 401 || status === 403 || status === 404) {
					throw e; // bad key or model: retrying cannot help
				}
				if (last) break;
				const delay = 2 * 2 ** attempt;
				this.log(`  Attempt ${attempt + 1} failed: ${msg.slice(0, 300)}; retrying in ${delay}s...`);
				await sleep(delay * 1000);
			}
		}
		throw lastError instanceof Error ? lastError : new Error(String(lastError));
	}
}

export interface Embedder {
	provider: string;
	model: string;
	embed(texts: string[]): Promise<number[][]>;
}

export function createEmbedder(http: Http, settings: ConstellationSettings, usage: Usage, log: Logger): Embedder {
	if (settings.embeddingProvider === "ollama") {
		const model = settings.ollamaEmbeddingModel;
		return {
			provider: "ollama",
			model,
			embed: (texts) =>
				inBatches(texts, 50, async (chunk) => {
					usage.embeddingCalls++;
					const data = await postJson(http, `${settings.ollamaBaseUrl.replace(/\/+$/, "")}/api/embed`, {
						model,
						input: chunk,
					});
					return checkVectors(data?.embeddings, chunk.length);
				}, log),
		};
	}
	const model = settings.geminiEmbeddingModel.startsWith("models/")
		? settings.geminiEmbeddingModel
		: `models/${settings.geminiEmbeddingModel}`;
	const dim = Number(settings.geminiEmbeddingDimension) || 0;
	return {
		provider: "gemini",
		// Dimension is part of the identity: changing it must trigger a re-embed.
		model: dim ? `${model}@${dim}` : model,
		embed: (texts) =>
			inBatches(texts, 50, async (chunk) => {
				if (!settings.geminiApiKey) throw new Error("No Gemini API key configured. Add one in Settings → Constellation.");
				usage.embeddingCalls++;
				const data = await postJson(
					http,
					`https://generativelanguage.googleapis.com/v1beta/${model}:batchEmbedContents`,
					{
						requests: chunk.map((text) => ({
							model,
							content: { parts: [{ text }] },
							...(dim ? { outputDimensionality: dim } : {}),
						})),
					},
					{ "x-goog-api-key": settings.geminiApiKey },
				);
				return checkVectors((data?.embeddings ?? []).map((e: { values: number[] }) => e?.values), chunk.length);
			}, log),
	};
}

function checkVectors(vectors: unknown, expected: number): number[][] {
	if (!Array.isArray(vectors) || vectors.length !== expected || !vectors.every((v) => Array.isArray(v) && v.length)) {
		throw new Error(`embedding response held ${Array.isArray(vectors) ? vectors.length : 0} vectors, expected ${expected}`);
	}
	return vectors as number[][];
}

async function inBatches(
	texts: string[],
	size: number,
	call: (chunk: string[]) => Promise<number[][]>,
	log: Logger,
): Promise<number[][]> {
	const out: number[][] = [];
	for (let i = 0; i < texts.length; i += size) {
		const chunk = texts.slice(i, i + size);
		for (let attempt = 0; ; attempt++) {
			try {
				out.push(...(await call(chunk)));
				break;
			} catch (e) {
				const status = e instanceof HttpError ? e.status : 0;
				if (attempt >= 4 || status === 400 || status === 401 || status === 403 || status === 404) throw e;
				const wait = (e as HttpError).retryAfterS ?? 2 * 2 ** attempt;
				log(`  Embedding request failed (${String((e as Error).message).slice(0, 160)}); retrying in ${wait}s...`);
				await sleep(Math.min(wait, 60) * 1000);
			}
		}
	}
	return out;
}

/** Runs `fn` over items with at most `limit` in flight; never rejects. */
export async function mapPool<T, R>(
	items: T[],
	limit: number,
	fn: (item: T, index: number) => Promise<R>,
	onDone?: (completed: number) => void,
): Promise<(R | Error)[]> {
	const results: (R | Error)[] = new Array(items.length);
	let next = 0;
	let completed = 0;
	const worker = async () => {
		while (next < items.length) {
			const i = next++;
			try {
				results[i] = await fn(items[i], i);
			} catch (e) {
				results[i] = e instanceof Error ? e : new Error(String(e));
			}
			onDone?.(++completed);
		}
	};
	await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
	return results;
}
