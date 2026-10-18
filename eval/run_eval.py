"""Link-quality evaluation and stage ablations for the Constellation pipeline.

Runs the real LangGraph pipeline in dry-run mode on a temporary copy of
``eval/data/vault`` once per configuration, compares the links it would write with
the hand-labelled pairs in ``eval/data/gold_links.json`` and reports precision,
recall, F1, tag counts, LLM calls, tokens, time and estimated cost.

Every LLM and embedding response is recorded to a cassette so the evaluation
can be replayed offline and deterministically:

    python3 eval/run_eval.py --replay              # offline, from eval/cassettes/
    python3 eval/run_eval.py --record              # live calls, needs API keys
    python3 eval/run_eval.py --replay --write      # also refresh RESULTS.md
    python3 eval/run_eval.py --replay --check-baseline

Cassettes: ``eval/cassettes/llm/<config>.json`` holds that configuration's LLM
responses (keyed by a hash of prompt, model, schema and inputs, with the token
usage and latency of the original call). Embedding vectors are stored once, in
``eval/cassettes/embeddings.json``, keyed by text -- they are the same across
configurations and too large to duplicate. Vectors are stored as float16; the
pipeline sees the float16 values in record mode too, so record and replay rank
neighbours identically.

While recording, a request already present in any cassette is reused rather
than sent again (extraction prompts are identical across most configurations),
so recording every configuration costs little more than recording one.
"""

import argparse
import asyncio
import base64
import contextvars
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

# Never send traces anywhere: replay must be fully offline, and a local .env
# may turn LangSmith tracing on (load_dotenv does not override these).
for _var in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING"):
    os.environ[_var] = "false"

import numpy as np  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from constellation.core import config, prompts, usage, utils  # noqa: E402
from constellation.core import log as logmod  # noqa: E402
from constellation.nodes import io  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
VAULT_DIR = DATA_DIR / "vault"
GOLD_PATH = DATA_DIR / "gold_links.json"
CASSETTE_DIR = EVAL_DIR / "cassettes" / "llm"
EMBEDDINGS_CASSETTE = EVAL_DIR / "cassettes" / "embeddings.json"
RESULTS_MD = EVAL_DIR / "RESULTS.md"
RESULTS_JSON = EVAL_DIR / "results.json"
BASELINE_PATH = EVAL_DIR / "baseline.json"

# Pinned so results do not depend on whatever a local .env overrides.
PINNED = {
    "llm_model": "openai/gpt-oss-120b",
    "embedding_model": "models/gemini-embedding-2",
    "max_output_tokens": 2048,
    "max_note_chars": 12000,
    "reasoning_effort": "low",
    "concurrency": 2,
}

# USD per million tokens, Groq list price for openai/gpt-oss-120b (Oct 2026).
# Embeddings (Gemini) are not included; at this size they are within the free tier.
PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60

# name -> (description, settings). Settings not given keep the pipeline default.
CONFIGS = {
    "default": ("Full pipeline (k=10, 5 links/note)", {}),
    "no-concept-verify": ("Skip concept verification", {"skip_concept_verification": True}),
    "no-relation-verify": ("Skip relationship verification", {"skip_relationship_verification": True}),
    "k5": ("Retrieval k=5", {"retrieval_k": 5}),
    "k20": ("Retrieval k=20", {"retrieval_k": 20}),
    "max-links-3": ("MAX_LINKS_PER_NOTE=3", {"max_links": 3}),
    "max-links-10": ("MAX_LINKS_PER_NOTE=10", {"max_links": 10}),
    "retrieval-only": ("Embedding top-k only, no relationship LLM (5 links/note)", {"relationship_mode": "retrieval"}),
    "retrieval-only-2": ("Embedding top-k only, 2 links/note", {"relationship_mode": "retrieval", "max_links": 2}),
}

DEFAULT_SETTINGS = {
    "retrieval_k": 10,
    "skip_concept_verification": False,
    "skip_relationship_verification": False,
    "relationship_mode": "llm",
    "max_links": 5,
}


class CassetteMiss(RuntimeError):
    """Replay asked for a response that was never recorded."""


# --- Cassettes ------------------------------------------------------------------


def _prompt_name(prompt) -> str:
    for name in dir(prompts):
        if getattr(prompts, name) is prompt:
            return name
    return type(prompt).__name__


def request_key(prompt, model_name: str, schema, inputs: dict) -> str:
    payload = json.dumps(
        {"prompt": _prompt_name(prompt), "model": model_name, "schema": schema.__name__, "inputs": inputs},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _text_key(kind: str, model: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\x1f{model}\x1f{text}".encode("utf-8")).hexdigest()[:32]


def _encode_vector(vec) -> str:
    return base64.b64encode(np.asarray(vec, dtype=np.float16).tobytes()).decode("ascii")


def _decode_vector(data: str) -> list:
    return np.frombuffer(base64.b64decode(data), dtype=np.float16).astype(np.float32).tolist()


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_llm_pool() -> dict:
    """Every recorded LLM response across all cassettes (record-mode reuse)."""
    pool = {}
    for path in sorted(CASSETTE_DIR.glob("*.json")):
        pool.update(_load_json(path, {}).get("llm", {}))
    return pool


# Token usage of the LLM responses produced inside one recorded request.
_call_usage = contextvars.ContextVar("eval_call_usage", default=None)


def _usage_of(message) -> tuple:
    meta = getattr(message, "usage_metadata", None) or {}
    if not meta:
        token_usage = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
        meta = {
            "input_tokens": token_usage.get("prompt_tokens", 0),
            "output_tokens": token_usage.get("completion_tokens", 0),
        }
    return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)


class LLMRecorder:
    """Stands in for utils.async_invoke_with_retry: serves recorded responses,
    and in record mode makes (and records) the real call on a miss."""

    def __init__(self, mode: str, entries: dict, pool: dict = None):
        self.mode = mode
        self.entries = entries  # what this configuration used
        self.pool = pool or {}  # reusable responses from any cassette
        self.real = utils.async_invoke_with_retry
        self.real_record = usage.record_llm_response
        self.live_calls = 0

    def record_usage(self, message) -> None:
        bucket = _call_usage.get()
        if bucket is not None:
            bucket.append(list(_usage_of(message)))
        self.real_record(message)

    async def __call__(self, prompt, model_name, schema, inputs, max_retries=None, base_delay=2.0):
        key = request_key(prompt, model_name, schema, inputs)
        entry = self.entries.get(key) or self.pool.get(key)
        if entry is None:
            if self.mode != "record":
                raise CassetteMiss(f"No recorded response for {_prompt_name(prompt)} ({key}); re-record.")
            entry = await self._record(key, prompt, model_name, schema, inputs, max_retries, base_delay)
        else:
            for inp, out in entry["usage"]:
                self.real_record(_Usage(inp, out))
        self.entries[key] = entry
        if entry.get("error"):
            raise RuntimeError(entry["error"])
        return schema.model_validate(entry["result"])

    async def _record(self, key, prompt, model_name, schema, inputs, max_retries, base_delay):
        bucket = []
        token = _call_usage.set(bucket)
        started = time.perf_counter()
        result, error = None, None
        try:
            result = await self.real(prompt, model_name, schema, inputs, max_retries, base_delay)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            _call_usage.reset(token)
        self.live_calls += 1
        return {
            "prompt": _prompt_name(prompt),
            "model": model_name,
            "result": result.model_dump() if result is not None else None,
            "error": error,
            "usage": bucket,
            "latency_s": round(time.perf_counter() - started, 3),
        }


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


class EmbeddingRecorder(Embeddings):
    """Embeddings served from the shared cassette; live (and recorded) on a miss
    in record mode. Counts the requests a standalone run would make."""

    def __init__(self, mode: str, store: dict, real):
        self.mode = mode
        self.store = store
        self.real = real
        self.provider = getattr(real, "provider", "google")
        self.model_name = getattr(real, "model_name", PINNED["embedding_model"])
        self.requests = 0
        self.live_requests = 0
        self.dirty = False

    def _lookup(self, kind: str, texts: list, fetch) -> list:
        keys = [_text_key(kind, self.model_name, t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.store]
        if missing:
            if self.mode != "record":
                raise CassetteMiss(f"{len(missing)} {kind} embedding(s) were never recorded; re-record.")
            vectors = fetch([t for _, t in missing])
            self.live_requests += 1
            for (k, _), vec in zip(missing, vectors):
                self.store[k] = _encode_vector(vec)
            self.dirty = True
        return [_decode_vector(self.store[k]) for k in keys]

    def _fetch_live(self, kind: str, texts: list) -> list:
        """Same Google calls the pipeline makes (embed_documents / embed_query),
        but paced for the free tier, which counts every text as one request
        against a per-minute quota: small chunks, rotate keys on 429, and wait
        out the window when every key is exhausted."""
        if getattr(self.real, "provider", "google") != "google":
            return self.real.embed_documents(texts) if kind == "doc" else [self.real.embed_query(t) for t in texts]
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        keys = list(config.GOOGLE_API_KEYS)
        vectors, start, key_index, exhausted, waits = [], 0, 0, 0, 0
        while start < len(texts):
            chunk = texts[start : start + 10]
            client = GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=keys[key_index])
            try:
                if kind == "doc":
                    got = client.embed_documents(chunk)
                else:
                    got = [client.embed_query(t) for t in chunk]
            except Exception as e:
                if "429" not in str(e) and "resource_exhausted" not in str(e).lower():
                    raise
                key_index = (key_index + 1) % len(keys)
                exhausted += 1
                if exhausted >= len(keys):
                    waits += 1
                    if waits > 20:
                        raise RuntimeError("Embedding quota still exhausted after 10 minutes (daily limit?).") from e
                    print("  embedding quota exhausted on every key; waiting 30s", flush=True)
                    time.sleep(30)
                    exhausted = 0
                continue
            vectors.extend(got)
            exhausted = 0
            start += len(chunk)
        return vectors

    def embed_documents(self, texts):
        texts = list(texts)
        batch = max(config.EMBED_BATCH_SIZE, 1)
        self.requests += -(-len(texts) // batch)
        return self._lookup("doc", texts, lambda ts: self._fetch_live("doc", ts))

    def embed_query(self, text):
        self.requests += 1
        return self._lookup("query", [text], lambda ts: self._fetch_live("query", ts))[0]


# --- Scoring --------------------------------------------------------------------


def _pair(a: str, b: str) -> tuple:
    a, b = (t[:-3] if t.lower().endswith(".md") else t for t in (a, b))
    return tuple(sorted((a, b)))


def load_gold(path: Path = GOLD_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: {_pair(a, b) for a, b in data.get(k, [])} for k in ("required", "optional", "distractors")}


def predicted_pairs(state: dict) -> set:
    """Undirected pairs of the links the run would actually render."""
    titles = {n["title"] for n in state.get("notes") or []}
    effective = io.build_effective_links(state.get("links") or [], existing_titles=titles)
    return {_pair(src, item["target"]) for src, items in effective.items() for item in items}


def score(pred: set, gold: dict) -> dict:
    required, optional = gold["required"], gold["optional"]
    tp = len(pred & required)
    fp = len(pred - required - optional)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(required) if required else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pairs": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": len(required - pred),
        "optional_hits": len(pred & optional),
        "distractor_hits": len(pred & gold["distractors"]),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# --- Running a configuration ----------------------------------------------------


def apply_settings(settings: dict) -> None:
    s = {**DEFAULT_SETTINGS, **settings}
    config.set_providers(llm="groq", embeddings="google")
    config.set_models(PINNED["llm_model"], PINNED["llm_model"], PINNED["llm_model"])
    config.EMBEDDING_MODEL = PINNED["embedding_model"]
    config.LLM_MAX_OUTPUT_TOKENS = PINNED["max_output_tokens"]
    config.LLM_MAX_NOTE_CHARS = PINNED["max_note_chars"]
    config.LLM_REASONING_EFFORT = PINNED["reasoning_effort"]
    config.LLM_CONCURRENCY = PINNED["concurrency"]
    config.set_backlinks(False)
    config.set_annotate_links(False)
    config.set_tags_in_frontmatter(False)
    config.set_excluded_dirs([])
    config.set_max_links(s["max_links"])
    config.set_ablation(
        retrieval_k=s["retrieval_k"],
        skip_concept_verification=s["skip_concept_verification"],
        skip_relationship_verification=s["skip_relationship_verification"],
        relationship_mode=s["relationship_mode"],
    )


_SAVED_NAMES = (
    "LLM_PROVIDER",
    "EMBEDDING_PROVIDER",
    "LLM_EXTRACTION",
    "LLM_VERIFICATION",
    "LLM_RELATIONSHIP",
    "EMBEDDING_MODEL",
    "LLM_MAX_OUTPUT_TOKENS",
    "LLM_MAX_NOTE_CHARS",
    "LLM_REASONING_EFFORT",
    "LLM_CONCURRENCY",
    "BACKLINKS",
    "ANNOTATE_LINKS",
    "TAGS_IN_FRONTMATTER",
    "EXCLUDED_DIRS",
    "MAX_LINKS_PER_NOTE",
    "RETRIEVAL_K",
    "SKIP_CONCEPT_VERIFICATION",
    "SKIP_RELATIONSHIP_VERIFICATION",
    "RELATIONSHIP_MODE",
)


def run_config(
    name: str,
    mode: str,
    gold: dict,
    emb_store: dict,
    pool: dict = None,
    vault_dir: Path = VAULT_DIR,
    cassette_dir: Path = CASSETTE_DIR,
) -> dict:
    """Runs one configuration and returns its metrics row. mode: replay|record."""
    description, settings = CONFIGS[name]
    saved = {n: getattr(config, n) for n in _SAVED_NAMES}
    cassette_path = cassette_dir / f"{name}.json"
    cassette = _load_json(cassette_path, {})
    entries = {} if mode == "record" else dict(cassette.get("llm", {}))
    llm = LLMRecorder(mode, entries, pool if mode == "record" else None)
    try:
        apply_settings(settings)
        emb = EmbeddingRecorder(mode, emb_store, config.embeddings)
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / "vault"
            shutil.copytree(vault_dir, vault)
            patches = [
                mock.patch.object(utils, "async_invoke_with_retry", llm),
                mock.patch.object(usage, "record_llm_response", llm.record_usage),
                mock.patch.object(config, "embeddings", emb),
            ]
            if mode == "replay":
                patches.append(mock.patch.object(config, "missing_keys", return_value=[]))
            for p in patches:
                p.start()
            try:
                started = time.perf_counter()
                state = asyncio.run(main.arun_pipeline(str(vault), dry_run=True))
                wall = time.perf_counter() - started
                counters = usage.snapshot()
                # Scored while this configuration's link cap is still active.
                metrics = score(predicted_pairs(state), gold)
            finally:
                for p in reversed(patches):
                    p.stop()
    finally:
        for n, v in saved.items():
            setattr(config, n, v)

    if mode == "record":
        _dump_json(
            cassette_path,
            {
                "config": name,
                "description": description,
                "settings": settings,
                "pinned": PINNED,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "llm": entries,
            },
        )

    tags = state.get("tags_by_note") or {}
    latency = sum(e.get("latency_s", 0.0) for e in entries.values())
    cost = (counters["input_tokens"] * PRICE_PER_M_INPUT + counters["output_tokens"] * PRICE_PER_M_OUTPUT) / 1e6
    return {
        "config": name,
        "description": description,
        **metrics,
        "notes": len(state.get("notes") or []),
        "failed_notes": len(state.get("failed_notes") or []),
        "tags": sum(len(v) for v in tags.values()),
        "notes_tagged": len(tags),
        "llm_calls": counters["llm_calls"],
        "input_tokens": counters["input_tokens"],
        "output_tokens": counters["output_tokens"],
        "embedding_requests": emb.requests,
        "cost_usd": round(cost, 4),
        # Recorded per-request latency summed and divided by the concurrency: an
        # estimate of the live run's LLM time (replay itself takes seconds).
        "est_live_llm_s": round(latency / max(PINNED["concurrency"], 1), 1),
        "replay_wall_s": round(wall, 2),
        "live_requests": llm.live_calls,
    }


# --- Reporting ------------------------------------------------------------------

_COLUMNS = (
    ("Config", "config"),
    ("Pairs", "pairs"),
    ("TP", "tp"),
    ("FP", "fp"),
    ("Distractor hits", "distractor_hits"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("Tags", "tags"),
    ("LLM calls", "llm_calls"),
    ("Tokens in/out", None),
    ("Est. cost $", "cost_usd"),
    ("Est. live LLM time s", "est_live_llm_s"),
)


def markdown_table(rows: list) -> str:
    lines = ["| " + " | ".join(h for h, _ in _COLUMNS) + " |", "|" + "---|" * len(_COLUMNS)]
    for r in rows:
        cells = []
        for _header, key in _COLUMNS:
            if key is None:
                cells.append(f"{r['input_tokens']:,} / {r['output_tokens']:,}")
            elif key in ("precision", "recall", "f1"):
                cells.append(f"{r[key] * 100:.1f}")
            elif key == "config":
                cells.append(f"`{r[key]}`")
            else:
                cells.append(str(r[key]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


START, END = "<!-- results:start -->", "<!-- results:end -->"


def write_results(rows: list, gold: dict) -> None:
    block = "\n".join(
        [
            START,
            f"Vault: {rows[0]['notes']} notes; gold: {len(gold['required'])} required pairs, "
            f"{len(gold['optional'])} optional, {len(gold['distractors'])} distractors. "
            f"Model `{PINNED['llm_model']}`, embeddings `{PINNED['embedding_model']}`, concurrency "
            f"{PINNED['concurrency']}. Precision/recall/F1 in %.",
            "",
            markdown_table(rows),
            "",
            "| Config | What changes |",
            "|---|---|",
            *[f"| `{r['config']}` | {r['description']} |" for r in rows],
            END,
        ]
    )
    text = (
        RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else f"# Evaluation results\n\n{START}\n{END}\n"
    )
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    RESULTS_MD.write_text(head + block + tail, encoding="utf-8")
    # Replay timing varies run to run; leave it out so an unchanged eval leaves no diff.
    _dump_json(RESULTS_JSON, [{k: v for k, v in r.items() if k != "replay_wall_s"} for r in rows])


def check_baseline(rows: list, tolerance_points: float = 5.0) -> bool:
    baseline = _load_json(BASELINE_PATH, None)
    default = next((r for r in rows if r["config"] == "default"), None)
    if baseline is None or default is None:
        print("Baseline check needs eval/baseline.json and the 'default' config.")
        return False
    floor = baseline["f1"] * 100 - tolerance_points
    ok = default["f1"] * 100 >= floor
    print(
        f"Default F1 {default['f1'] * 100:.1f} vs baseline {baseline['f1'] * 100:.1f} "
        f"(floor {floor:.1f}): {'OK' if ok else 'REGRESSION'}"
    )
    return ok


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", action="store_true", help="Offline replay from eval/cassettes (default).")
    mode.add_argument("--record", action="store_true", help="Make live calls (needs API keys) and record cassettes.")
    parser.add_argument("--configs", default=",".join(CONFIGS), help="Comma-separated configurations to run.")
    parser.add_argument("--write", action="store_true", help="Update eval/RESULTS.md and eval/results.json.")
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Exit 1 if default-config F1 is more than 5 points below eval/baseline.json.",
    )
    parser.add_argument("--update-baseline", action="store_true", help="Store default-config metrics as the baseline.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show pipeline logs.")
    return parser.parse_args(argv)


def main_cli(argv=None) -> int:
    args = parse_args(argv)
    mode = "record" if args.record else "replay"
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        print(f"Unknown config(s): {', '.join(unknown)}. Choose from: {', '.join(CONFIGS)}")
        return 2
    if mode == "record":
        missing = []
        if not config.GROQ_API_KEYS:
            missing.append("GROQ_API_KEY_1")
        if not config.GOOGLE_API_KEYS:
            missing.append("GOOGLE_API_KEY_1")
        if missing:
            print(f"--record needs {', '.join(missing)} in the environment or .env.")
            return 2
    if not args.verbose:
        logmod.set_sink(lambda _msg: None)

    gold = load_gold()
    emb_store = _load_json(EMBEDDINGS_CASSETTE, {}).get("vectors", {})
    pool = load_llm_pool() if mode == "record" else None
    rows = []
    for name in names:
        print(f"[{mode}] {name} ...", flush=True)
        try:
            row = run_config(name, mode, gold, emb_store, pool)
        finally:
            if mode == "record":
                _dump_json(
                    EMBEDDINGS_CASSETTE, {"model": PINNED["embedding_model"], "dtype": "float16", "vectors": emb_store}
                )
        if mode == "record":
            pool.update(_load_json(CASSETTE_DIR / f"{name}.json", {}).get("llm", {}))
        rows.append(row)
        print(
            f"  P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} pairs={row['pairs']} "
            f"llm_calls={row['llm_calls']} live={row['live_requests']} failed_notes={row['failed_notes']}"
        )

    print("\n" + markdown_table(rows))
    if args.write:
        write_results(rows, gold)
        print(f"\nWrote {RESULTS_MD.relative_to(ROOT)} and {RESULTS_JSON.relative_to(ROOT)}")
    if args.update_baseline:
        default = next((r for r in rows if r["config"] == "default"), None)
        if default:
            _dump_json(BASELINE_PATH, {k: default[k] for k in ("precision", "recall", "f1", "pairs", "tp", "fp")})
            print(f"Wrote {BASELINE_PATH.relative_to(ROOT)}")
    if args.check_baseline and not check_baseline(rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
