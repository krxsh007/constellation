"""API keys, model names and the rotating Google embedding client.

Nothing in here prompts for input at import time -- importing this module must
stay side-effect free enough to be safe inside a Streamlit script run. Missing
keys are supplied by the caller through ``set_groq_keys`` / ``set_google_keys``.
"""

import os
import time
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from constellation.core import usage
from constellation.core.log import log

load_dotenv()

# Groq retired these on 2026-08-16; map old IDs so stale .env/sidebar values still work.
_DEPRECATED_LLM_MODELS = {
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "llama-3.1-8b-instant": "openai/gpt-oss-20b",
}


def _resolve_llm_model(name: str) -> str:
    return _DEPRECATED_LLM_MODELS.get(name.strip(), name)


# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
LLM_MODEL = _resolve_llm_model(os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
LLM_EXTRACTION = _resolve_llm_model(os.getenv("LLM_EXTRACTION", LLM_MODEL))
LLM_VERIFICATION = _resolve_llm_model(os.getenv("LLM_VERIFICATION", LLM_MODEL))
LLM_RELATIONSHIP = _resolve_llm_model(os.getenv("LLM_RELATIONSHIP", LLM_MODEL))

# Groq free/on_demand tier reserves input + max_tokens against an ~8K TPM bucket
# per request, so these two have to be budgeted together:
#   (LLM_MAX_NOTE_CHARS / 4) + LLM_MAX_OUTPUT_TOKENS  must stay under ~8000.
# The gpt-oss models are reasoning models -- their reasoning tokens are billed
# against max_tokens, so an output budget that looks generous for plain JSON can
# be consumed entirely by reasoning, leaving an empty completion.
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
LLM_MAX_NOTE_CHARS = int(os.getenv("LLM_MAX_NOTE_CHARS", "12000"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "2"))

# Reasoning effort for gpt-oss style models. "low" keeps the reasoning budget
# small so the answer itself fits inside LLM_MAX_OUTPUT_TOKENS. Set to "" to
# leave the parameter off the request entirely.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low").strip()

# Maximum wikilinks written to each note (0 or negative disables cap).
MAX_LINKS_PER_NOTE = int(os.getenv("MAX_LINKS_PER_NOTE", "5"))

# Whether to append reasoning to wikilinks: "- [[Target]] — reason"
ANNOTATE_LINKS = os.getenv("ANNOTATE_LINKS", "false").strip().lower() in ("true", "1", "yes")

# Vault folders to ignore (e.g. Templates, Archive, Daily Notes)
def _parse_list_env(val: str) -> List[str]:
    return [item.strip() for item in val.split(",") if item.strip()]

EXCLUDED_DIRS: List[str] = _parse_list_env(os.getenv("EXCLUDED_DIRS", ""))


def _dedupe(keys: List[str]) -> List[str]:
    seen = set()
    unique = []
    for key in keys:
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _keys_from_env(needle: str) -> List[str]:
    return _dedupe([v for k, v in os.environ.items() if needle in k])


# Mutated in place (never rebound) so modules that did ``from core.config import
# GROQ_API_KEYS`` keep seeing updates.
GROQ_API_KEYS: List[str] = _keys_from_env("GROQ_API_KEY")
GOOGLE_API_KEYS: List[str] = _keys_from_env("GOOGLE_API_KEY")


def set_groq_keys(keys: List[str]) -> None:
    GROQ_API_KEYS[:] = _dedupe(keys)


def set_google_keys(keys: List[str]) -> None:
    global current_google_key_index
    GOOGLE_API_KEYS[:] = _dedupe(keys)
    current_google_key_index = 0


def set_models(extraction: str = None, verification: str = None, relationship: str = None) -> None:
    global LLM_EXTRACTION, LLM_VERIFICATION, LLM_RELATIONSHIP
    if extraction:
        LLM_EXTRACTION = _resolve_llm_model(extraction)
    if verification:
        LLM_VERIFICATION = _resolve_llm_model(verification)
    if relationship:
        LLM_RELATIONSHIP = _resolve_llm_model(relationship)


def set_excluded_dirs(dirs) -> None:
    global EXCLUDED_DIRS
    if isinstance(dirs, str):
        EXCLUDED_DIRS = _parse_list_env(dirs)
    else:
        EXCLUDED_DIRS = [str(d).strip() for d in dirs if str(d).strip()]


def set_max_links(val: int) -> None:
    global MAX_LINKS_PER_NOTE
    MAX_LINKS_PER_NOTE = int(val)


def set_annotate_links(val: bool) -> None:
    global ANNOTATE_LINKS
    ANNOTATE_LINKS = bool(val)


current_google_key_index = 0

# Texts per embedding request, and the first retry delay (doubles each attempt).
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
EMBED_BASE_DELAY = 2.0


class RotatingGoogleEmbeddings(Embeddings):
    """Google embeddings with retry, exponential backoff and API key rotation."""

    provider = "google"

    def __init__(self, model_name: str = None):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name or EMBEDDING_MODEL

    def _get_embedding_instance(self):
        if not GOOGLE_API_KEYS:
            raise RuntimeError(
                "No Google API key configured. Add GOOGLE_API_KEY_1 to your .env "
                "or paste a key in the sidebar."
            )
        key = GOOGLE_API_KEYS[current_google_key_index]
        return GoogleGenerativeAIEmbeddings(model=self.model_name, google_api_key=key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds in batches of EMBED_BATCH_SIZE; one request per batch instead of
        one per text.

        Some langchain-google-genai versions return the wrong number of vectors
        for a batch, so a batch that errors or comes back the wrong length falls
        back to one request per text -- correctness first, speed when possible.
        """
        texts = list(texts)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            chunk = texts[start:start + EMBED_BATCH_SIZE]
            try:
                result = self._with_retry(lambda inst: inst.embed_documents(chunk))
                if len(result) != len(chunk):
                    raise ValueError(f"expected {len(chunk)} vectors, got {len(result)}")
                vectors.extend(result)
            except Exception as e:
                log(f"  Batch embedding failed ({str(e)[:160]}). Falling back to one request per text...")
                vectors.extend(self.embed_query(text) for text in chunk)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._with_retry(lambda inst: inst.embed_query(text))

    def _with_retry(self, call):
        """Runs call(client) with key rotation on rate limits and exponential backoff.

        Blocking (time.sleep) by design: callers in async nodes run embedding in a
        worker thread via asyncio.to_thread, so this never stalls the event loop.
        """
        global current_google_key_index
        max_retries = max(len(GOOGLE_API_KEYS), 5)
        base_delay = EMBED_BASE_DELAY

        for attempt in range(max_retries):
            try:
                result = call(self._get_embedding_instance())
                usage.record_embedding_call()
                return result
            except Exception as e:
                err_msg = str(e).lower()
                if "rate limit" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg:
                    if len(GOOGLE_API_KEYS) > 1:
                        log(f"  Google rate limit on key {current_google_key_index + 1}. Rotating...")
                        current_google_key_index = (current_google_key_index + 1) % len(GOOGLE_API_KEYS)
                        if attempt < max_retries - 1:
                            continue
                    else:
                        log("  Google rate limit hit. No alternative keys to switch to.")

                if attempt == max_retries - 1:
                    raise

                delay = base_delay * (2 ** attempt)
                log(f"  Embedding failed (attempt {attempt + 1}): {str(e)[:160]}. Retrying in {delay:.0f}s...")
                time.sleep(delay)


embeddings = RotatingGoogleEmbeddings()


def describe_config() -> str:
    excluded_str = ", ".join(EXCLUDED_DIRS) if EXCLUDED_DIRS else "none"
    return (
        f"Extraction: {LLM_EXTRACTION} | Verification: {LLM_VERIFICATION} | "
        f"Relationship: {LLM_RELATIONSHIP} | Embeddings: {EMBEDDING_MODEL}\n"
        f"Output budget: {LLM_MAX_OUTPUT_TOKENS} tokens | Note cap: {LLM_MAX_NOTE_CHARS} chars"
        f"{f' | Reasoning effort: {LLM_REASONING_EFFORT}' if LLM_REASONING_EFFORT else ''}\n"
        f"Max links/note: {MAX_LINKS_PER_NOTE} | Annotate: {ANNOTATE_LINKS} | Excluded dirs: {excluded_str}\n"
        f"{len(GROQ_API_KEYS)} Groq key(s), {len(GOOGLE_API_KEYS)} Google key(s) loaded for rotation."
    )


def missing_keys() -> List[str]:
    missing = []
    if not GROQ_API_KEYS:
        missing.append("Groq")
    if not GOOGLE_API_KEYS:
        missing.append("Google")
    return missing
