import asyncio
import time
from typing import List

from langchain_groq import ChatGroq

from core import config
from core.log import log

current_key_index = 0
key_rotation_lock = None


def get_key_rotation_lock():
    global key_rotation_lock
    if key_rotation_lock is None:
        key_rotation_lock = asyncio.Lock()
    return key_rotation_lock


def reset_key_rotation():
    """Called at the start of a run so a fresh Streamlit run starts on key 1."""
    global current_key_index, key_rotation_lock
    current_key_index = 0
    key_rotation_lock = None


def _groq_key(index: int) -> str:
    if not config.GROQ_API_KEYS:
        raise RuntimeError(
            "No Groq API key configured. Add GROQ_API_KEY_1 to your .env "
            "or paste a key in the sidebar."
        )
    return config.GROQ_API_KEYS[index % len(config.GROQ_API_KEYS)]


def _max_retries(override=None) -> int:
    if override is not None:
        return override
    return max(len(config.GROQ_API_KEYS), 5)


def minify_concepts(concepts: List[dict]) -> List[dict]:
    """Strips bulky fields (keywords, related_concepts) to drastically reduce token usage."""
    return [
        {
            "concept_name": c.get("concept_name"),
            "source_note": c.get("source_note"),
            "explanation": c.get("explanation")
        }
        for c in concepts
    ]


def invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries=None, base_delay=2.0):
    """Invoke a chain with exponential backoff, structured outputs, and API key rotation."""
    global current_key_index
    max_retries = _max_retries(max_retries)
    for attempt in range(max_retries):
        try:
            llm = ChatGroq(groq_api_key=_groq_key(current_key_index), model=model_name, max_tokens=1024)
            chain = prompt | llm.with_structured_output(pydantic_schema)
            return chain.invoke(inputs)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg:
                if len(config.GROQ_API_KEYS) > 1:
                    log(f"  Rate limit hit on key {current_key_index + 1}. Switching to next key...")
                    current_key_index = (current_key_index + 1) % len(config.GROQ_API_KEYS)
                    if attempt < max_retries - 1:
                        continue  # Retry immediately with new key
                else:
                    log("  Rate limit hit. No alternative keys to switch to.")
            elif "failed to call a function" in err_msg or "tool_use_failed" in err_msg:
                log(f"  Groq tool parsing failed (attempt {attempt + 1}). Retrying...")
                if attempt < max_retries - 1:
                    continue

            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt

            delay = base_delay * (2 ** attempt)
            log(f"  Attempt {attempt + 1} failed: {str(e)[:200]}")
            log(f"  Retrying in {delay:.0f}s...")
            time.sleep(delay)


async def async_invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries=None, base_delay=2.0):
    """Async wrapper for invoking a chain with exponential backoff and API key rotation."""
    global current_key_index
    max_retries = _max_retries(max_retries)
    for attempt in range(max_retries):
        key_used = current_key_index
        try:
            llm = ChatGroq(groq_api_key=_groq_key(key_used), model=model_name, max_tokens=1024)
            chain = prompt | llm.with_structured_output(pydantic_schema)
            return await chain.ainvoke(inputs)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg:
                if len(config.GROQ_API_KEYS) > 1:
                    async with get_key_rotation_lock():
                        if current_key_index == key_used:
                            next_key = (current_key_index + 1) % len(config.GROQ_API_KEYS)
                            log(f"  Rate limit hit on key {current_key_index + 1}. Switching to key {next_key + 1}...")
                            current_key_index = next_key
                        else:
                            log(f"  Key {key_used + 1} was already rotated by another task. Using key {current_key_index + 1}...")

                    if attempt < max_retries - 1:
                        continue  # Retry immediately with new key
                else:
                    log("  Rate limit hit. No alternative keys to switch to.")
            elif "failed to call a function" in err_msg or "tool_use_failed" in err_msg:
                log(f"  Groq tool parsing failed (attempt {attempt + 1}). Retrying...")
                if attempt < max_retries - 1:
                    continue

            if attempt == max_retries - 1:
                raise  # Re-raise on final attempt

            delay = base_delay * (2 ** attempt)
            log(f"  Attempt {attempt + 1} failed: {str(e)[:200]}")
            log(f"  Retrying in {delay:.0f}s...")
            await asyncio.sleep(delay)


async def abatch_invoke_with_retry(prompt, model_name, pydantic_schema, inputs_list, max_retries=None, base_delay=2.0):
    """Concurrently invoke multiple inputs using the async retry logic."""
    tasks = [
        async_invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries, base_delay)
        for inputs in inputs_list
    ]
    # return_exceptions=True prevents one failed request from bringing down the entire batch
    return await asyncio.gather(*tasks, return_exceptions=True)
