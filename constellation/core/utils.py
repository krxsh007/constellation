import asyncio
import json
import re
import time
from typing import List, get_args

from langchain_groq import ChatGroq

from constellation.core import config
from constellation.core.log import log

current_key_index = 0
key_rotation_lock = None

# After this many consecutive structured-output parse failures on the same
# request we stop using with_structured_output() and fall back to raw JSON
# mode + manual pydantic parsing.
_FALLBACK_AFTER = 3


class LLMResponseError(ValueError):
    """The API call succeeded but the body is unusable (empty, or unparseable).

    The message always starts with the marker below so the retry loops classify
    it the same way they classify Groq's own tool/JSON errors.
    """


def clean_path(path: str) -> str:
    """Strips outer quotes and whitespace, and expands ~ for user directories."""
    if not path or not str(path).strip():
        return ""
    from pathlib import Path
    return str(Path(str(path).strip().strip("'\"").strip()).expanduser())


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
            "explanation": _truncate_text(c.get("explanation"), 240),
        }
        for c in concepts
    ]


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def prepare_note_text(content: str, max_chars: int = None) -> str:
    """Strip auto-generated sections and cap size so Groq free-tier TPM checks pass."""
    # Imported lazily: core.* must not depend on nodes.* at import time, otherwise
    # anything that imports nodes.io first gets a circular import.
    from constellation.nodes.io import strip_auto_sections

    max_chars = max_chars or config.LLM_MAX_NOTE_CHARS
    text = strip_auto_sections(content).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[... note truncated for token limit ...]"


def _shrink_llm_inputs(inputs: dict, note_char_limit: int) -> dict:
    """Return a copy of prompt inputs with note text truncated further for 413 retries."""
    shrunk = dict(inputs)
    if "text" in shrunk:
        shrunk["text"] = prepare_note_text(str(shrunk["text"]), note_char_limit)
    if "concepts" in shrunk:
        try:
            concepts = json.loads(shrunk["concepts"])
            shrunk["concepts"] = json.dumps(minify_concepts(concepts))
        except (TypeError, json.JSONDecodeError):
            pass
    return shrunk


def _is_rate_limit_error(err_msg: str) -> bool:
    return "rate limit" in err_msg or "429" in err_msg


# Anything that means "the model answered, but not in the shape we asked for".
# This has to include pydantic/LangChain parse failures as well as Groq's own
# tool-call errors -- with method="json_mode" a bad payload surfaces as a
# ValidationError or OutputParserException, never as tool_use_failed, so a
# narrower check leaves the raw-JSON fallback below unreachable.
_UNUSABLE_RESPONSE_MARKER = "model response unusable"

_PARSE_ERROR_MARKERS = (
    _UNUSABLE_RESPONSE_MARKER,
    "failed to call a function",
    "tool_use_failed",
    "json_validate_failed",
    "failed to parse tool call",
    "validation error",
    "invalid json",
    "outputparserexception",
    "expecting value",
    "expecting ',' delimiter",
    "expecting property name",
    "unterminated string",
    "field required",
)


def _is_tool_or_json_parse_error(err_msg: str) -> bool:
    return any(marker in err_msg for marker in _PARSE_ERROR_MARKERS)


def _is_request_too_large(err_msg: str) -> bool:
    return "too large" in err_msg or "413" in err_msg


# Models whose reasoning tokens are billed against max_tokens. Without an
# explicit effort setting they can spend the entire output budget thinking and
# return an empty completion, which looks exactly like a parse failure.
_REASONING_MODEL_MARKERS = ("gpt-oss", "qwen3", "deepseek-r1")


def _is_reasoning_model(model_name: str) -> bool:
    lowered = (model_name or "").lower()
    return any(marker in lowered for marker in _REASONING_MODEL_MARKERS)


def _create_llm(model_name: str, api_key_index: int = None) -> ChatGroq:
    index = current_key_index if api_key_index is None else api_key_index
    extra = {}
    effort = getattr(config, "LLM_REASONING_EFFORT", "")
    if effort and _is_reasoning_model(model_name):
        extra["reasoning_effort"] = effort
    return ChatGroq(
        groq_api_key=_groq_key(index),
        model=model_name,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        temperature=0,
        **extra,
    )


def _response_text(result) -> str:
    """Pull usable text out of a chat response, or say precisely why there is none."""
    content = getattr(result, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        content = "".join(parts)
    if content.strip():
        return content

    meta = getattr(result, "response_metadata", {}) or {}
    finish = meta.get("finish_reason") or meta.get("stop_reason")
    usage = getattr(result, "usage_metadata", None) or meta.get("token_usage") or {}
    extra = getattr(result, "additional_kwargs", {}) or {}
    hint = ""
    if extra.get("reasoning_content") or finish == "length":
        hint = (" The output budget was spent before any answer was emitted -- raise "
                "LLM_MAX_OUTPUT_TOKENS or set LLM_REASONING_EFFORT=low.")
    raise LLMResponseError(
        f"Model response unusable: empty content (finish_reason={finish!r}, "
        f"usage={usage}).{hint}"
    )


def _augmented_prompt(prompt, pydantic_schema):
    """Append the expected JSON schema to a prompt.

    Groq requires the word 'json' somewhere in the messages when json_mode is on.
    We append a concrete SystemMessage (not a template tuple) so the curly braces
    in the schema aren't treated as format specifiers.
    """
    from langchain_core.messages import SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

    schema_hint = json.dumps(pydantic_schema.model_json_schema(), indent=2)
    json_msg = SystemMessage(
        content=f"You MUST respond with valid JSON matching this exact schema:\n{schema_hint}"
    )
    return ChatPromptTemplate.from_messages(list(prompt.messages) + [json_msg])


def _structured_chain(prompt, model_name: str, pydantic_schema, api_key_index: int = None):
    """Use json_mode: model outputs free-form JSON, LangChain validates against the schema."""
    llm = _create_llm(model_name, api_key_index)
    structured = llm.with_structured_output(pydantic_schema, method="json_mode")
    return _augmented_prompt(prompt, pydantic_schema) | structured


def _extract_json_text(raw: str) -> str:
    """Pull the first JSON object or array out of an LLM response string."""
    # Try to find a fenced code block first
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
    if m:
        return m.group(1).strip()
    # Otherwise find first { ... } or [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = raw.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == start_char:
                depth += 1
            elif raw[i] == end_char:
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
        # Unbalanced (truncated) -- hand back the tail so the repair pass can close it.
        return raw[start:].strip()
    return raw.strip()


# Field-name variants smaller models produce instead of the canonical pydantic
# names.  These are only applied when the key is NOT already a real field on the
# target model -- 'source' is a genuine field on RelationshipModel, so renaming
# it unconditionally used to make every valid relationship fail validation.
_FIELD_ALIASES = {
    "concept": "concept_name",
    "name": "concept_name",
    "title": "concept_name",
    "keywords": "important_keywords",
    "key_words": "important_keywords",
    "score": "importance_score",
    "importance": "importance_score",
    "related": "related_concepts",
    "source": "source_note",
    "description": "explanation",
    "type": "relationship",
}


def _nested_models(annotation):
    """Yield pydantic model classes referenced anywhere inside a type annotation."""
    from pydantic import BaseModel

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation
        return
    for arg in get_args(annotation):
        yield from _nested_models(arg)


def _schema_field_names(pydantic_schema) -> set:
    """Every field name valid anywhere in a schema, including nested models."""
    names = set()
    seen = set()
    stack = [pydantic_schema]
    while stack:
        model = stack.pop()
        if model in seen:
            continue
        seen.add(model)
        for field_name, field_info in getattr(model, "model_fields", {}).items():
            names.add(field_name)
            stack.extend(_nested_models(field_info.annotation))
    return names


def _list_item_model(annotation):
    """Return the pydantic model behind a List[Model] annotation, else None."""
    return next(iter(_nested_models(annotation)), None)


def _normalise_field_names(data, valid_fields: set):
    """Map LLM field-name variants to canonical names, without clobbering real fields."""
    if isinstance(data, dict):
        normalised = {}
        for key, value in data.items():
            canonical = key
            if key not in valid_fields:
                alias = _FIELD_ALIASES.get(key)
                if alias and alias in valid_fields and alias not in data:
                    canonical = alias
            normalised[canonical] = _normalise_field_names(value, valid_fields)
        return normalised
    if isinstance(data, list):
        return [_normalise_field_names(item, valid_fields) for item in data]
    return data


def _bracket_stack(s: str):
    """Return the unclosed bracket stack for s, or None if s ends inside a string."""
    stack = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in '[{':
            stack.append(ch)
        elif ch in ']}':
            if stack:
                stack.pop()
    if in_string or escape:
        return None
    return stack


def _close(s: str, stack) -> str:
    return s + ''.join(']' if ch == '[' else '}' for ch in reversed(stack))


def _drop_incomplete_tail(s: str) -> str:
    """Cut back to the last complete element or key/value pair, then close up.

    Scanning from the end, every structural character that sits outside a string
    literal is a candidate cut point: a closing brace/bracket ends an element, a
    comma ends a key/value pair. The first cut that yields parseable JSON wins.
    """
    cut_points = []
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in '}],':
            cut_points.append(i)

    for idx in reversed(cut_points):
        # Keep a closing bracket, drop a trailing comma.
        candidate = s[:idx + 1] if s[idx] in '}]' else s[:idx]
        stack = _bracket_stack(candidate)
        if stack is None:
            continue
        closed = _close(candidate, stack)
        try:
            json.loads(closed)
            return closed
        except json.JSONDecodeError:
            continue
    return s


def _repair_truncated_json(s: str) -> str:
    """Best-effort repair of JSON truncated by a max_tokens cutoff."""
    s = s.rstrip()
    if not s:
        return s

    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass

    # Escape-aware scan: tells us both what is still open and whether we were
    # cut off inside a string literal (counting quotes miscounts escaped ones).
    stack = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in '[{':
            stack.append(ch)
        elif ch in ']}':
            if stack:
                stack.pop()

    repaired = s[:-1] if escape else s
    if in_string:
        repaired += '"'
    repaired = re.sub(r',\s*$', '', repaired)
    # A dangling "key": with no value cannot be closed -- drop it.
    repaired = re.sub(r',?\s*"[^"]*"\s*:\s*$', '', repaired)
    repaired = _close(repaired, stack)

    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        return _drop_incomplete_tail(s)


def _prune_invalid_items(data, pydantic_schema, valid_fields: set):
    """Drop individually-invalid list items so one bad entry can't lose the whole response."""
    if not isinstance(data, dict):
        return None

    cleaned = dict(data)
    dropped = False
    for field_name, field_info in getattr(pydantic_schema, "model_fields", {}).items():
        item_model = _list_item_model(field_info.annotation)
        if item_model is None:
            continue
        value = cleaned.get(field_name)
        if not isinstance(value, list):
            continue
        kept = []
        for item in value:
            try:
                item_model.model_validate(_normalise_field_names(item, valid_fields))
                kept.append(item)
            except Exception:
                dropped = True
        cleaned[field_name] = kept

    if not dropped:
        return None
    try:
        result = pydantic_schema.model_validate(_normalise_field_names(cleaned, valid_fields))
    except Exception:
        return None
    log("  Dropped malformed items from the LLM response; keeping the valid ones.")
    return result


def _fallback_parse(raw_text: str, pydantic_schema):
    """Parse raw LLM text into a pydantic model, tolerating minor format issues."""
    json_str = _extract_json_text(raw_text or "")
    valid_fields = _schema_field_names(pydantic_schema)

    candidates = [json_str]
    repaired = _repair_truncated_json(json_str)
    if repaired != json_str:
        candidates.append(repaired)

    last_error = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, json.JSONDecodeError) as e:
            last_error = e
            continue
        try:
            return pydantic_schema.model_validate(_normalise_field_names(data, valid_fields))
        except Exception as e:
            last_error = e
            pruned = _prune_invalid_items(data, pydantic_schema, valid_fields)
            if pruned is not None:
                return pruned

    raise LLMResponseError(
        f"Model response unusable: could not parse as {pydantic_schema.__name__} "
        f"({last_error}). Raw output began: {(raw_text or '')[:300]!r}"
    )


async def _raw_invoke(prompt, model_name: str, pydantic_schema, inputs: dict,
                      note_char_limit: int, api_key_index: int = None):
    """Invoke the LLM without structured-output constraints and parse manually."""
    llm = _create_llm(model_name, api_key_index)
    chain = _augmented_prompt(prompt, pydantic_schema) | llm
    result = await chain.ainvoke(_shrink_llm_inputs(inputs, note_char_limit))
    return _fallback_parse(_response_text(result), pydantic_schema)


def _raw_invoke_sync(prompt, model_name: str, pydantic_schema, inputs: dict,
                     note_char_limit: int):
    """Sync version of raw invoke fallback."""
    llm = _create_llm(model_name)
    chain = _augmented_prompt(prompt, pydantic_schema) | llm
    result = chain.invoke(_shrink_llm_inputs(inputs, note_char_limit))
    return _fallback_parse(_response_text(result), pydantic_schema)


def invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries=None, base_delay=2.0):
    """Invoke a chain with exponential backoff, structured outputs, and API key rotation."""
    global current_key_index
    max_retries = _max_retries(max_retries)
    note_char_limit = config.LLM_MAX_NOTE_CHARS
    json_failures = 0
    for attempt in range(max_retries):
        try:
            if json_failures >= _FALLBACK_AFTER:
                log(f"  Falling back to raw JSON parsing (attempt {attempt + 1})...")
                return _raw_invoke_sync(prompt, model_name, pydantic_schema, inputs, note_char_limit)
            chain = _structured_chain(prompt, model_name, pydantic_schema)
            return chain.invoke(_shrink_llm_inputs(inputs, note_char_limit))
        except Exception as e:
            err_msg = str(e).lower()
            if _is_rate_limit_error(err_msg):
                if len(config.GROQ_API_KEYS) > 1:
                    log(f"  Rate limit hit on key {current_key_index + 1}. Switching to next key...")
                    current_key_index = (current_key_index + 1) % len(config.GROQ_API_KEYS)
                    if attempt < max_retries - 1:
                        continue  # Retry immediately with new key
                else:
                    log("  Rate limit hit. No alternative keys to switch to.")
            elif _is_request_too_large(err_msg):
                next_limit = max(note_char_limit // 2, 4000)
                if next_limit < note_char_limit and attempt < max_retries - 1:
                    log(f"  Request too large for Groq TPM cap ({str(e)[:200]}). "
                        f"Retrying with {next_limit} chars...")
                    note_char_limit = next_limit
                    continue
            elif _is_tool_or_json_parse_error(err_msg):
                json_failures += 1
                log(f"  Model output rejected (attempt {attempt + 1}): {str(e)[:400]}")
                if len(config.GROQ_API_KEYS) > 1:
                    current_key_index = (current_key_index + 1) % len(config.GROQ_API_KEYS)
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue

            if attempt == max_retries - 1:
                log(f"  Giving up after {max_retries} attempts: {str(e)[:400]}")
                raise  # Re-raise on final attempt

            delay = base_delay * (2 ** attempt)
            log(f"  Attempt {attempt + 1} failed: {str(e)[:400]}")
            log(f"  Retrying in {delay:.0f}s...")
            time.sleep(delay)


async def async_invoke_with_retry(prompt, model_name, pydantic_schema, inputs, max_retries=None, base_delay=2.0):
    """Async wrapper for invoking a chain with exponential backoff and API key rotation."""
    global current_key_index
    max_retries = _max_retries(max_retries)
    note_char_limit = config.LLM_MAX_NOTE_CHARS
    json_failures = 0
    for attempt in range(max_retries):
        key_used = current_key_index
        try:
            if json_failures >= _FALLBACK_AFTER:
                log(f"  Falling back to raw JSON parsing (attempt {attempt + 1})...")
                return await _raw_invoke(prompt, model_name, pydantic_schema, inputs,
                                         note_char_limit, api_key_index=key_used)
            chain = _structured_chain(prompt, model_name, pydantic_schema, api_key_index=key_used)
            return await chain.ainvoke(_shrink_llm_inputs(inputs, note_char_limit))
        except Exception as e:
            err_msg = str(e).lower()
            if _is_rate_limit_error(err_msg):
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
            elif _is_request_too_large(err_msg):
                next_limit = max(note_char_limit // 2, 4000)
                if next_limit < note_char_limit and attempt < max_retries - 1:
                    log(f"  Request too large for Groq TPM cap ({str(e)[:200]}). "
                        f"Retrying with {next_limit} chars...")
                    note_char_limit = next_limit
                    continue
            elif _is_tool_or_json_parse_error(err_msg):
                json_failures += 1
                log(f"  Model output rejected (attempt {attempt + 1}): {str(e)[:400]}")
                if len(config.GROQ_API_KEYS) > 1:
                    async with get_key_rotation_lock():
                        next_key = (current_key_index + 1) % len(config.GROQ_API_KEYS)
                        current_key_index = next_key
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue

            if attempt == max_retries - 1:
                log(f"  Giving up after {max_retries} attempts: {str(e)[:400]}")
                raise  # Re-raise on final attempt

            delay = base_delay * (2 ** attempt)
            log(f"  Attempt {attempt + 1} failed: {str(e)[:400]}")
            log(f"  Retrying in {delay:.0f}s...")
            await asyncio.sleep(delay)


async def abatch_invoke_with_retry(prompt, model_name, pydantic_schema, inputs_list, max_retries=None, base_delay=2.0):
    """Concurrently invoke multiple inputs using the async retry logic."""
    concurrency = max(1, config.LLM_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(inputs):
        async with semaphore:
            return await async_invoke_with_retry(
                prompt, model_name, pydantic_schema, inputs, max_retries, base_delay
            )

    tasks = [_run(inputs) for inputs in inputs_list]
    # return_exceptions=True prevents one failed request from bringing down the entire batch
    return await asyncio.gather(*tasks, return_exceptions=True)
