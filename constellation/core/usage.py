"""Per-run usage counters: LLM calls, tokens and embedding requests.

Counting lives in one tiny module (rather than on the pipeline state) because
the calls happen deep inside retry helpers and worker threads that never see
the state. ``reset()`` at the start of a run, ``snapshot()`` at the end.
"""

import threading

_lock = threading.Lock()
_counters = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "embedding_calls": 0}


def reset() -> None:
    with _lock:
        for key in _counters:
            _counters[key] = 0


def snapshot() -> dict:
    with _lock:
        return dict(_counters)


def record_llm_response(message) -> None:
    """Counts one completed LLM response and its tokens, when the provider
    reports them (``usage_metadata`` on LangChain AIMessages; some providers only
    fill ``response_metadata['token_usage']``)."""
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage:
        meta = getattr(message, "response_metadata", None) or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        usage = {
            "input_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0,
            "output_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0,
        }
    with _lock:
        _counters["llm_calls"] += 1
        _counters["input_tokens"] += int(usage.get("input_tokens") or 0)
        _counters["output_tokens"] += int(usage.get("output_tokens") or 0)


def record_embedding_call(count: int = 1) -> None:
    """One embedding API request (a batch request counts once)."""
    with _lock:
        _counters["embedding_calls"] += count
