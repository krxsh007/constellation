"""Tiny pluggable logger.

The pipeline nodes call ``log()`` instead of ``print()`` so the same code can
stream progress to a terminal (CLI) or into the Streamlit UI. Streamlit
installs a sink with ``set_sink()``; when no sink is installed we just print.
"""

from typing import Callable, Optional

_sink: Optional[Callable[[str], None]] = None


def set_sink(fn: Optional[Callable[[str], None]]) -> None:
    """Route all subsequent log() calls to fn. Pass None to reset to stdout."""
    global _sink
    _sink = fn


def clear_sink() -> None:
    set_sink(None)


def log(message: str = "") -> None:
    text = str(message)
    if _sink is None:
        print(text, flush=True)
        return
    try:
        _sink(text)
    except Exception:
        # Never let a broken UI sink take down the pipeline.
        print(text, flush=True)
