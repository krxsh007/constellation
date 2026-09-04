"""Tests for the LLM response parsing fallbacks in core.utils.

These guard the path that runs when Groq returns JSON that isn't quite what the
schema asked for -- a wrong field name, a payload cut off by max_tokens, one bad
item in an otherwise good list. Getting this wrong silently loses every concept
or link for a note.

Run with:  python tests/test_llm_parsing.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import config, utils
from core.state import ConceptExtractionOutput, RelationshipExtractionOutput

failures = 0


def assert_that(label, condition):
    global failures
    if condition:
        print(f"PASS  {label}")
    else:
        failures += 1
        print(f"FAIL  {label}")


def concept(name="Docker", **over):
    base = {
        "concept_name": name,
        "category": "tooling",
        "explanation": "Container runtime.",
        "important_keywords": ["containers"],
        "related_concepts": [],
        "importance_score": 8,
        "source_note": "a.md",
    }
    base.update(over)
    return base


def link(source="Docker", target="Kubernetes", **over):
    base = {
        "source": source,
        "target": target,
        "relationship": "used_by",
        "evidence": "K8s schedules containers.",
        "from_note": "a.md",
        "to_note": "b.md",
    }
    base.update(over)
    return base


def run():
    # --- field-name handling -------------------------------------------------
    out = utils._fallback_parse(json.dumps({"links": [link()]}), RelationshipExtractionOutput)
    assert_that("a valid relationship keeps its 'source' field", out.links[0].source == "Docker")

    raw = {"concepts": [{**concept(), "source": "a.md"}], "tags": ["#x"]}
    del raw["concepts"][0]["source_note"]
    out = utils._fallback_parse(json.dumps(raw), ConceptExtractionOutput)
    assert_that("'source' is aliased to source_note where that is the real field",
                out.concepts[0].source_note == "a.md")

    raw = {"concepts": [{**concept(), "name": "Docker"}], "tags": []}
    del raw["concepts"][0]["concept_name"]
    out = utils._fallback_parse(json.dumps(raw), ConceptExtractionOutput)
    assert_that("'name' is aliased to concept_name", out.concepts[0].concept_name == "Docker")

    # --- fenced / prose-wrapped output ---------------------------------------
    fenced = "Sure!\n```json\n" + json.dumps({"links": [link()]}) + "\n```\nHope that helps."
    out = utils._fallback_parse(fenced, RelationshipExtractionOutput)
    assert_that("JSON inside a code fence is extracted", len(out.links) == 1)

    # --- truncation repair ---------------------------------------------------
    full = json.dumps({"links": [link(), link("Redis", "Postgres")]})
    cut = full[: full.index('{"source": "Redis"')] + '{"source": "Redis", "target": "Post'
    out = utils._fallback_parse(cut, RelationshipExtractionOutput)
    assert_that("a payload cut mid-object keeps the complete entries",
                [l.source for l in out.links] == ["Docker"])

    cut_in_string = json.dumps({"links": [link(evidence="a very long ev")]})[:-14]
    out = utils._fallback_parse(cut_in_string, RelationshipExtractionOutput)
    assert_that("a payload cut inside a string still parses", isinstance(out.links, list))

    escaped = '{"links": [' + json.dumps(link(evidence='he said \\"hi\\" then left')) + ']}'
    out = utils._fallback_parse(escaped, RelationshipExtractionOutput)
    assert_that("escaped quotes are not miscounted", len(out.links) == 1)

    # --- per-item pruning ----------------------------------------------------
    mixed = {"concepts": [concept("Docker"), {"concept_name": "Broken"}], "tags": ["#x"]}
    out = utils._fallback_parse(json.dumps(mixed), ConceptExtractionOutput)
    assert_that("one malformed item does not lose the good ones",
                [c.concept_name for c in out.concepts] == ["Docker"])

    # --- hard failures still raise -------------------------------------------
    try:
        utils._fallback_parse("no json here at all", ConceptExtractionOutput)
        assert_that("unparseable output raises", False)
    except Exception:
        assert_that("unparseable output raises", True)

    # --- error classification ------------------------------------------------
    assert_that("pydantic validation errors count as parse errors",
                utils._is_tool_or_json_parse_error("1 validation error for conceptextractionoutput"))
    assert_that("groq tool errors still count as parse errors",
                utils._is_tool_or_json_parse_error("tool_use_failed"))
    assert_that("rate limits are not misread as parse errors",
                not utils._is_tool_or_json_parse_error("rate limit reached for model"))
    assert_that("413s are detected", utils._is_request_too_large("request too large for tpm"))

    # --- empty / truncated completions are diagnosed, not swallowed ----------
    from langchain_core.messages import AIMessage

    try:
        utils._response_text(AIMessage(content="   ", response_metadata={"finish_reason": "length"}))
        assert_that("an empty completion raises", False)
    except utils.LLMResponseError as e:
        assert_that("an empty completion raises with the finish_reason", "length" in str(e))
        assert_that("an empty completion is classified as a parse error",
                    utils._is_tool_or_json_parse_error(str(e).lower()))

    assert_that("block-style content is joined",
                utils._response_text(AIMessage(content=[{"type": "text", "text": '{"a": 1}'}])) == '{"a": 1}')

    try:
        utils._fallback_parse("I could not answer that.", ConceptExtractionOutput)
        assert_that("a prose answer raises", False)
    except utils.LLMResponseError as e:
        assert_that("a parse failure quotes the raw output", "I could not answer" in str(e))

    # --- reasoning models get an explicit effort setting ----------------------
    saved_keys = list(config.GROQ_API_KEYS)
    config.set_groq_keys(["gsk_test_key"])
    try:
        assert_that("gpt-oss gets a reasoning_effort so it cannot eat the output budget",
                    utils._create_llm("openai/gpt-oss-20b").reasoning_effort == config.LLM_REASONING_EFFORT)
        assert_that("non-reasoning models are left alone",
                    utils._create_llm("llama-3.1-8b-instant").reasoning_effort is None)
        assert_that("the output budget comes from config",
                    utils._create_llm("openai/gpt-oss-20b").max_tokens == config.LLM_MAX_OUTPUT_TOKENS)
    finally:
        config.set_groq_keys(saved_keys)

    # --- the raw-JSON fallback actually engages ------------------------------
    calls = {"structured": 0, "raw": 0}
    real_structured, real_raw = utils._structured_chain, utils._raw_invoke_sync

    def fake_structured(*a, **kw):
        calls["structured"] += 1
        raise ValueError("1 validation error for ConceptExtractionOutput\nconcepts\n  Field required")

    def fake_raw(prompt, model_name, schema, inputs, note_char_limit):
        calls["raw"] += 1
        return utils._fallback_parse(json.dumps({"concepts": [concept()], "tags": []}), schema)

    utils._structured_chain, utils._raw_invoke_sync = fake_structured, fake_raw
    try:
        result = utils.invoke_with_retry(None, "m", ConceptExtractionOutput, {"text": "x"},
                                         max_retries=6, base_delay=0)
        assert_that("repeated schema failures fall back to raw JSON parsing",
                    calls["raw"] == 1 and result.concepts[0].concept_name == "Docker")
        assert_that("the fallback kicks in after the configured number of failures",
                    calls["structured"] == utils._FALLBACK_AFTER)
    finally:
        utils._structured_chain, utils._raw_invoke_sync = real_structured, real_raw

    print(f"\n{'FAILED' if failures else 'All tests passed'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
