"""Tests for the LLM response parsing fallbacks in core.utils.

These guard the path that runs when Groq returns JSON that isn't quite what the
schema asked for -- a wrong field name, a payload cut off by max_tokens, one bad
item in an otherwise good list. Getting this wrong silently loses every concept
or link for a note.

Run with:  python tests/test_llm_parsing.py
      or:  python -m unittest discover tests
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from constellation.core import config, utils
from constellation.core.state import ConceptExtractionOutput, RelationshipExtractionOutput


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


class TestLLMParsing(unittest.TestCase):
    def test_field_name_handling(self):
        out = utils._fallback_parse(json.dumps({"links": [link()]}), RelationshipExtractionOutput)
        self.assertEqual(out.links[0].source, "Docker")

        raw = {"concepts": [{**concept(), "source": "a.md"}], "tags": ["#x"]}
        del raw["concepts"][0]["source_note"]
        out = utils._fallback_parse(json.dumps(raw), ConceptExtractionOutput)
        self.assertEqual(out.concepts[0].source_note, "a.md")

        raw = {"concepts": [{**concept(), "name": "Docker"}], "tags": []}
        del raw["concepts"][0]["concept_name"]
        out = utils._fallback_parse(json.dumps(raw), ConceptExtractionOutput)
        self.assertEqual(out.concepts[0].concept_name, "Docker")

    def test_fenced_and_prose_wrapped_output(self):
        fenced = "Sure!\n```json\n" + json.dumps({"links": [link()]}) + "\n```\nHope that helps."
        out = utils._fallback_parse(fenced, RelationshipExtractionOutput)
        self.assertEqual(len(out.links), 1)

    def test_truncation_repair(self):
        full = json.dumps({"links": [link(), link("Redis", "Postgres")]})
        cut = full[: full.index('{"source": "Redis"')] + '{"source": "Redis", "target": "Post'
        out = utils._fallback_parse(cut, RelationshipExtractionOutput)
        self.assertEqual([l.source for l in out.links], ["Docker"])

        cut_in_string = json.dumps({"links": [link(evidence="a very long ev")]})[:-14]
        out = utils._fallback_parse(cut_in_string, RelationshipExtractionOutput)
        self.assertIsInstance(out.links, list)

        escaped = '{"links": [' + json.dumps(link(evidence='he said \\"hi\\" then left')) + ']}'
        out = utils._fallback_parse(escaped, RelationshipExtractionOutput)
        self.assertEqual(len(out.links), 1)

    def test_per_item_pruning(self):
        mixed = {"concepts": [concept("Docker"), {"concept_name": "Broken"}], "tags": ["#x"]}
        out = utils._fallback_parse(json.dumps(mixed), ConceptExtractionOutput)
        self.assertEqual([c.concept_name for c in out.concepts], ["Docker"])

    def test_hard_failures_raise(self):
        with self.assertRaises(Exception):
            utils._fallback_parse("no json here at all", ConceptExtractionOutput)

        with self.assertRaises(utils.LLMResponseError) as cm:
            utils._fallback_parse("I could not answer that.", ConceptExtractionOutput)
        self.assertIn("I could not answer", str(cm.exception))

    def test_error_classification(self):
        self.assertTrue(utils._is_tool_or_json_parse_error("1 validation error for conceptextractionoutput"))
        self.assertTrue(utils._is_tool_or_json_parse_error("tool_use_failed"))
        self.assertFalse(utils._is_tool_or_json_parse_error("rate limit reached for model"))
        self.assertTrue(utils._is_request_too_large("request too large for tpm"))

    def test_empty_completions(self):
        with self.assertRaises(utils.LLMResponseError) as cm:
            utils._response_text(AIMessage(content="   ", response_metadata={"finish_reason": "length"}))
        self.assertIn("length", str(cm.exception))
        self.assertTrue(utils._is_tool_or_json_parse_error(str(cm.exception).lower()))

        self.assertEqual(
            utils._response_text(AIMessage(content=[{"type": "text", "text": '{"a": 1}'}])),
            '{"a": 1}',
        )

    def test_reasoning_models_effort(self):
        saved_keys = list(config.GROQ_API_KEYS)
        config.set_groq_keys(["gsk_test_key"])
        try:
            self.assertEqual(
                utils._create_llm("openai/gpt-oss-20b").reasoning_effort,
                config.LLM_REASONING_EFFORT,
            )
            self.assertIsNone(utils._create_llm("llama-3.1-8b-instant").reasoning_effort)
            self.assertEqual(
                utils._create_llm("openai/gpt-oss-20b").max_tokens,
                config.LLM_MAX_OUTPUT_TOKENS,
            )
        finally:
            config.set_groq_keys(saved_keys)

    def test_raw_json_fallback_engagement(self):
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
            result = utils.invoke_with_retry(
                None, "m", ConceptExtractionOutput, {"text": "x"}, max_retries=6, base_delay=0
            )
            self.assertEqual(calls["raw"], 1)
            self.assertEqual(result.concepts[0].concept_name, "Docker")
            self.assertEqual(calls["structured"], utils._FALLBACK_AFTER)
        finally:
            utils._structured_chain, utils._raw_invoke_sync = real_structured, real_raw


if __name__ == "__main__":
    unittest.main()
