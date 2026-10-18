"""Tests for the evaluation harness (eval/run_eval.py) and the ablation hooks
it drives. Offline: the pipeline runs against the recorded cassettes.

Run with:  python3 -m unittest discover tests
"""

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document  # noqa: E402

from constellation.core import config, prompts  # noqa: E402
from constellation.core import log as logmod  # noqa: E402
from constellation.core.state import ConceptExtractionOutput  # noqa: E402
from constellation.nodes import concepts, relations  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)

_ABLATION = ("RETRIEVAL_K", "SKIP_CONCEPT_VERIFICATION", "SKIP_RELATIONSHIP_VERIFICATION", "RELATIONSHIP_MODE")


def setUpModule():
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    logmod.clear_sink()


async def _no_llm(*_args, **_kwargs):
    raise AssertionError("this stage must not call the LLM")


def _concept(name, note):
    return {
        "concept_name": name,
        "category": "c",
        "explanation": f"{name} explained",
        "important_keywords": [],
        "related_concepts": [],
        "importance_score": 5,
        "source_note": note,
    }


class FakeStore:
    """Similarity search over fixed neighbours; records the k it was asked for."""

    def __init__(self, neighbours):
        self.neighbours = neighbours
        self.ks = []

    def similarity_search_with_score(self, query, k):
        self.ks.append(k)
        return [
            (Document(page_content=n, metadata={"name": n, "note": note}), dist)
            for n, note, dist in self.neighbours[:k]
        ]


class AblationCase(unittest.TestCase):
    def setUp(self):
        self._saved = {n: getattr(config, n) for n in _ABLATION}

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(config, n, v)


class TestAblationHooks(AblationCase):
    def test_skip_concept_verification_keeps_raw_concepts_and_normalises_tags(self):
        config.set_ablation(skip_concept_verification=True)
        note = {"title": "A.md", "content": "# A\nbody"}
        state = {
            "new_notes": [note],
            "notes": [note],
            "raw_concepts": [_concept("Docker", "A.md")],
            "raw_tags_by_note": {"A.md": ["Machine Learning", "#ml,"]},
            "dir": "",
            "dry_run": True,
        }
        with mock.patch.object(concepts, "abatch_invoke_with_retry", _no_llm):
            out = asyncio.run(concepts.concept_verifier(state))
        self.assertEqual([c["concept_name"] for c in out["new_concepts"]], ["Docker"])
        self.assertEqual(out["tags_by_note"]["A.md"], ["#machine-learning", "#ml"])
        self.assertEqual(out["failed_notes"], [])

    def test_skip_relationship_verification_keeps_links_without_llm(self):
        config.set_ablation(skip_relationship_verification=True)
        state = {
            "raw_links": [
                {
                    "source": "x",
                    "target": "y",
                    "relationship": "uses",
                    "evidence": "",
                    "from_note": "A",
                    "to_note": "B.md",
                }
            ],
            "concepts": [_concept("x", "A.md"), _concept("y", "B.md")],
            "notes": [{"title": "A.md"}, {"title": "B.md"}],
            "pair_similarity": {relations.pair_key("A.md", "B.md"): 0.7},
        }
        # An LLM call would raise inside the verifier's try/except and leave no links.
        with mock.patch.object(relations, "abatch_invoke_with_retry", _no_llm):
            out = asyncio.run(relations.relationship_verifier(state))
        # Titles are still resolved and links still scored on the skip path.
        self.assertEqual(
            [(l["from_note"], l["to_note"], l["score"]) for l in out["new_links"]], [("A.md", "B.md", 0.7)]
        )

    def test_retrieval_k_limits_search_and_prompt(self):
        config.set_ablation(retrieval_k=2)
        neighbours = [(f"c{i}", f"N{i}.md", float(i)) for i in range(5)]
        store = FakeStore(neighbours)
        all_concepts = [_concept("q", "A.md")] + [_concept(n, note) for n, note, _ in neighbours]
        inputs, sims = relations._retrieve_related(store, {"A.md": [all_concepts[0]]}, all_concepts)
        self.assertEqual(store.ks, [2])
        self.assertEqual(len(json.loads(inputs[0]["existing_concepts"])), 2)
        self.assertEqual(set(sims), {relations.pair_key("A.md", "N0.md"), relations.pair_key("A.md", "N1.md")})

    def test_retrieval_mode_links_similar_notes_without_llm(self):
        config.set_ablation(relationship_mode="retrieval")
        store = FakeStore([("b", "B.md", 0.25), ("a2", "A.md", 0.0)])
        state = {
            "new_concepts": [_concept("a", "A.md")],
            "concepts": [_concept("a", "A.md"), _concept("b", "B.md")],
            "dir": "x",
            "dry_run": False,
        }
        with (
            mock.patch.object(relations, "abatch_invoke_with_retry", _no_llm),
            mock.patch.object(relations.vectorstore, "load_index", return_value=store),
        ):
            out = asyncio.run(relations.relationship_extractor(state))
        # Both directions, never a self-link, scored from the FAISS distance.
        self.assertEqual(
            {(l["from_note"], l["to_note"]) for l in out["raw_links"]}, {("A.md", "B.md"), ("B.md", "A.md")}
        )
        self.assertEqual(out["pair_similarity"], {relations.pair_key("A.md", "B.md"): 0.8})


class TestHarness(unittest.TestCase):
    def test_score_ignores_optional_pairs_and_counts_distractors(self):
        gold = {"required": {("A", "B"), ("C", "D")}, "optional": {("A", "C")}, "distractors": {("B", "D")}}
        s = run_eval.score({("A", "B"), ("A", "C"), ("B", "D")}, gold)
        self.assertEqual((s["tp"], s["fp"], s["fn"], s["optional_hits"], s["distractor_hits"]), (1, 1, 1, 1, 1))
        self.assertEqual((s["precision"], s["recall"], s["f1"]), (0.5, 0.5, 0.5))

    def test_gold_labels_match_the_vault(self):
        titles = {p.stem for p in run_eval.VAULT_DIR.rglob("*.md")}
        gold = run_eval.load_gold()
        for group, pairs in gold.items():
            for a, b in pairs:
                self.assertIn(a, titles, group)
                self.assertIn(b, titles, group)
        self.assertFalse(gold["required"] & gold["optional"])
        self.assertFalse((gold["required"] | gold["optional"]) & gold["distractors"])

    def test_replay_refuses_unrecorded_requests(self):
        recorder = run_eval.LLMRecorder("replay", {})
        with self.assertRaises(run_eval.CassetteMiss):
            asyncio.run(recorder(prompts.prompt_concept_extraction, "m", ConceptExtractionOutput, {"text": "x"}))
        emb = run_eval.EmbeddingRecorder("replay", {}, config.embeddings)
        with self.assertRaises(run_eval.CassetteMiss):
            emb.embed_query("never recorded")

    def test_replay_reproduces_recorded_usage_and_errors(self):
        key = run_eval.request_key(prompts.prompt_concept_extraction, "m", ConceptExtractionOutput, {"text": "x"})
        ok = {"result": {"concepts": [], "tags": ["#t"]}, "error": None, "usage": [[10, 5], [7, 3]]}
        recorder = run_eval.LLMRecorder("replay", {key: ok})
        run_eval.usage.reset()
        out = asyncio.run(recorder(prompts.prompt_concept_extraction, "m", ConceptExtractionOutput, {"text": "x"}))
        self.assertEqual(out.tags, ["#t"])
        snap = run_eval.usage.snapshot()
        self.assertEqual((snap["llm_calls"], snap["input_tokens"], snap["output_tokens"]), (2, 17, 8))
        recorder.entries[key] = {**ok, "result": None, "error": "RuntimeError: boom"}
        with self.assertRaisesRegex(RuntimeError, "boom"):
            asyncio.run(recorder(prompts.prompt_concept_extraction, "m", ConceptExtractionOutput, {"text": "x"}))

    def test_baseline_check_tolerates_five_points_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps({"f1": 0.80}))
            with mock.patch.object(run_eval, "BASELINE_PATH", path), mock.patch("builtins.print"):
                self.assertTrue(run_eval.check_baseline([{"config": "default", "f1": 0.751}]))
                self.assertFalse(run_eval.check_baseline([{"config": "default", "f1": 0.749}]))
                self.assertFalse(run_eval.check_baseline([{"config": "k5", "f1": 0.9}]))


@unittest.skipUnless((run_eval.CASSETTE_DIR / "default.json").exists(), "no recorded cassettes")
class TestReplay(unittest.TestCase):
    def test_default_replay_is_deterministic_and_matches_baseline(self):
        # Fails with CassetteMiss if a pipeline change alters any prompt input.
        gold = run_eval.load_gold()
        store = json.loads(run_eval.EMBEDDINGS_CASSETTE.read_text(encoding="utf-8"))["vectors"]
        first = run_eval.run_config("default", "replay", gold, store)
        second = run_eval.run_config("default", "replay", gold, store)
        for key in ("tp", "fp", "pairs", "f1", "llm_calls", "input_tokens", "output_tokens"):
            self.assertEqual(first[key], second[key], key)
        baseline = json.loads(run_eval.BASELINE_PATH.read_text(encoding="utf-8"))
        self.assertAlmostEqual(first["f1"], baseline["f1"], places=4)


if __name__ == "__main__":
    unittest.main()
