"""Tests for the evaluation harness (eval/run_eval.py) and the ablation hooks
it drives. Offline: the pipeline runs against the recorded cassettes.

Run with:  python3 -m unittest discover tests
"""

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from constellation.core import config  # noqa: E402
from constellation.core import log as logmod  # noqa: E402
from constellation.nodes import relations  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


def setUpModule():
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    logmod.clear_sink()


class TestScoring(unittest.TestCase):
    def test_optional_pairs_are_neither_rewarded_nor_penalised(self):
        gold = {"required": {("A", "B"), ("C", "D")}, "optional": {("A", "C")}, "distractors": {("B", "D")}}
        pred = {("A", "B"), ("A", "C"), ("B", "D")}
        s = run_eval.score(pred, gold)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 1, 1))
        self.assertEqual(s["precision"], 0.5)
        self.assertEqual(s["recall"], 0.5)
        self.assertEqual(s["optional_hits"], 1)
        self.assertEqual(s["distractor_hits"], 1)

    def test_pairs_are_undirected_and_drop_md_suffix(self):
        self.assertEqual(run_eval._pair("B.md", "A.md"), run_eval._pair("A", "B"))

    def test_gold_titles_exist_in_vault(self):
        titles = {p.stem for p in run_eval.VAULT_DIR.rglob("*.md")}
        gold = run_eval.load_gold()
        for group in gold.values():
            for a, b in group:
                self.assertIn(a, titles)
                self.assertIn(b, titles)
        self.assertFalse(gold["required"] & gold["optional"])

    def test_vectors_round_trip_as_float16(self):
        vec = [0.1, -0.25, 0.5]
        out = run_eval._decode_vector(run_eval._encode_vector(vec))
        for a, b in zip(vec, out):
            self.assertAlmostEqual(a, b, places=3)


class TestAblationHooks(unittest.TestCase):
    def test_defaults_keep_current_behaviour(self):
        self.assertEqual(config.RETRIEVAL_K, 10)
        self.assertFalse(config.SKIP_CONCEPT_VERIFICATION)
        self.assertFalse(config.SKIP_RELATIONSHIP_VERIFICATION)
        self.assertEqual(config.RELATIONSHIP_MODE, "llm")

    def test_retrieval_only_links_cover_both_directions(self):
        links = relations.retrieval_only_links({relations.pair_key("A.md", "B.md"): 0.8})
        self.assertEqual({(l["from_note"], l["to_note"]) for l in links}, {("A.md", "B.md"), ("B.md", "A.md")})

    def test_skipping_relationship_verification_makes_no_llm_call(self):
        saved = config.SKIP_RELATIONSHIP_VERIFICATION
        config.set_ablation(skip_relationship_verification=True)
        try:
            state = {
                "raw_links": [
                    {
                        "source": "x",
                        "target": "y",
                        "relationship": "uses",
                        "evidence": "",
                        "from_note": "A.md",
                        "to_note": "B.md",
                    }
                ],
                "concepts": [
                    {"concept_name": "x", "source_note": "A.md"},
                    {"concept_name": "y", "source_note": "B.md"},
                ],
                "notes": [{"title": "A.md"}, {"title": "B.md"}],
                "pair_similarity": {relations.pair_key("A.md", "B.md"): 0.7},
            }
            out = asyncio.run(relations.relationship_verifier(state))
        finally:
            config.SKIP_RELATIONSHIP_VERIFICATION = saved
        self.assertEqual([(l["from_note"], l["to_note"]) for l in out["new_links"]], [("A.md", "B.md")])


@unittest.skipUnless((run_eval.CASSETTE_DIR / "default.json").exists(), "no recorded cassettes")
class TestReplay(unittest.TestCase):
    def test_default_replay_is_offline_and_matches_baseline(self):
        gold = run_eval.load_gold()
        store = json.loads(run_eval.EMBEDDINGS_CASSETTE.read_text(encoding="utf-8"))["vectors"]
        first = run_eval.run_config("default", "replay", gold, store)
        second = run_eval.run_config("default", "replay", gold, store)
        self.assertEqual(first["live_requests"], 0)
        for key in ("tp", "fp", "pairs", "f1", "llm_calls", "input_tokens", "output_tokens"):
            self.assertEqual(first[key], second[key], key)
        baseline = json.loads(run_eval.BASELINE_PATH.read_text(encoding="utf-8"))
        self.assertAlmostEqual(first["f1"], baseline["f1"], places=4)


if __name__ == "__main__":
    unittest.main()
