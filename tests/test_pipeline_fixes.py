"""Regression tests for the pipeline bug fixes: failed-note retries, score-ranked
link caps, dangling links, cached incoming links, concurrent user edits and
non-blocking batched embeddings.

No network: every LLM / embedding call is replaced with a fake.

Run with:  python3 -m unittest discover tests
"""

import asyncio
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constellation.core import config
from constellation.core.state import ConceptExtractionOutput, ConceptModel
from constellation.nodes import concepts as concepts_mod
from constellation.nodes import io
from constellation.nodes.relations import filter_and_score_links, pair_key


def setUpModule():
    from constellation.core import log as logmod
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    from constellation.core import log as logmod
    logmod.clear_sink()


def _concept(name, note):
    return ConceptModel(
        concept_name=name, category="c", explanation=f"{name} explained",
        important_keywords=[], related_concepts=[], importance_score=5, source_note=note,
    )


def _link(src, dst, score=0.0, evidence=""):
    return {"from_note": src, "to_note": dst, "score": score, "evidence": evidence,
            "source": "x", "target": "y", "relationship": "uses"}


class VaultCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name)
        self._saved = (config.MAX_LINKS_PER_NOTE, getattr(config, "BACKLINKS", False))
        config.set_max_links(5)

    def tearDown(self):
        config.set_max_links(self._saved[0])
        if hasattr(config, "set_backlinks"):
            config.set_backlinks(self._saved[1])
        self._tmp.cleanup()

    def write(self, name, text):
        path = self.vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def read(self, name):
        return (self.vault / name).read_text(encoding="utf-8")

    def write_cache(self, **data):
        base = {"files": {}, "concepts": [], "links": [], "tags": {}}
        base.update(data)
        (self.vault / io.CACHE_FILENAME).write_text(json.dumps(base), encoding="utf-8")

    def cache(self):
        return json.loads((self.vault / io.CACHE_FILENAME).read_text(encoding="utf-8"))

    def hashes_for(self, *names):
        return {n: {"hash": io.get_core_hash(self.read(n))} for n in names}


class TestFailedNotesRetry(VaultCase):
    def test_failed_extraction_is_recorded_and_hash_not_saved(self):
        self.write("A.md", "# A\nalpha")
        self.write("B.md", "# B\nbeta")
        state = io.vault_reader({"dir": str(self.vault)})

        good = ConceptExtractionOutput(concepts=[_concept("Alpha", "A.md")], tags=["#a"])

        async def fake_batch(prompt, model, schema, inputs):
            return [good, RuntimeError("boom")]

        with mock.patch.object(concepts_mod, "abatch_invoke_with_retry", fake_batch):
            out = asyncio.run(concepts_mod.concept_extractor(state))
        self.assertEqual(out["failed_notes"], ["B.md"])

        state.update(out)
        state.update(io.summary_reporter(state) or {})
        files = self.cache()["files"]
        self.assertIn("A.md", files)
        self.assertNotIn("B.md", files, "a failed note must be retried next run")

    def test_failed_note_keeps_previous_links_on_disk(self):
        self.write("A.md", "# A\nedited\n\n## Related Links\n- [[B]]\n")
        self.write("B.md", "# B")
        self.write_cache(
            files={"A.md": {"hash": "stale"}, **self.hashes_for("B.md")},
            links=[_link("A.md", "B.md", 0.5)],
            rendered={"A.md": ["B.md"]},
        )
        state = io.vault_reader({"dir": str(self.vault)})
        state["failed_notes"] = ["A.md"]
        out = io.link_writer(state)
        self.assertIn("- [[B]]", self.read("A.md"))
        self.assertEqual(out["notes_written"], 0)
        self.assertEqual(out["rendered_links"].get("A.md"), ["B.md"])


class TestScoreRankedCap(unittest.TestCase):
    def test_cap_keeps_highest_scores_then_sorts_alphabetically(self):
        links = [_link("A.md", "Z.md", 0.9), _link("A.md", "B.md", 0.1),
                 _link("A.md", "M.md", 0.8), _link("A.md", "C.md", 0.2)]
        eff = io.build_effective_links(links, max_links=2, backlinks=False)
        self.assertEqual([i["target"] for i in eff["A.md"]], ["M.md", "Z.md"])

    def test_render_note_ranks_by_score(self):
        targets = [{"target": "Aardvark", "score": 0.1}, {"target": "Zebra", "score": 0.95},
                   {"target": "Mole", "score": 0.5}]
        rendered = io.render_note("# Root", targets=targets, max_links=2)
        self.assertNotIn("[[Aardvark]]", rendered)
        self.assertLess(rendered.index("[[Mole]]"), rendered.index("[[Zebra]]"))

    def test_links_get_scores_from_similarity_and_support(self):
        state = {
            "notes": [{"title": t} for t in ("A.md", "B.md", "C.md")],
            "concepts": [], "links": [],
            "pair_similarity": {pair_key("A.md", "B.md"): 0.6, pair_key("A.md", "C.md"): 0.7},
        }
        raw = [
            {"source": "x1", "target": "y1", "from_note": "A", "to_note": "B.md"},
            {"source": "x2", "target": "y2", "from_note": "A.md", "to_note": "B.md"},
            {"source": "x3", "target": "y3", "from_note": "A.md", "to_note": "C.md"},
            {"source": "x4", "target": "y4", "from_note": "A.md", "to_note": "A.md"},
        ]
        valid, rejected = filter_and_score_links(raw, state)
        scores = {l["to_note"]: l["score"] for l in valid}
        self.assertEqual(rejected, 2)  # duplicate pair + self link
        self.assertAlmostEqual(scores["B.md"], 0.65)  # 0.6 + one extra supporting pair
        self.assertAlmostEqual(scores["C.md"], 0.7)


class TestDanglingAndIncomingLinks(VaultCase):
    def test_deleted_target_is_removed_from_linking_note(self):
        self.write("A.md", "# A\n\n## Related Links\n- [[B]]\n- [[C]]\n")
        self.write("C.md", "# C")
        self.write_cache(
            files=self.hashes_for("A.md", "C.md"),
            links=[_link("A.md", "B.md", 0.5), _link("A.md", "C.md", 0.5)],
            rendered={"A.md": ["B.md", "C.md"]},
        )
        state = io.vault_reader({"dir": str(self.vault)})  # B.md was deleted
        self.assertEqual(state["new_notes"], [])
        state.update(io.link_writer(state))
        text = self.read("A.md")
        self.assertNotIn("[[B]]", text)
        self.assertIn("[[C]]", text)

    def test_editing_target_keeps_incoming_link(self):
        self.write("A.md", "# A")
        self.write("B.md", "# B\nchanged")
        self.write_cache(
            files={**self.hashes_for("A.md"), "B.md": {"hash": "old"}},
            links=[_link("A.md", "B.md", 0.5), _link("B.md", "A.md", 0.5)],
        )
        state = io.vault_reader({"dir": str(self.vault)})
        pairs = {(l["from_note"], l["to_note"]) for l in state["links"]}
        self.assertIn(("A.md", "B.md"), pairs, "A is unchanged and B exists: keep A->B")
        self.assertNotIn(("B.md", "A.md"), pairs, "links FROM re-processed B are regenerated")


class TestConcurrentEdits(VaultCase):
    def test_note_edited_during_run_is_skipped_and_not_hashed(self):
        self.write("A.md", "# A")
        self.write("B.md", "# B")
        state = io.vault_reader({"dir": str(self.vault)})
        state["links"] = [_link("A.md", "B.md", 0.5)]
        state["new_links"] = list(state["links"])
        self.write("A.md", "# A\nuser typed this mid-run")

        state.update(io.link_writer(state))
        self.assertEqual(state["skipped_due_to_edit"], ["A.md"])
        self.assertEqual(self.read("A.md"), "# A\nuser typed this mid-run")
        state.update(io.summary_reporter(state) or {})
        self.assertNotIn("A.md", self.cache()["files"])
        self.assertIn("B.md", self.cache()["files"])


class TestStableOrdering(VaultCase):
    def test_duplicate_filenames_resolve_deterministically(self):
        self.write("zeta/Note.md", "z")
        self.write("alpha/Note.md", "a")
        found = [p.relative_to(self.vault).as_posix() for p in io.iter_notes(self.vault, [])]
        self.assertEqual(found, sorted(found))
        state = io.vault_reader({"dir": str(self.vault)})
        by_title = {n["title"]: n["content"] for n in state["notes"]}
        self.assertEqual(by_title["Note.md"], "a")
        self.assertEqual(by_title["zeta/Note.md"], "z")


class _FakeClient:
    def __init__(self, calls, wrong_length=False):
        self.calls = calls
        self.wrong_length = wrong_length

    def embed_documents(self, texts):
        self.calls.append(("batch", len(texts)))
        n = len(texts) - 1 if self.wrong_length else len(texts)
        return [[float(i)] for i in range(n)]

    def embed_query(self, text):
        self.calls.append(("single", 1))
        return [1.0]


class TestEmbeddingBatching(unittest.TestCase):
    def test_batches_of_fifty(self):
        calls = []
        emb = config.RotatingGoogleEmbeddings()
        with mock.patch.object(emb, "_get_embedding_instance", return_value=_FakeClient(calls)), \
                mock.patch.object(config, "GOOGLE_API_KEYS", ["k"]):
            vectors = emb.embed_documents([f"t{i}" for i in range(120)])
        self.assertEqual(len(vectors), 120)
        self.assertEqual(calls, [("batch", 50), ("batch", 50), ("batch", 20)])

    def test_bad_batch_falls_back_to_single_requests(self):
        calls = []
        emb = config.RotatingGoogleEmbeddings()
        with mock.patch.object(emb, "_get_embedding_instance",
                               return_value=_FakeClient(calls, wrong_length=True)), \
                mock.patch.object(config, "GOOGLE_API_KEYS", ["k"]):
            vectors = emb.embed_documents(["a", "b", "c"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual([c[0] for c in calls], ["batch", "single", "single", "single"])

    def test_indexing_runs_off_the_event_loop(self):
        seen = {}

        def fake_index(state, verified):
            seen["thread"] = threading.get_ident()

        async def fake_batch(prompt, model, schema, inputs):
            return [ConceptExtractionOutput(concepts=[_concept("X", "A.md")], tags=[])]

        async def run():
            seen["loop_thread"] = threading.get_ident()
            state = {"new_notes": [{"title": "A.md", "content": "x"}],
                     "raw_concepts": [{"concept_name": "X", "source_note": "A.md", "explanation": ""}],
                     "concepts": [], "dir": "/nonexistent"}
            return await concepts_mod.concept_verifier(state)

        with mock.patch.object(concepts_mod, "_index_concepts", fake_index), \
                mock.patch.object(concepts_mod, "abatch_invoke_with_retry", fake_batch):
            asyncio.run(run())
        self.assertNotEqual(seen["thread"], seen["loop_thread"])


if __name__ == "__main__":
    unittest.main()
