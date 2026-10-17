"""Offline end-to-end scenarios on a temporary fixture vault.

These run the real LangGraph pipeline with the fake LLM / embedding back-ends
from test_features.py, and cover the user-visible guarantees that the
lower-level tests do not: incremental re-runs, hand-written text surviving,
excluded folders staying untouched and strip -> restore round-tripping.

Run with:  python3 -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from constellation.core import config
from constellation.core import log as logmod
from constellation.core.backup import restore_backup
from constellation.nodes import concepts as concepts_mod
from constellation.nodes import io
from constellation.nodes import relations as relations_mod
from tests.test_features import ConfigGuard, _fake_llm_factory
from tests.test_vectorstore import FakeEmbeddings


def setUpModule():
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    logmod.clear_sink()


class TestEndToEnd(ConfigGuard):
    def setUp(self):
        super().setUp()
        config.set_backlinks(False)
        config.set_tags_in_frontmatter(False)
        config.set_annotate_links(False)
        config.set_excluded_dirs(["Templates"])
        self.write("Alpha.md", "# Alpha\nabout docker")
        self.write("Beta.md", "# Beta\nabout kubernetes")
        self.write("Templates/Skip.md", "# Skip\nabout templates")
        self.calls = {"concepts": 0, "relations": 0}
        self.emb = FakeEmbeddings()

    def run_pipeline(self, dry_run=False):
        fake = _fake_llm_factory(self.calls)
        with mock.patch.object(concepts_mod, "abatch_invoke_with_retry", fake), \
                mock.patch.object(relations_mod, "abatch_invoke_with_retry", fake), \
                mock.patch.object(config, "embeddings", self.emb), \
                mock.patch.object(config, "missing_keys", return_value=[]):
            return main.run_pipeline(str(self.vault), dry_run=dry_run)

    def test_tags_and_links_written_and_excluded_folder_untouched(self):
        self.run_pipeline()
        for name, other in (("Alpha.md", "Beta"), ("Beta.md", "Alpha")):
            text = self.read(name)
            self.assertIn("## Tags", text)
            self.assertIn("## Related Links", text)
            self.assertIn(f"[[{other}]]", text)
            self.assertNotIn("[[Skip]]", text)
        self.assertEqual(self.read("Templates/Skip.md"), "# Skip\nabout templates")

    def test_dry_run_writes_nothing_and_saves_no_cache(self):
        before = {p.name: p.read_text(encoding="utf-8") for p in self.vault.rglob("*.md")}
        self.run_pipeline(dry_run=True)
        after = {p.name: p.read_text(encoding="utf-8") for p in self.vault.rglob("*.md")}
        self.assertEqual(before, after)
        self.assertFalse((self.vault / io.CACHE_FILENAME).exists())
        self.assertFalse((self.vault / ".constellation_backups").exists())

    def test_unchanged_rerun_makes_no_llm_calls(self):
        self.run_pipeline()
        before = dict(self.calls)
        self.run_pipeline()
        self.assertEqual(self.calls, before)

    def test_editing_one_note_reprocesses_only_that_note(self):
        self.run_pipeline()
        per_run = self.calls["concepts"]
        self.assertGreater(per_run, 0)
        self.calls["concepts"] = 0

        self.write("Beta.md", self.read("Beta.md") + "\nnow also about helm\n")
        state = self.run_pipeline()

        self.assertEqual(state["run_report"]["notes_processed"], 1)
        # Two notes cost `per_run` extraction calls, so one note costs half.
        self.assertEqual(self.calls["concepts"] * 2, per_run)
        self.assertIn("now also about helm", self.read("Beta.md"))
        self.assertIn("[[Beta]]", self.read("Alpha.md"))
        self.assertEqual(self.read("Templates/Skip.md"), "# Skip\nabout templates")

    def test_hand_written_text_below_related_links_survives(self):
        self.run_pipeline()
        mine = "\n\nMy own closing thoughts, written by hand.\n"
        self.write("Alpha.md", self.read("Alpha.md").rstrip("\n") + mine)
        self.run_pipeline()
        text = self.read("Alpha.md")
        self.assertIn("My own closing thoughts, written by hand.", text)
        self.assertIn("[[Beta]]", text)
        self.assertEqual(text.count("## Related Links"), 1)
        self.assertEqual(text.count("## Tags"), 1)

    def test_strip_then_restore_round_trips(self):
        self.run_pipeline()
        snapshot = {p.relative_to(self.vault).as_posix(): p.read_text(encoding="utf-8")
                    for p in self.vault.rglob("*.md") if ".constellation_backups" not in p.parts}
        self.assertIn("## Related Links", snapshot["Alpha.md"])

        result = io.strip_vault(self.vault)
        self.assertEqual(result["notes_cleaned"], 2)
        self.assertEqual(self.read("Alpha.md").rstrip(), "# Alpha\nabout docker")
        self.assertNotIn("## Tags", self.read("Beta.md"))

        restored = restore_backup(self.vault, Path(result["backup_dir"]).name)
        self.assertEqual(restored, 2)
        for rel, text in snapshot.items():
            self.assertEqual(self.read(rel), text, rel)


if __name__ == "__main__":
    unittest.main()
