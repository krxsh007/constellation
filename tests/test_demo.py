"""The offline demo: `constellation demo` and the UI's "Try the demo vault"
button run the real pipeline on examples/demo-vault with canned model output.

Run with:  python3 -m unittest discover tests
"""

import io as stdio
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
from constellation import demo
from constellation.core import config
from constellation.core import log as logmod
from constellation.nodes import io
from tests.test_features import ConfigGuard

ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    logmod.clear_sink()


def _snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


class TestDemo(ConfigGuard):
    def setUp(self):
        super().setUp()
        config.set_backlinks(False)
        config.set_tags_in_frontmatter(False)
        config.set_annotate_links(False)
        config.set_max_links(5)
        self.bundled = _snapshot(demo.DEMO_VAULT)
        # No keys at all: the demo must not need (or reach) any provider.
        keys = mock.patch.multiple(config, GROQ_API_KEYS=[], GOOGLE_API_KEYS=[])
        keys.start()
        self.addCleanup(keys.stop)

    def tearDown(self):
        self.assertEqual(_snapshot(demo.DEMO_VAULT), self.bundled, "the bundled vault must never change")
        super().tearDown()

    def run_demo(self, dry_run=True):
        return demo.run_demo(str(self.vault / "copy"), dry_run=dry_run)

    def test_preview_produces_tags_links_and_diffs_without_writing(self):
        vault, state = self.run_demo()
        notes = sorted(p.name for p in vault.rglob("*.md") if "Templates" not in p.parts)
        self.assertEqual(len(notes), 15)
        self.assertEqual(_snapshot(vault), _snapshot(demo.DEMO_VAULT), "preview writes nothing")
        self.assertFalse((vault / io.CACHE_FILENAME).exists())

        self.assertEqual(state["run_report"]["notes_processed"], 15)
        self.assertEqual(state["failed_notes"], [])
        self.assertEqual(set(state["tags_by_note"]), set(notes))
        pairs = {(l["from_note"], l["to_note"]) for l in state["links"]}
        self.assertIn(("Kubernetes.md", "Raft Consensus.md"), pairs)
        self.assertIn(("Zettelkasten.md", "Text Embeddings.md"), pairs, "cross-topic link")
        self.assertGreaterEqual(len(pairs), 15)
        self.assertTrue(all(0 < l["score"] <= 1.5 for l in state["links"]))
        self.assertTrue(all(len([p for p in pairs if p[0] == n]) <= 5 for n in notes), "link cap holds")

        after = state["preview"]["Kubernetes.md"]["after"]
        self.assertIn("## Related Links\n- [[Raft Consensus]]", after)
        self.assertIn("#kubernetes", after)
        self.assertNotIn("Daily Note.md", state["preview"], "Templates is excluded")

    def test_verification_passes_drop_what_extraction_over_produced(self):
        _, state = self.run_demo()
        concepts = {c["concept_name"] for c in state["concepts"]}
        self.assertNotIn("Works On My Machine", concepts)
        self.assertIn("Container Image", concepts)
        self.assertNotIn("#notes-to-self", state["tags_by_note"]["Docker.md"])
        pairs = {(l["from_note"], l["to_note"]) for l in state["links"]}
        self.assertNotIn(("Vector Search.md", "CAP Theorem.md"), pairs)
        self.assertNotIn(("Helm Charts.md", "Event Sourcing.md"), pairs)

    def test_write_mode_updates_only_the_copy_and_reruns_for_free(self):
        vault, state = self.run_demo(dry_run=False)
        self.assertEqual(state["run_report"]["notes_written"], 15)
        self.assertIn("[[Raft Consensus]]", (vault / "DevOps" / "Kubernetes.md").read_text(encoding="utf-8"))
        with demo.offline_backends():
            again = main.run_pipeline(str(vault))
        self.assertEqual(again["run_report"]["llm_calls"], 0, "content-hash cache skips unchanged notes")

    def test_cli_prints_report_links_and_diff(self):
        out = stdio.StringIO()
        with redirect_stdout(out):
            main.cli(["demo", "--dir", str(self.vault / "cli")])
        text = out.getvalue()
        for needle in (
            "Demo vault:",
            "preview (nothing written)",
            "Links:",
            "Kubernetes",
            "-> Raft Consensus",
            "+- [[Raft Consensus]]",
        ):
            self.assertIn(needle, text)

    def test_refuses_a_non_empty_destination(self):
        (self.vault / "busy").mkdir()
        (self.vault / "busy" / "x.md").write_text("mine", encoding="utf-8")
        with self.assertRaises(ValueError):
            demo.copy_demo_vault(str(self.vault / "busy"))


class TestDemoButton(ConfigGuard):
    def test_button_runs_a_preview_on_the_demo_vault(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        # The app installs and clears its own log sink; keep the test output quiet afterwards.
        self.addCleanup(logmod.set_sink, lambda _msg: None)
        with mock.patch.multiple(config, GROQ_API_KEYS=[], GOOGLE_API_KEYS=[]):
            at.run()
            at.button(key="demo_btn").click().run()
        self.assertFalse(at.exception, at.exception)
        result = at.session_state["result"]
        self.assertTrue(at.session_state["result_dry_run"])
        self.assertGreaterEqual(len(result["links"]), 15)
        self.assertEqual(at.session_state["vault_path"], at.session_state["demo_vault"])


if __name__ == "__main__":
    unittest.main()
