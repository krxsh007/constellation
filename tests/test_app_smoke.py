"""Smoke tests: the Streamlit app renders without exceptions, and the HTML
builders behind the custom UI escape and mark state correctly."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAKE_RESULT = {
    "notes": [{"title": "Docker"}, {"title": "Kubernetes"}, {"title": "Templates"}],
    "links": [
        {"from_note": "Docker.md", "to_note": "Kubernetes.md", "relationship": "used_by",
         "evidence": "Kubernetes schedules </script><b>containers</b>", "score": 0.82},
        {"from_note": "Kubernetes.md", "to_note": "Docker.md", "relationship": "depends_on",
         "evidence": "Pods run container images", "score": 0.61},
    ],
    "tags_by_note": {"Docker": ["#containers", "#devops"], "Kubernetes": ["#orchestration"]},
    "concepts": [{"source_note": "Docker", "concept_name": "Container", "category": "tech",
                  "importance_score": 8, "explanation": "Isolated process"}],
    "run_report": {"notes_total": 3, "notes_processed": 2, "notes_written": 0, "links_new": 2,
                   "input_tokens": 1200, "output_tokens": 300, "duration_s": 4.2},
    "preview": {"Docker": {"before": "# Docker\n", "after": "# Docker\n\n## Related Links\n- [[Kubernetes]]\n"}},
}


class AppSmokeTest(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)

    def test_renders(self):
        at = self._app().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.tabs), 4)

    def test_renders_with_vault_and_result(self):
        with tempfile.TemporaryDirectory() as vault:
            (Path(vault) / "Docker.md").write_text("# Docker\n", encoding="utf-8")
            at = self._app()
            at.session_state["vault_path"] = vault
            at.session_state["result"] = FAKE_RESULT
            at.session_state["result_dry_run"] = True
            at.session_state["result_duration"] = 4.2
            at.session_state["log"] = ["--- [Vault Reader] ---", "read 3 notes"]
            at.run()
            self.assertFalse(at.exception, at.exception)
            self.assertEqual(len(at.tabs), 9)  # 4 top-level + 5 result sub-tabs
            self.assertTrue(any(b.label == "Run preview" for b in at.button))
            # the options dict survives a rerun after a widget change
            at.checkbox(key="w_dry_run").uncheck().run()
            self.assertFalse(at.session_state["opts"]["dry_run"])
            self.assertTrue(any(b.label == "Run Constellation" for b in at.button))


class HelpersTest(unittest.TestCase):
    def test_detect_stage_and_diff(self):
        from constellation.ui import helpers

        self.assertEqual(helpers.detect_stage("\n--- [Link Writer] ---"), "Link Writer")
        self.assertIsNone(helpers.detect_stage("hello"))
        self.assertIn("-", helpers.note_diff("a\n", "b\n"))

    def test_graph_data_dedupes_and_filters(self):
        from constellation.ui import helpers

        data = helpers.graph_data(FAKE_RESULT["links"] + [FAKE_RESULT["links"][0]])
        self.assertEqual({n["id"] for n in data["nodes"]}, {"Docker", "Kubernetes"})
        self.assertEqual(len(data["edges"]), 2)
        only = helpers.graph_data(FAKE_RESULT["links"], query="kube")
        self.assertEqual([n["id"] for n in only["nodes"] if n["match"]], ["Kubernetes"])
        self.assertEqual(helpers.graph_data(FAKE_RESULT["links"], query="zzz")["nodes"], [])

    def test_stage_states(self):
        from constellation.ui import helpers

        s = helpers.stage_states("Concept Verifier")
        self.assertEqual(s[:4], ["done", "done", "running", "pending"])
        self.assertEqual(helpers.stage_states("Concept Verifier", failed=True)[2], "failed")
        self.assertEqual(set(helpers.stage_states("Link Writer", finished=True)), {"done"})
        self.assertEqual(set(helpers.stage_states(None)), {"pending"})


class ComponentsTest(unittest.TestCase):
    def test_graph_html_escapes_script_breakout(self):
        from constellation.ui import components, helpers

        links = [{"from_note": "Evil </script><script>alert(1)</script>.md", "to_note": "B.md",
                  "evidence": "<!-- x </SCRIPT>", "score": 0.5}]
        html = components.graph_html(helpers.graph_data(links))
        self.assertNotIn("</script><script>alert", html)
        self.assertNotIn("<!-- x", html)
        # exactly the template's own closing tags remain
        self.assertEqual(html.lower().count("</script>"), 2)
        self.assertIn(components.VIS_NETWORK_URL, html)
        payload = re.search(r"var DATA = (.*?);\n", html).group(1)
        data = json.loads(payload.replace("<\\/", "</").replace("<\\!--", "<!--"))
        self.assertEqual(data["nodes"][1]["id"], "Evil </script><script>alert(1)</script>")

    def test_stepper_marks_states(self):
        from constellation.ui import components
        from constellation.ui.helpers import STAGES

        html = components.stepper("Relationship Extractor")
        states = re.findall(r'data-state="(\w+)"', html)
        self.assertEqual(len(states), len(STAGES))
        self.assertEqual(states, ["done"] * 3 + ["running"] + ["pending"] * 3)
        failed = re.findall(r'data-state="(\w+)"', components.stepper("Link Writer", failed=True))
        self.assertEqual(failed[5], "failed")
        self.assertEqual(re.findall(r'data-state="(\w+)"', components.stepper("Summary Reporter", finished=True)),
                         ["done"] * 7)

    def test_html_escaping(self):
        from constellation.ui import components

        self.assertNotIn("<b>", components.log_panel(["<b>bold</b>"]))
        diff = components.diff_html("--- a\n+++ b\n@@ -1 +1 @@\n-<i>old</i>\n+new\n")
        self.assertIn("cx-d-del", diff)
        self.assertIn("cx-d-add", diff)
        self.assertNotIn("<i>old", diff)
        self.assertNotIn("<x>", components.metric_cards([("<x>", "<x>")]))
        self.assertNotIn("<x>", components.chip("<x>"))


if __name__ == "__main__":
    unittest.main()
