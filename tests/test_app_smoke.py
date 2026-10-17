"""Smoke test: the Streamlit app renders without exceptions."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AppSmokeTest(unittest.TestCase):
    def test_renders(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.tabs), 4)

    def test_helpers(self):
        from constellation.ui import helpers

        self.assertEqual(helpers.detect_stage("\n--- [Link Writer] ---"), "Link Writer")
        self.assertIsNone(helpers.detect_stage("hello"))
        dot, n, e = helpers.graph_dot([{"from_note": "A.md", "to_note": "B.md"}])
        self.assertEqual((n, e), (2, 1))
        self.assertIn("-", helpers.note_diff("a\n", "b\n"))


if __name__ == "__main__":
    unittest.main()
