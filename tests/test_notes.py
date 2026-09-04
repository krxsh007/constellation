"""Tests for the note read/write round-trip and parsing utilities.

These guard the riskiest part of the tool: it rewrites files in your vault, so
stripping and re-rendering a note must never lose hand-written content and must
be stable across runs.

Run with:  python tests/test_notes.py
      or:  python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constellation.core.utils import clean_path
from constellation.nodes.concepts import _normalise_tags
from constellation.nodes.io import get_core_hash, render_note, strip_auto_sections, vault_reader
from constellation.nodes.relations import _resolve_title

CASES = [
    (
        "plain note is untouched",
        "# A\n\nBody text.",
        "# A\n\nBody text.",
    ),
    (
        "generated sections are stripped",
        "# A\n\nBody.\n\n## Tags\n#x #y\n\n## Related Links\n- [[B]]\n- [[C]]\n",
        "# A\n\nBody.",
    ),
    (
        "prose appended below the tags survives",
        "# A\n\nBody.\n\n## Tags\n#x #y\n\nMore thoughts.\n",
        "# A\n\nBody.\n\nMore thoughts.",
    ),
    (
        "a user heading after the links survives",
        "# A\n\nBody.\n\n## Related Links\n- [[B]]\n\n## My Notes\nhello\n",
        "# A\n\nBody.\n\n## My Notes\nhello",
    ),
    (
        "a bullet that is not a wikilink survives",
        "# A\n\n## Related Links\n- [[B]]\n- a manual todo\n",
        "# A\n\n- a manual todo",
    ),
    (
        "no stray blank line is introduced",
        "# A\n\nBody.\n\n## Tags\n#x\nMore.\n",
        "# A\n\nBody.\n\nMore.",
    ),
    (
        "a '#' inside a code fence is not treated as a tag",
        "# A\n\n```py\n# not a tag\n```\n\n## Tags\n#x\n",
        "# A\n\n```py\n# not a tag\n```",
    ),
    ("a note that is only auto sections becomes empty", "## Tags\n#x\n", ""),
    ("an empty note stays empty", "", ""),
]


class TestNoteStrippingAndRendering(unittest.TestCase):
    def test_strip_auto_sections_cases(self):
        for name, source, expected in CASES:
            with self.subTest(case=name):
                got = strip_auto_sections(source)
                self.assertEqual(got, expected, f"Failed on case: {name}")

    def test_render_and_strip_roundtrip(self):
        base = "# A\n\nBody.\n\nAppended later by hand."
        first = render_note(base, ["#x", "#y"], {"B"})
        second = render_note(strip_auto_sections(first), ["#x", "#y"], {"B"})
        third = render_note(strip_auto_sections(second), ["#x", "#y"], {"B"})

        self.assertEqual(strip_auto_sections(first), base, "strip(render(x)) == x")
        self.assertEqual(first, second, "render is stable across runs (1 vs 2)")
        self.assertEqual(second, third, "render is stable across runs (2 vs 3)")
        self.assertEqual(get_core_hash(first), get_core_hash(base), "hash ignores auto sections")
        self.assertNotEqual(
            get_core_hash(first),
            get_core_hash(first + "\nnew user text\n"),
            "hash detects text appended at the bottom",
        )
        self.assertIn("[[B]]", first, "wikilinks are rendered without the .md suffix")

    def test_empty_note_rendering(self):
        self.assertEqual(render_note("", [], set()), "")
        rendered = render_note("", ["#tag"], None)
        self.assertTrue(rendered.startswith("## Tags\n"), "No unnecessary leading blank line")
        self.assertEqual(strip_auto_sections(rendered), "")


class TestSanitizationAndNormalization(unittest.TestCase):
    def test_tag_normalisation(self):
        raw = ["#machine-learning", "deep learning", "#AI,", "PYTHON!", "#tech/web", ""]
        cleaned = _normalise_tags(raw)
        self.assertIn("#machine-learning", cleaned)
        self.assertIn("#deep-learning", cleaned)
        self.assertIn("#ai", cleaned)
        self.assertIn("#python", cleaned)
        self.assertIn("#tech/web", cleaned)
        self.assertNotIn("", cleaned)

    def test_resolve_title(self):
        title_map = {"docker.md": "Docker.md", "kubernetes.md": "Kubernetes.md"}
        self.assertEqual(_resolve_title("docker", title_map), "Docker.md")
        self.assertEqual(_resolve_title("Docker.md", title_map), "Docker.md")
        self.assertEqual(_resolve_title("[[Docker]]", title_map), "Docker.md")
        self.assertEqual(_resolve_title('"Kubernetes"', title_map), "Kubernetes.md")
        self.assertEqual(_resolve_title("unknown", title_map), "unknown")

    def test_clean_path(self):
        self.assertEqual(clean_path('  "C:/Vault"  '), "C:/Vault")
        self.assertEqual(clean_path("'/some/path'"), "/some/path")
        self.assertEqual(clean_path(""), "")

    def test_vault_reader_missing_dir(self):
        with self.assertRaises(ValueError):
            vault_reader({"dir": ""})
        with self.assertRaises(ValueError):
            vault_reader({"dir": "/non/existent/path/for/sure/12345"})


if __name__ == "__main__":
    unittest.main()
