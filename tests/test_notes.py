"""Tests for the note read/write round-trip.

These guard the riskiest part of the tool: it rewrites files in your vault, so
stripping and re-rendering a note must never lose hand-written content and must
be stable across runs.

Run with:  python tests/test_notes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.io import get_core_hash, render_note, strip_auto_sections

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


def run():
    failures = 0

    for name, source, expected in CASES:
        got = strip_auto_sections(source)
        if got == expected:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}\n   expected: {expected!r}\n   got     : {got!r}")

    def assert_that(name, condition):
        nonlocal failures
        if condition:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}")

    base = "# A\n\nBody.\n\nAppended later by hand."
    first = render_note(base, ["#x", "#y"], {"B"})
    second = render_note(strip_auto_sections(first), ["#x", "#y"], {"B"})
    third = render_note(strip_auto_sections(second), ["#x", "#y"], {"B"})

    assert_that("strip(render(x)) == x", strip_auto_sections(first) == base)
    assert_that("render is stable across runs", first == second == third)
    assert_that("hash ignores the auto sections", get_core_hash(first) == get_core_hash(base))
    assert_that(
        "hash detects text appended at the bottom",
        get_core_hash(first) != get_core_hash(first + "\nnew user text\n"),
    )
    assert_that("wikilinks are rendered without the .md suffix", "[[B]]" in first)

    print(f"\n{'FAILED' if failures else 'All tests passed'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
