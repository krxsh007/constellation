# Contributing

Thanks for taking a look. The short version: everything runs offline, so you can work on any part of the project without API keys.

## Setup

```bash
git clone https://github.com/krxsh007/constellation && cd constellation
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install          # optional: ruff, format check and whitespace hooks on commit (pip install pre-commit)
constellation demo          # sanity check: full pipeline on the bundled vault, no keys needed
```

## Checks (the same ones CI runs)

```bash
ruff check .
ruff format --check .
pyright
python3 -m unittest discover tests          # or: pytest --cov
```

Obsidian plugin:

```bash
cd obsidian-plugin
npm ci
npm run build
npm test
```

## Guidelines

- **Tests stay offline.** New pipeline behaviour gets a test that runs the real graph with the fake LLM and embedding back-ends (see `tests/test_e2e.py` and `tests/test_features.py`). Never call a real provider from a test.
- **Notes are user data.** Anything that changes what is written into a note needs a test that hand-written text around the generated sections survives (`tests/test_notes.py`).
- **Python and plugin behaviour stay in sync.** If you change section-rewriting or link-rendering rules in `constellation/nodes/io.py`, regenerate the parity fixture with `python obsidian-plugin/scripts/gen_parity_fixture.py` and make the plugin match.
- **Formatting.** New Python files must pass `ruff format`. Older modules are listed in `[tool.ruff.format] exclude` in `pyproject.toml` until they are reformatted. Don't reformat them as part of an unrelated change.
- **Commits.** Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `ci:`, ...), and add an entry to `CHANGELOG.md` under *Unreleased* for anything user-visible.
- Read [docs/DESIGN.md](docs/DESIGN.md) before changing the cache, the index format or the write path; the trade-offs there are deliberate.

## Releasing

Bump `version` in `pyproject.toml`, move the *Unreleased* changelog entries under the new version, then tag `vX.Y.Z`. The release workflow builds the sdist and wheel and attaches them to the GitHub release.
