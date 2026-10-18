# Constellation for Obsidian

Native Obsidian plugin for [Constellation](../README.md#obsidian-plugin): an LLM extracts the concepts of each note, finds notes that are genuinely related, and writes `## Tags` and `## Related Links` sections (or frontmatter tags) back into your notes, all from inside Obsidian.

See the [main README](../README.md#obsidian-plugin) for installation, usage and where data is stored.

```bash
npm install
npm run build   # produces main.js
npm test        # vitest
```

## Releasing

Obsidian and [BRAT](https://github.com/TfTHacker/obsidian42-brat) read
`manifest.json` from the repository root and download `main.js`,
`manifest.json` and `styles.css` from a GitHub release tagged with the bare
version number.

```bash
cd obsidian-plugin
npm version patch        # bumps package.json, both manifest.json and versions.json copies
git push && git push --tags
```

Pushing the tag (e.g. `0.1.1`, no `v`) runs `.github/workflows/plugin-release.yml`,
which checks the tag matches both manifests, builds, tests and publishes the release.

### Install with BRAT (before it's in the community store)

1. Install the **Obsidian42 - BRAT** community plugin.
2. Command palette → *BRAT: Add a beta plugin for testing* → enter `krxsh007/constellation`.
3. Enable **Constellation** under Settings → Community plugins.

### Listing in the community store

After the first release exists, open a PR to
[obsidianmd/obsidian-releases](https://github.com/obsidianmd/obsidian-releases)
adding an entry for `constellation` to `community-plugins.json`.
