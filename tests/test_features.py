"""Tests for the optional features: backlinks, frontmatter tags, local
providers, usage accounting / run report, dry-run previews, checkpointing,
watch mode and the new CLI flags.

The full-pipeline tests run the real LangGraph graph with fake LLM and
embedding back-ends -- nothing touches the network.

Run with:  python3 -m unittest discover tests
"""

import asyncio
import io as stdio
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage

import main
from constellation.core import config, usage, utils
from constellation.core import log as logmod
from constellation.core import prompts
from constellation.core.state import (
    ConceptExtractionOutput,
    ConceptModel,
    RelationshipExtractionOutput,
    RelationshipModel,
)
from constellation.nodes import concepts as concepts_mod
from constellation.nodes import io
from constellation.nodes import relations as relations_mod
from tests.test_vectorstore import FakeEmbeddings


def setUpModule():
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    logmod.clear_sink()


class ConfigGuard(unittest.TestCase):
    """Restores every config knob a test may flip."""

    def setUp(self):
        self._saved = {
            name: getattr(config, name)
            for name in ("BACKLINKS", "TAGS_IN_FRONTMATTER", "LLM_PROVIDER", "EMBEDDING_PROVIDER",
                         "MAX_LINKS_PER_NOTE", "ANNOTATE_LINKS", "EXCLUDED_DIRS")
        }
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(config, name, value)
        self._tmp.cleanup()

    def write(self, rel, text):
        p = self.vault / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def read(self, rel):
        return (self.vault / rel).read_text(encoding="utf-8")


class TestBacklinks(ConfigGuard):
    def test_backlinks_mirror_links_and_respect_cap(self):
        links = [{"from_note": "A.md", "to_note": "B.md", "score": 0.5, "evidence": "why"},
                 {"from_note": "C.md", "to_note": "B.md", "score": 0.9, "evidence": ""}]
        config.set_backlinks(False)
        self.assertNotIn("B.md", io.build_effective_links(links))

        config.set_backlinks(True)
        eff = io.build_effective_links(links)
        self.assertEqual([i["target"] for i in eff["B.md"]], ["A.md", "C.md"])
        self.assertEqual(eff["B.md"][0]["evidence"], "why")
        self.assertEqual(eff["B.md"][0]["score"], 0.5)

        capped = io.build_effective_links(links, max_links=1)
        self.assertEqual([i["target"] for i in capped["B.md"]], ["C.md"], "highest score survives")

    def test_toggling_backlinks_rewrites_targets(self):
        self.write("A.md", "# A")
        self.write("B.md", "# B")
        state = io.vault_reader({"dir": str(self.vault)})
        state["links"] = [{"from_note": "A.md", "to_note": "B.md", "score": 1.0}]
        config.set_backlinks(True)
        state.update(io.link_writer(state))
        self.assertIn("[[A]]", self.read("B.md"))
        self.assertIn("[[B]]", self.read("A.md"))


class TestFrontmatterTags(ConfigGuard):
    def test_creates_and_removes_frontmatter_losslessly(self):
        body = "# Note\n\nText.\n"
        added_content, added = io.apply_frontmatter_tags(body, ["#ml", "python"], [])
        self.assertEqual(added, ["ml", "python"])
        self.assertTrue(added_content.startswith("---\ntags:\n  - ml\n  - python\n---\n# Note"))
        restored, _ = io.apply_frontmatter_tags(added_content, [], added)
        self.assertEqual(restored, body)

    def test_user_tags_are_kept_and_not_duplicated(self):
        note = "---\ntitle: T\ntags: [Python, mine]\n---\nbody"
        out, added = io.apply_frontmatter_tags(note, ["#python", "#ml"], [])
        self.assertEqual(added, ["ml"], "user already has python")
        self.assertIn("tags: [Python, mine, ml]", out)
        self.assertIn("title: T", out)
        back, _ = io.apply_frontmatter_tags(out, [], ["ml"])
        self.assertEqual(back, note, "only the tool's tags are removed")

    def test_block_list_and_previous_tool_tags(self):
        note = "---\ntags:\n  - mine\n  - old-tool\n---\nbody"
        out, added = io.apply_frontmatter_tags(note, ["new-tool"], ["old-tool"])
        self.assertEqual(out, "---\ntags:\n  - mine\n  - new-tool\n---\nbody")
        self.assertEqual(added, ["new-tool"])

    def test_untouched_when_nothing_to_do(self):
        note = "---\ntags: [a]\n---\nbody"
        self.assertEqual(io.apply_frontmatter_tags(note, [], ["zzz"]), (note, []))
        self.assertEqual(io.apply_frontmatter_tags("plain", [], []), ("plain", []))

    def test_hash_ignores_tool_frontmatter_tags(self):
        base = "---\ntitle: T\n---\nbody"
        tagged, added = io.apply_frontmatter_tags(base, ["x"], [])
        self.assertEqual(io.get_core_hash(tagged, added), io.get_core_hash(base))
        self.assertNotEqual(io.get_core_hash(tagged), io.get_core_hash(base))

    def test_link_writer_frontmatter_mode_and_switch_back(self):
        self.write("A.md", "---\ntags: [mine]\n---\n# A")
        self.write("B.md", "# B")
        config.set_tags_in_frontmatter(True)
        state = io.vault_reader({"dir": str(self.vault)})
        state["tags_by_note"] = {"A.md": ["#ml"]}
        state.update(io.link_writer(state))
        state.update(io.summary_reporter(state))
        text = self.read("A.md")
        self.assertIn("tags: [mine, ml]", text)
        self.assertNotIn("## Tags", text)
        cache = json.loads(self.read(io.CACHE_FILENAME))
        self.assertEqual(cache["frontmatter_tags"], {"A.md": ["ml"]})

        # The tool's own frontmatter edit must not look like a user edit.
        again = io.vault_reader({"dir": str(self.vault)})
        self.assertEqual(again["new_notes"], [])

        # Switch back to a ## Tags section: tool tag leaves the frontmatter.
        config.set_tags_in_frontmatter(False)
        again.update(io.link_writer(again))
        text = self.read("A.md")
        self.assertIn("tags: [mine]", text)
        self.assertIn("## Tags\n#ml", text)
        self.assertEqual(again["fm_tags_by_note"], {})

    def test_strip_removes_only_tool_frontmatter_tags(self):
        self.write("A.md", "---\ntags: [mine, ml]\n---\n# A\n\n## Related Links\n- [[B]]\n")
        (self.vault / io.CACHE_FILENAME).write_text(json.dumps({"frontmatter_tags": {"A.md": ["ml"]}}))
        io.strip_vault(self.vault)
        self.assertEqual(self.read("A.md"), "---\ntags: [mine]\n---\n# A\n")


class TestProviders(ConfigGuard):
    def test_missing_keys_respects_provider(self):
        with mock.patch.object(config, "GROQ_API_KEYS", []), mock.patch.object(config, "GOOGLE_API_KEYS", []):
            config.set_providers(llm="groq", embeddings="google")
            self.assertEqual(config.missing_keys(), ["Groq", "Google"])
            config.set_providers(llm="ollama")
            self.assertEqual(config.missing_keys(), ["Google"])
            config.set_providers(embeddings="ollama")
            self.assertEqual(config.missing_keys(), [])

    def test_invalid_provider_rejected(self):
        with self.assertRaises(ValueError):
            config.set_providers(llm="openai")

    def test_ollama_llm_is_created_lazily(self):
        fake = types.ModuleType("langchain_ollama")
        fake.ChatOllama = lambda **kw: ("chat", kw)
        fake.OllamaEmbeddings = lambda **kw: mock.Mock(
            embed_documents=lambda texts: [[1.0, 0.0]] * len(texts), embed_query=lambda t: [1.0, 0.0])
        config.set_providers(llm="ollama", embeddings="ollama")
        with mock.patch.dict(sys.modules, {"langchain_ollama": fake}):
            kind, kwargs = utils._create_llm("openai/gpt-oss-120b")
            self.assertEqual(kind, "chat")
            self.assertEqual(kwargs["model"], config.OLLAMA_LLM_MODEL)
            self.assertEqual(kwargs["base_url"], config.OLLAMA_BASE_URL)
            self.assertEqual(config.embeddings.provider, "ollama")
            self.assertEqual(config.embeddings.model_name, config.OLLAMA_EMBEDDING_MODEL)
            self.assertEqual(len(config.embeddings.embed_documents(["a", "b"])), 2)

    def test_missing_langchain_ollama_gives_clear_error(self):
        config.set_providers(llm="ollama")
        with mock.patch.dict(sys.modules, {"langchain_ollama": None}):
            with self.assertRaises(RuntimeError) as cm:
                utils._create_llm("x")
        self.assertIn("pip install langchain-ollama", str(cm.exception))

    def test_embeddings_identity_follows_provider(self):
        config.set_providers(embeddings="google")
        self.assertEqual(config.embeddings.provider, "google")
        self.assertEqual(config.embeddings.model_name, config.EMBEDDING_MODEL)


class TestUsageAccounting(unittest.TestCase):
    def setUp(self):
        usage.reset()

    def test_unwrap_records_tokens_and_reraises_parse_errors(self):
        raw = AIMessage(content="{}", usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18})
        parsed = ConceptExtractionOutput(concepts=[], tags=[])
        self.assertIs(utils._unwrap_structured({"raw": raw, "parsed": parsed, "parsing_error": None}), parsed)
        with self.assertRaises(ValueError):
            utils._unwrap_structured({"raw": raw, "parsed": None, "parsing_error": ValueError("validation error")})
        snap = usage.snapshot()
        self.assertEqual((snap["llm_calls"], snap["input_tokens"], snap["output_tokens"]), (2, 22, 14))

    def test_structured_chain_counts_real_groq_response_shape(self):
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_groq import ChatGroq

        payload = json.dumps({"concepts": [], "tags": ["#x"]})
        message = AIMessage(content=payload, usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})

        def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=message)])

        with mock.patch.object(config, "GROQ_API_KEYS", ["gsk_test"]), \
                mock.patch.object(config, "LLM_PROVIDER", "groq"), \
                mock.patch.object(ChatGroq, "_generate", fake_generate):
            chain = utils._structured_chain(prompts.prompt_concept_extraction, "openai/gpt-oss-20b",
                                            ConceptExtractionOutput)
            result = chain.invoke({"text": "hello"})
        self.assertEqual(result.tags, ["#x"])
        self.assertEqual(usage.snapshot()["input_tokens"], 5)

    def test_embedding_calls_counted(self):
        from tests.test_pipeline_fixes import _FakeClient
        emb = config.RotatingGoogleEmbeddings()
        with mock.patch.object(emb, "_get_embedding_instance", return_value=_FakeClient([])), \
                mock.patch.object(config, "GOOGLE_API_KEYS", ["k"]):
            emb.embed_documents(["a"] * 60)
        self.assertEqual(usage.snapshot()["embedding_calls"], 2)


def _fake_llm_factory(calls, fail_relationships=False):
    """A stand-in for abatch_invoke_with_retry that answers by prompt type."""

    async def fake(prompt, model, schema, inputs_list):
        out = []
        for inputs in inputs_list:
            usage.record_llm_response(AIMessage(content="x", usage_metadata={
                "input_tokens": 10, "output_tokens": 2, "total_tokens": 12}))
            if prompt is prompts.prompt_concept_extraction or prompt is prompts.prompt_concept_verification:
                calls["concepts"] += 1
                title = inputs["text"].split("\n")[0].lstrip("# ").strip()
                out.append(ConceptExtractionOutput(concepts=[ConceptModel(
                    concept_name=f"{title} concept", category="c", explanation=f"about {title}",
                    important_keywords=[], related_concepts=[], importance_score=5)], tags=[f"#{title.lower()}"]))
            elif prompt is prompts.prompt_relation:
                if fail_relationships:
                    raise RuntimeError("simulated crash after extraction")
                calls["relations"] += 1
                new = json.loads(inputs["new_concepts"])[0]
                other = json.loads(inputs["existing_concepts"])[0]
                out.append(RelationshipExtractionOutput(links=[RelationshipModel(
                    source=new["concept_name"], target=other["concept_name"], relationship="uses",
                    evidence="related", from_note=new["source_note"], to_note=other["source_note"])]))
            else:
                out.append(RelationshipExtractionOutput.model_validate(
                    {"links": json.loads(inputs["relationships"])}))
        return out

    return fake


class TestFullPipeline(ConfigGuard):
    def setUp(self):
        super().setUp()
        config.set_backlinks(False)
        config.set_tags_in_frontmatter(False)
        config.set_annotate_links(False)
        config.set_excluded_dirs([])
        self.write("Alpha.md", "# Alpha\nabout docker")
        self.write("Beta.md", "# Beta\nabout kubernetes")
        self.calls = {"concepts": 0, "relations": 0}
        self.emb = FakeEmbeddings()

    def run_pipeline(self, dry_run=False, fail_relationships=False):
        fake = _fake_llm_factory(self.calls, fail_relationships)
        with mock.patch.object(concepts_mod, "abatch_invoke_with_retry", fake), \
                mock.patch.object(relations_mod, "abatch_invoke_with_retry", fake), \
                mock.patch.object(config, "embeddings", self.emb), \
                mock.patch.object(config, "missing_keys", return_value=[]):
            return main.run_pipeline(str(self.vault), dry_run=dry_run)

    def test_real_run_writes_links_report_and_backup(self):
        state = self.run_pipeline()
        self.assertIn("[[Beta]]", self.read("Alpha.md"))
        self.assertIn("[[Alpha]]", self.read("Beta.md"))
        report = state["run_report"]
        for key in ("started_at", "duration_s", "notes_total", "notes_processed", "notes_failed",
                    "notes_written", "links_total", "links_new", "links_removed", "llm_calls",
                    "input_tokens", "output_tokens", "embedding_calls", "skipped_due_to_edit",
                    "backup_dir", "dry_run"):
            self.assertIn(key, report)
        self.assertEqual(report["notes_total"], 2)
        self.assertEqual(report["notes_written"], 2)
        self.assertEqual(report["links_new"], report["links_total"])
        self.assertGreater(report["llm_calls"], 0)
        self.assertEqual(report["input_tokens"], report["llm_calls"] * 10)
        self.assertGreater(report["embedding_calls"], 0)
        self.assertFalse(report["dry_run"])
        self.assertTrue(Path(report["backup_dir"]).is_dir())
        saved = json.loads(self.read(io.REPORT_FILENAME))
        self.assertEqual(saved["links_total"], report["links_total"])
        self.assertTrue(all(isinstance(l.get("score"), float) for l in state["links"]))
        self.assertIn("Links:", io.format_run_report(report))

        # Second run: nothing changed, so no LLM work and no writes.
        before = dict(self.calls)
        state2 = self.run_pipeline()
        self.assertEqual(self.calls, before)
        self.assertEqual(state2["run_report"]["notes_written"], 0)
        self.assertEqual(state2["run_report"]["llm_calls"], 0)

    def test_dry_run_previews_without_writing(self):
        state = self.run_pipeline(dry_run=True)
        self.assertEqual(self.read("Alpha.md"), "# Alpha\nabout docker")
        self.assertFalse((self.vault / io.CACHE_FILENAME).exists())
        self.assertFalse((self.vault / io.REPORT_FILENAME).exists())
        self.assertEqual(set(state["preview"]), {"Alpha.md", "Beta.md"})
        self.assertEqual(state["preview"]["Alpha.md"]["before"], "# Alpha\nabout docker")
        self.assertIn("[[Beta]]", state["preview"]["Alpha.md"]["after"])
        self.assertTrue(state["run_report"]["dry_run"])

    def test_checkpoint_survives_crash_after_extraction(self):
        # Crash in the writer stage, i.e. after extraction was paid for.
        with mock.patch.object(io, "render_note", side_effect=RuntimeError("crash")):
            with self.assertRaises(RuntimeError):
                self.run_pipeline()
        self.assertEqual(self.read("Alpha.md"), "# Alpha\nabout docker")

        cache = json.loads(self.read(io.CACHE_FILENAME))
        self.assertEqual(set(cache["files"]), {"Alpha.md", "Beta.md"})
        self.assertEqual(sorted(cache["pending"]), ["Alpha.md", "Beta.md"])
        self.assertEqual(len(cache["concepts"]), 2)
        extraction_calls = self.calls["concepts"]

        state = self.run_pipeline()
        self.assertEqual(self.calls["concepts"], extraction_calls, "extraction is not paid for twice")
        self.assertIn("[[Beta]]", self.read("Alpha.md"))
        self.assertNotIn("pending", json.loads(self.read(io.CACHE_FILENAME)))
        self.assertEqual(state["run_report"]["notes_processed"], 0)


class TestWatchMode(ConfigGuard):
    def test_detect_changes_ignores_own_writes(self):
        a = self.write("A.md", "one")
        before = main.snapshot_mtimes(self.vault)
        a.write_text("tool wrote this")
        import os
        os.utime(a, ns=(1, 1))
        after = main.snapshot_mtimes(self.vault)
        own = {str(a.resolve()): io.content_hash("tool wrote this")}
        self.assertEqual(main.detect_changes(before, after, own), [])
        self.assertEqual(main.detect_changes(before, after, {}), [str(a.resolve())])

        self.write("B.md", "new")
        self.assertEqual(len(main.detect_changes(after, main.snapshot_mtimes(self.vault))), 1)

    def test_watch_reruns_on_change(self):
        a = self.write("A.md", "one")
        runs = []

        def fake_run(vault, dry_run=False):
            runs.append(vault)
            return {"run_report": {}, "written_hashes": {}}

        polls = {"n": 0}

        def fake_sleep(_seconds):
            polls["n"] += 1
            if polls["n"] == 2:  # nothing changes on the first poll
                import os
                a.write_text("user edit")
                os.utime(a, ns=(10**18, 10**18))

        with redirect_stdout(stdio.StringIO()):
            count = main.watch(str(self.vault), 0.01, run=fake_run, sleep=fake_sleep, max_runs=2)
        self.assertEqual(count, 2)
        self.assertEqual(polls["n"], 2)


class TestCLI(ConfigGuard):
    def test_new_flags_parse_and_apply(self):
        args = main.parse_args(["/v", "--backlinks", "--frontmatter-tags", "--watch", "30",
                                "--llm-provider", "ollama", "--embedding-provider", "ollama"])
        self.assertEqual(args.watch, 30.0)
        main.apply_args_to_config(args)
        self.assertTrue(config.BACKLINKS)
        self.assertTrue(config.TAGS_IN_FRONTMATTER)
        self.assertEqual((config.LLM_PROVIDER, config.EMBEDDING_PROVIDER), ("ollama", "ollama"))
        self.assertEqual(main.parse_args(["/v", "--restore", "20260101-000000"]).restore, "20260101-000000")
        self.assertTrue(main.parse_args(["/v", "--strip"]).strip)
        self.assertTrue(main.parse_args(["/v", "--list-backups"]).list_backups)
        with redirect_stderr(stdio.StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["/v", "--strip", "--list-backups"])

    def test_strip_list_and_restore_commands(self):
        self.write("A.md", "# A\n\n## Related Links\n- [[B]]\n")
        out = stdio.StringIO()
        with redirect_stdout(out):
            main.cli([str(self.vault), "--strip"])
            main.cli([str(self.vault), "--list-backups"])
        self.assertIn("Cleaned 1 note(s).", out.getvalue())
        self.assertEqual(self.read("A.md"), "# A\n")
        name = out.getvalue().strip().splitlines()[-1].split()[0]
        with redirect_stdout(stdio.StringIO()):
            main.cli([str(self.vault), "--restore", name])
        self.assertIn("[[B]]", self.read("A.md"))


if __name__ == "__main__":
    unittest.main()
