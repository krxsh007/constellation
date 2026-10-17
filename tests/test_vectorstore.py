"""Tests for the pickle-free FAISS persistence and the embedding manifest.

The index lives inside the user's vault, which may be synced or shared, so it
must never be loaded with pickle; and vectors from a different embedding model
must never be mixed with the current ones.

Run with:  python3 -m unittest discover tests
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from constellation.core import log as logmod
from constellation.core import vectorstore
from constellation.nodes import concepts as concepts_mod


def setUpModule():
    logmod.set_sink(lambda _msg: None)


def tearDownModule():
    logmod.clear_sink()


class FakeEmbeddings(Embeddings):
    """Deterministic, offline embeddings."""

    def __init__(self, provider="fake", model_name="fake-model", dim=8):
        self.provider = provider
        self.model_name = model_name
        self.dim = dim
        self.calls = 0

    def _vec(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dim]]

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _docs(*pairs):
    return [Document(page_content=f"{n}: about {n}", metadata={"name": n, "note": note}) for n, note in pairs]


class TestPickleFreeIndex(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = self._tmp.name
        self.emb = FakeEmbeddings()

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_without_pickle(self):
        store = vectorstore.add_documents(None, _docs(("Docker", "A.md"), ("K8s", "B.md")), self.emb)
        vectorstore.save_index(store, self.vault, self.emb)

        path = vectorstore.index_path(self.vault)
        self.assertFalse((path / "index.pkl").exists())
        self.assertTrue((path / "docstore.json").exists())
        manifest = json.loads((path / "manifest.json").read_text())
        self.assertEqual(manifest, {"provider": "fake", "model": "fake-model", "dimension": 8})

        loaded = vectorstore.load_index(self.vault, self.emb)
        self.assertIsNotNone(loaded)
        self.assertEqual(vectorstore.count(loaded), 2)
        self.assertEqual(vectorstore.indexed_notes(loaded), {"A.md", "B.md"})
        hit = loaded.similarity_search("Docker: about Docker", k=1)[0]
        self.assertEqual(hit.metadata["name"], "Docker")

    def test_deletions_survive_roundtrip(self):
        store = vectorstore.add_documents(None, _docs(("X", "A.md"), ("Y", "B.md"), ("Z", "C.md")), self.emb)
        vectorstore.drop_notes(store, {"B.md"})
        vectorstore.save_index(store, self.vault, self.emb)
        loaded = vectorstore.load_index(self.vault, self.emb)
        self.assertEqual(vectorstore.indexed_notes(loaded), {"A.md", "C.md"})

    def test_legacy_pickle_is_never_loaded(self):
        path = vectorstore.index_path(self.vault)
        path.mkdir(parents=True)
        (path / "index.faiss").write_bytes(b"junk")
        (path / "index.pkl").write_bytes(b"cos\nsystem\n(S'echo pwned'\ntR.")
        self.assertIsNone(vectorstore.load_index(self.vault, self.emb))

    def test_manifest_mismatch_drops_index(self):
        store = vectorstore.add_documents(None, _docs(("X", "A.md")), self.emb)
        vectorstore.save_index(store, self.vault, self.emb)
        other = FakeEmbeddings(provider="ollama", model_name="nomic-embed-text")
        self.assertIsNone(vectorstore.load_index(self.vault, other))
        self.assertFalse(vectorstore.index_path(self.vault).exists())

    def test_corrupt_docstore_is_rebuilt(self):
        store = vectorstore.add_documents(None, _docs(("X", "A.md")), self.emb)
        vectorstore.save_index(store, self.vault, self.emb)
        (vectorstore.index_path(self.vault) / "docstore.json").write_text('{"ids": [], "docs": {}}')
        self.assertIsNone(vectorstore.load_index(self.vault, self.emb))

    def test_dropped_index_is_rebuilt_from_cached_concepts(self):
        state = {
            "dir": self.vault,
            "notes": [{"title": "A.md"}, {"title": "B.md"}],
            "new_notes": [],
            "concepts": [
                {"concept_name": "X", "explanation": "x", "source_note": "A.md"},
                {"concept_name": "Y", "explanation": "y", "source_note": "B.md"},
            ],
        }
        from unittest import mock
        with mock.patch.object(concepts_mod.config, "embeddings", self.emb):
            concepts_mod._index_concepts(state, [])
        loaded = vectorstore.load_index(self.vault, self.emb)
        self.assertEqual(vectorstore.indexed_notes(loaded), {"A.md", "B.md"})


if __name__ == "__main__":
    unittest.main()
