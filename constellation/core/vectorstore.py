"""Local FAISS index living inside the vault at ``.linker_faiss_index``.

Keeping the index next to the notes means the whole tool stays offline-friendly
and portable: copy the vault, keep the brain.

The index is persisted WITHOUT pickle. LangChain's ``save_local`` writes an
``index.pkl`` and ``load_local`` needs ``allow_dangerous_deserialization=True``
to read it back -- with the file living inside a vault that people sync and
share, that is arbitrary code execution for whoever can drop a file in it.
Instead we store:

- ``index.faiss``   -- raw vectors, via ``faiss.write_index`` / ``read_index``
- ``docstore.json`` -- ordered docstore ids plus each document's text/metadata
- ``manifest.json`` -- embedding provider, model and dimension

and rebuild the LangChain FAISS object from those. A legacy ``index.pkl`` is
never opened; the index is simply rebuilt from the cached concepts.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, cast

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from constellation.core.log import log

INDEX_DIRNAME = ".linker_faiss_index"
INDEX_FILENAME = "index.faiss"
DOCSTORE_FILENAME = "docstore.json"
MANIFEST_FILENAME = "manifest.json"
LEGACY_PICKLE_FILENAME = "index.pkl"


def index_path(vault_dir: str) -> Path:
    return Path(vault_dir) / INDEX_DIRNAME


def index_exists(vault_dir: str) -> bool:
    path = index_path(vault_dir)
    return (path / INDEX_FILENAME).exists() and (path / DOCSTORE_FILENAME).exists()


def embedding_identity(embeddings) -> dict:
    """What produced the vectors. Vectors from different models (or providers)
    live in different spaces, so mixing them makes similarity meaningless."""
    return {
        "provider": str(getattr(embeddings, "provider", "") or type(embeddings).__name__),
        "model": str(getattr(embeddings, "model_name", "") or getattr(embeddings, "model", "") or ""),
    }


def read_manifest(vault_dir: str) -> Optional[dict]:
    try:
        with open(index_path(vault_dir) / MANIFEST_FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def drop_index(vault_dir: str) -> None:
    shutil.rmtree(index_path(vault_dir), ignore_errors=True)


def load_index(vault_dir: str, embeddings) -> Optional[FAISS]:
    """Returns the on-disk index, or None if it is missing, legacy, built with a
    different embedding model, or unreadable. None means "rebuild"."""
    path = index_path(vault_dir)
    if not index_exists(vault_dir):
        if (path / LEGACY_PICKLE_FILENAME).exists():
            log("  Found a legacy pickle-based FAISS index; ignoring it (pickle is unsafe to load) "
                "and rebuilding from cached concepts.")
        return None

    manifest = read_manifest(vault_dir)
    wanted = embedding_identity(embeddings)
    if not manifest or any(manifest.get(k) != v for k, v in wanted.items()):
        found = f"{manifest.get('provider')}/{manifest.get('model')}" if manifest else "unknown"
        log(f"  FAISS index was built with {found}, now using "
            f"{wanted['provider']}/{wanted['model']}. Dropping it and re-embedding all concepts.")
        drop_index(vault_dir)
        return None

    try:
        index = faiss.read_index(str(path / INDEX_FILENAME))
        with open(path / DOCSTORE_FILENAME, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = [str(i) for i in data["ids"]]
        docs = {
            doc_id: Document(page_content=str(doc.get("page_content", "")), metadata=dict(doc.get("metadata") or {}))
            for doc_id, doc in data["docs"].items()
        }
        if len(ids) != index.ntotal or any(i not in docs for i in ids):
            raise ValueError("docstore does not match the vector index")
        if manifest.get("dimension") and int(manifest["dimension"]) != index.d:
            raise ValueError("vector dimension does not match the manifest")
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(docs),
            index_to_docstore_id={pos: doc_id for pos, doc_id in enumerate(ids)},
        )
    except Exception as e:
        log(f"  Warning: could not load FAISS index ({e}). Rebuilding it.")
        drop_index(vault_dir)
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_index(store: FAISS, vault_dir: str, embeddings=None) -> None:
    path = index_path(vault_dir)
    path.mkdir(parents=True, exist_ok=True)

    ordered = [store.index_to_docstore_id[pos] for pos in sorted(store.index_to_docstore_id)]
    docs = {}
    for doc_id in ordered:
        doc = cast(Document, store.docstore.search(doc_id))
        docs[doc_id] = {"page_content": doc.page_content, "metadata": doc.metadata}

    tmp_index = path / (INDEX_FILENAME + ".tmp")
    faiss.write_index(store.index, str(tmp_index))
    os.replace(tmp_index, path / INDEX_FILENAME)
    _atomic_write_text(path / DOCSTORE_FILENAME, json.dumps({"ids": ordered, "docs": docs}))

    manifest = embedding_identity(embeddings if embeddings is not None else store.embedding_function)
    manifest["dimension"] = int(store.index.d)
    _atomic_write_text(path / MANIFEST_FILENAME, json.dumps(manifest, indent=2))

    legacy = path / LEGACY_PICKLE_FILENAME
    if legacy.exists():
        legacy.unlink()


def _stored_docs(store: FAISS) -> dict:
    """{docstore id: Document}. Indexes built here always use an InMemoryDocstore."""
    return cast(InMemoryDocstore, store.docstore)._dict


def drop_notes(store: Optional[FAISS], titles: Iterable[str]) -> int:
    """Removes every vector whose 'note' metadata is in titles. Returns the count."""
    titles = set(titles)
    if store is None or not titles:
        return 0
    stale_ids = [
        doc_id
        for doc_id, doc in _stored_docs(store).items()
        if doc.metadata.get("note") in titles
    ]
    if stale_ids:
        store.delete(stale_ids)
    return len(stale_ids)


def indexed_notes(store: Optional[FAISS]) -> set:
    if store is None:
        return set()
    return {doc.metadata.get("note") for doc in _stored_docs(store).values()}


def add_documents(store: Optional[FAISS], docs: List[Document], embeddings) -> Optional[FAISS]:
    if not docs:
        return store
    if store is None:
        return FAISS.from_documents(docs, embeddings)
    try:
        store.add_documents(docs)
        return store
    except Exception as e:
        err_msg = str(e).lower()
        if "dimension" in err_msg or "d == " in err_msg:
            log(f"  Warning: Vector dimension mismatch ({e}). Rebuilding fresh FAISS index for current embedding model...")
            return FAISS.from_documents(docs, embeddings)
        raise


# Preview (dry-run) runs promise to leave the vault untouched, so the index they
# build is handed from the concept verifier to the relationship extractor here
# instead of being saved inside the vault.
_preview_stores: dict = {}


def _preview_key(vault_dir: str) -> str:
    return str(Path(vault_dir).expanduser().resolve())


def hold_preview_index(vault_dir: str, store: Optional[FAISS]) -> None:
    _preview_stores[_preview_key(vault_dir)] = store


def take_preview_index(vault_dir: str) -> Optional[FAISS]:
    return _preview_stores.pop(_preview_key(vault_dir), None)


def count(store: Optional[FAISS]) -> int:
    return 0 if store is None else len(_stored_docs(store))
