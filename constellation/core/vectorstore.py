"""Local FAISS index living inside the vault at ``.linker_faiss_index``.

Keeping the index next to the notes means the whole tool stays offline-friendly
and portable: copy the vault, keep the brain.
"""

from pathlib import Path
from typing import Iterable, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from constellation.core.log import log

INDEX_DIRNAME = ".linker_faiss_index"


def index_path(vault_dir: str) -> Path:
    return Path(vault_dir) / INDEX_DIRNAME


def index_exists(vault_dir: str) -> bool:
    return (index_path(vault_dir) / "index.faiss").exists()


def load_index(vault_dir: str, embeddings) -> Optional[FAISS]:
    """Returns the on-disk index, or None if it does not exist / is unreadable."""
    if not index_exists(vault_dir):
        return None
    try:
        return FAISS.load_local(
            str(index_path(vault_dir)),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        log(f"  Warning: could not load FAISS index ({e}). Starting a fresh one.")
        return None


def save_index(store: FAISS, vault_dir: str) -> None:
    path = index_path(vault_dir)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))


def drop_notes(store: Optional[FAISS], titles: Iterable[str]) -> int:
    """Removes every vector whose 'note' metadata is in titles. Returns the count."""
    titles = set(titles)
    if store is None or not titles:
        return 0
    stale_ids = [
        doc_id
        for doc_id, doc in store.docstore._dict.items()
        if doc.metadata.get("note") in titles
    ]
    if stale_ids:
        store.delete(stale_ids)
    return len(stale_ids)


def indexed_notes(store: Optional[FAISS]) -> set:
    if store is None:
        return set()
    return {doc.metadata.get("note") for doc in store.docstore._dict.values()}


def add_documents(store: Optional[FAISS], docs: List[Document], embeddings) -> FAISS:
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


def count(store: Optional[FAISS]) -> int:
    return 0 if store is None else len(store.docstore._dict)
