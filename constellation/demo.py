"""Offline demo: run the real pipeline on a bundled vault with canned model output.

``constellation demo`` (and the "Try the demo vault" button in the UI) copy
``examples/demo-vault`` to a temporary folder and run the full LangGraph
pipeline over it. Only the two network boundaries are replaced -- the batched
LLM call and the embedding model -- with answers read from
``examples/demo-responses.json``. Everything else (hashing, the FAISS index,
link scoring and capping, rendering, diffs) is the production code path.

The canned answers keep the shape of a real run: the extraction pass returns a
few concepts and tags that the verification pass then drops, and the
relationship pass proposes a few weak links that its verifier rejects.
"""

import asyncio
import contextlib
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest import mock

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from constellation.core import config, prompts, usage
from constellation.core.state import (
    ConceptExtractionOutput,
    ConceptModel,
    RelationshipExtractionOutput,
    RelationshipModel,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
DEMO_VAULT = EXAMPLES_DIR / "demo-vault"
DEMO_RESPONSES = EXAMPLES_DIR / "demo-responses.json"
DEMO_EXCLUDED_DIRS = ["Templates"]

_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_NOISE_DIMS = 4
_NOISE_SCALE = 0.4


def load_responses(path: Path = DEMO_RESPONSES) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_demo_vault(dest: Optional[str] = None) -> Path:
    """Copies the bundled vault to `dest` (must not exist yet) or a fresh temp dir."""
    if not DEMO_VAULT.is_dir():
        raise FileNotFoundError(f"Demo vault not found at {DEMO_VAULT}. Run from a source checkout (pip install -e .).")
    if dest:
        target = Path(dest).expanduser()
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"'{target}' already exists and is not empty.")
    else:
        target = Path(tempfile.mkdtemp(prefix="constellation-demo-")) / "demo-vault"
    shutil.copytree(DEMO_VAULT, target, dirs_exist_ok=True)
    return target


# --------------------------------------------------------------------------- #
# Canned embeddings
# --------------------------------------------------------------------------- #
class DemoEmbeddings(Embeddings):
    """Embeds "Concept name: explanation" as the concept's canned topic weights
    plus a little deterministic noise, so related concepts sit close together
    in the FAISS index and similarity scores are not all identical."""

    provider = "demo"
    model_name = "canned-topic-vectors"

    def __init__(self, responses: dict):
        self.axes: List[str] = list(responses["embedding_axes"])
        self.topics = {
            c["concept_name"]: c.get("topics") or {}
            for note in responses["notes"].values()
            for c in note["concepts"] + note.get("rejected_concepts", [])
        }

    def _vec(self, text: str) -> List[float]:
        name = text.split(":", 1)[0].strip()
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        noise = [(b / 255.0 - 0.5) * 2 * _NOISE_SCALE for b in digest[:_NOISE_DIMS]]
        topics = self.topics.get(name)
        if topics is None:
            # Unknown text (a note the reviewer added): a deterministic hash vector.
            base = [b / 255.0 for b in digest[_NOISE_DIMS : _NOISE_DIMS + len(self.axes)]]
        else:
            base = [float(topics.get(axis, 0.0)) for axis in self.axes]
        vec = base + noise
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        usage.record_embedding_call()
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


# --------------------------------------------------------------------------- #
# Canned LLM
# --------------------------------------------------------------------------- #
def _concept_models(items: list) -> List[ConceptModel]:
    return [ConceptModel(**{k: v for k, v in c.items() if k != "topics"}) for c in items]


def _note_entry(responses: dict, text: str) -> tuple:
    match = _HEADING_RE.search(text or "")
    heading = match.group(1).strip() if match else (text or "").strip().split("\n")[0][:60] or "Untitled"
    entry = responses["notes"].get(heading)
    if entry is None:
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "note"
        entry = {
            "tags": [f"#{slug}"],
            "raw_tags": [],
            "rejected_concepts": [],
            "concepts": [
                {
                    "concept_name": heading,
                    "category": "General",
                    "importance_score": 5,
                    "explanation": f"The main idea of the note '{heading}'.",
                    "important_keywords": [],
                    "related_concepts": [],
                }
            ],
        }
    return heading, entry


def _record_usage(inputs: dict, output) -> None:
    # Roughly what the real prompts cost: a fixed instruction block plus ~4 chars per token.
    usage.record_llm_response(
        AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 450 + len(json.dumps(inputs)) // 4,
                "output_tokens": len(output.model_dump_json()) // 4,
                "total_tokens": 0,
            },
        )
    )


def make_fake_llm(responses: dict, latency: float = 0.0):
    """A drop-in for ``abatch_invoke_with_retry`` that answers from `responses`."""
    relations = responses["relationships"]
    rejected = {(r["source"], r["target"]) for r in relations if r.get("verified") is False}

    def answer(prompt, inputs: dict):
        if prompt is prompts.prompt_concept_extraction:
            _, entry = _note_entry(responses, inputs["text"])
            return ConceptExtractionOutput(
                concepts=_concept_models(entry["concepts"] + entry.get("rejected_concepts", [])),
                tags=entry["tags"] + entry.get("raw_tags", []),
            )
        if prompt is prompts.prompt_concept_verification:
            _, entry = _note_entry(responses, inputs["text"])
            return ConceptExtractionOutput(concepts=_concept_models(entry["concepts"]), tags=entry["tags"])
        if prompt is prompts.prompt_relation:
            new = {c["concept_name"]: c["source_note"] for c in json.loads(inputs["new_concepts"])}
            existing = {c["concept_name"]: c["source_note"] for c in json.loads(inputs["existing_concepts"])}
            return RelationshipExtractionOutput(
                links=[
                    RelationshipModel(
                        source=r["source"],
                        target=r["target"],
                        relationship=r["relationship"],
                        evidence=r["evidence"],
                        from_note=new[r["source"]],
                        to_note=existing[r["target"]],
                    )
                    for r in relations
                    if r["source"] in new and r["target"] in existing
                ]
            )
        if prompt is prompts.prompt_relation_verify:
            proposed = json.loads(inputs["relationships"])
            return RelationshipExtractionOutput.model_validate(
                {"links": [link for link in proposed if (link.get("source"), link.get("target")) not in rejected]}
            )
        raise ValueError("The demo backend does not know this prompt.")

    async def fake(prompt, model, schema, inputs_list, *args, **kwargs):
        out = []
        for inputs in inputs_list:
            if latency:
                await asyncio.sleep(latency)
            result = answer(prompt, inputs)
            _record_usage(inputs, result)
            out.append(result)
        return out

    return fake


@contextlib.contextmanager
def offline_backends(responses: Optional[dict] = None, latency: float = 0.0):
    """Swaps the LLM batch call and the embedding model for the canned demo
    versions for the duration of the block. Process-wide: meant for the CLI
    demo, tests and the single-user local UI, not for a shared server."""
    from constellation.nodes import concepts as concepts_mod
    from constellation.nodes import relations as relations_mod

    responses = responses or load_responses()
    fake = make_fake_llm(responses, latency)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(concepts_mod, "abatch_invoke_with_retry", fake))
        stack.enter_context(mock.patch.object(relations_mod, "abatch_invoke_with_retry", fake))
        stack.enter_context(mock.patch.object(config, "embeddings", DemoEmbeddings(responses)))
        stack.enter_context(mock.patch.object(config, "missing_keys", return_value=[]))
        yield


def run_demo(dest: Optional[str] = None, dry_run: bool = True, latency: float = 0.0) -> tuple:
    """Copies the demo vault and runs the pipeline on it offline.
    Returns (vault_path, final_state)."""
    from main import run_pipeline

    vault = copy_demo_vault(dest)
    config.set_excluded_dirs(DEMO_EXCLUDED_DIRS)
    with offline_backends(latency=latency):
        state = run_pipeline(str(vault), dry_run=dry_run)
    return vault, state
