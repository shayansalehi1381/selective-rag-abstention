"""Shared fixtures. Every test here runs offline with no model weights.

``HashingEmbedder`` (from ``src.retriever``) stands in for the sentence-transformer.
It hashes tokens into a fixed-size bag-of-words vector, so dense similarity is
deterministic and predictable, which lets tests assert exact rankings.

Tests marked ``integration`` need the network or downloaded models. They only run
when ``RUN_INTEGRATION=1`` is set.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pytest

from src.data_loader import Chunk, make_chunk_id
from src.retriever import HashingEmbedder  # noqa: F401  (re-exported for tests)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test: set RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class FixedEmbedder:
    """Returns hand-specified vectors: lets tests make dense and BM25 disagree on purpose."""

    name = "fixed"

    def __init__(self, mapping: dict[str, Sequence[float]]) -> None:
        self.mapping = {k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()}

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        return np.stack([self.mapping[t] for t in texts])


CORPUS_TEXTS = [
    "BM25 is a sparse lexical ranking function based on term frequency and inverse document frequency.",
    "Dense retrieval encodes queries and passages into embeddings and runs vector similarity search with FAISS.",
    "Conformal prediction provides distribution-free coverage guarantees using a held-out calibration set.",
    "Reciprocal rank fusion merges multiple ranked lists by summing reciprocal ranks of each document.",
    "Selective classification lets a model abstain when confidence is low, trading coverage for lower risk.",
    "Cross-encoder rerankers jointly encode the query and passage for fine-grained relevance scoring.",
]


def make_chunks(texts: Sequence[str], arxiv_id: str = "0000.00000v1", title: str = "Test Paper") -> list[Chunk]:
    chunks, pos = [], 0
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(arxiv_id, i),
                arxiv_id=arxiv_id,
                title=title,
                chunk_index=i,
                text=text,
                char_start=pos,
                char_end=pos + len(text),
            )
        )
        pos += len(text) + 1
    return chunks


@pytest.fixture
def no_model_packages(monkeypatch):
    """Simulate an environment without sentence-transformers / torch (imports raise ImportError)."""
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"sentence_transformers", "torch"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def corpus_chunks() -> list[Chunk]:
    return make_chunks(CORPUS_TEXTS)


@pytest.fixture
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder()


# ---------------------------------------------------------------------------
# Minimal, valid PDF writer (so PDF extraction is tested on real PDF bytes)
# ---------------------------------------------------------------------------


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf_bytes(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a PDF with one text line per list entry, using the Type1 Helvetica font."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for i, lines in enumerate(pages):
        page_id, content_id = page_ids[i], page_ids[i] + 1
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
        for j, line in enumerate(lines):
            ops.append(("T* " if j else "") + f"({_pdf_escape(line)}) Tj")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += b"%d 0 obj\n" % obj_id + objects[obj_id] + b"\nendobj\n"
    xref_pos = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for obj_id in range(1, size):
        out += b"%010d 00000 n \n" % offsets[obj_id]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_pos)
    return bytes(out)


@pytest.fixture
def pdf_factory(tmp_path):
    def _make(name: str, pages: Sequence[Sequence[str]]):
        path = tmp_path / name
        path.write_bytes(make_pdf_bytes(pages))
        return path

    return _make
