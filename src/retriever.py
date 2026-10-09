"""Hybrid sparse + dense retrieval fused with Reciprocal Rank Fusion.

::

            ┌──► BM25Index  (rank-bm25, Okapi)  ──► ranked ids ─┐
    query ──┤                                                   ├──► RRF(k=60) ──► top-k
            └──► DenseIndex (FAISS, bge-small)  ──► ranked ids ─┘

Lexical matching catches exact terms (method names, acronyms, numbers) that dense
models blur together. Dense matching catches paraphrases that share no tokens. RRF
merges the two using ranks only, so it needs no score calibration between BM25 scores
(unbounded) and cosine similarities (between -1 and 1).

Each ``RetrievalResult`` keeps the per-retriever ranks and raw scores. These are kept
on purpose: they are features for the abstention layer (Phase 4).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from src.data_loader import Chunk, load_chunks_jsonl, save_chunks_jsonl

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_RRF_K = 60

# ---------------------------------------------------------------------------
# Tokenisation (shared by indexing and querying)
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself nor
    of off on once only or other our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which while
    who whom why will with would you your yours yourself yourselves also via using
    use used however thus et al
    """.split()
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """NFKC-normalise, lowercase, split on non-alphanumerics, drop stopwords."""
    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ---------------------------------------------------------------------------
# Sparse retrieval
# ---------------------------------------------------------------------------


class BM25Index:
    """Okapi BM25 over pre-tokenised chunks.

    ``search`` only returns documents that share at least one term with the query.
    A zero-overlap query therefore yields ``[]`` and not an arbitrary ranking of
    zero-score documents, which would otherwise pollute RRF.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._postings: dict[str, set[int]] = {}
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def build(self, texts: Sequence[str]) -> BM25Index:
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(t) for t in texts]
        self._size = len(corpus)
        self._postings = {}
        for doc_idx, toks in enumerate(corpus):
            for tok in set(toks):
                self._postings.setdefault(tok, set()).add(doc_idx)
        # rank-bm25 divides by the corpus size and the average doc length; guard both.
        has_tokens = any(corpus)
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if has_tokens else None
        return self

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(doc_index, score)`` pairs, best first."""
        if self._bm25 is None or k <= 0:
            return []
        q_tokens = tokenize(query)
        candidates: set[int] = set()
        for tok in q_tokens:
            candidates |= self._postings.get(tok, set())
        if not candidates:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(candidates, key=lambda i: (-scores[i], i))
        return [(i, float(scores[i])) for i in ranked[:k]]


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to an ``(n, d)`` float32 matrix of L2-normalised embeddings."""

    name: str

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """``sentence-transformers`` backend. The model is loaded lazily on first use.

    For BGE English models, queries get the instruction prefix recommended by the
    model authors. Passages are encoded without it. Pass ``query_instruction=""``
    to disable the prefix.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_instruction: str | None = None,
    ) -> None:
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        if query_instruction is None:
            is_bge_en = "bge" in model_name.lower() and "-en" in model_name.lower()
            query_instruction = BGE_QUERY_INSTRUCTION if is_bge_en else ""
        self.query_instruction = query_instruction
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.name)
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


class DenseIndex:
    """Exact cosine-similarity search (``IndexFlatIP`` over L2-normalised vectors).

    Flat search is exact and fast enough for corpora of around 10⁵ chunks. That
    matters here: approximate-search error would add noise to the abstention
    signals computed later.
    """

    def __init__(self) -> None:
        self._index = None

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    @property
    def dim(self) -> int | None:
        return None if self._index is None else int(self._index.d)

    def build(self, vectors: np.ndarray) -> DenseIndex:
        import faiss

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError(f"expected a non-empty (n, d) matrix, got shape {vectors.shape}")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(l2_normalize(vectors))
        return self

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._index is None or k <= 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._index.d:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self._index.d}")
        scores, ids = self._index.search(l2_normalize(q), min(k, len(self)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, path: Path | str) -> None:
        import faiss

        if self._index is None:
            raise RuntimeError("nothing to save: index is empty")
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path | str) -> DenseIndex:
        import faiss

        obj = cls()
        obj._index = faiss.read_index(str(path))
        return obj


# ---------------------------------------------------------------------------
# Rank fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: float = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse rankings with RRF (Cormack et al., 2009): ``score(d) = Σ_i w_i / (k + rank_i(d))``.

    Ranks are 1-based. A document missing from a list gets nothing from that list.
    A larger ``k`` flattens the contribution curve, so agreement between retrievers
    matters more than a single top rank. Ties are broken by the best individual rank
    and then by id, so the output is deterministic.
    """
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(ranked_lists, weights):
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:  # ignore duplicates within a single list
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float  # fused RRF score
    rank: int  # 1-based position in the fused ranking
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None


class HybridRetriever:
    """BM25 + dense retrieval fused with RRF.

    Args:
        embedder: any ``Embedder``. Tests inject a deterministic offline one.
        rrf_k: the RRF smoothing constant (60 is the value from the original paper).
        candidate_pool: depth each retriever contributes before fusion.
        weights: ``(bm25_weight, dense_weight)`` for weighted RRF.
    """

    INDEX_FILE = "dense.faiss"
    CHUNKS_FILE = "chunks.jsonl"
    CONFIG_FILE = "config.json"

    def __init__(
        self,
        embedder: Embedder,
        *,
        rrf_k: float = DEFAULT_RRF_K,
        candidate_pool: int = 50,
        weights: tuple[float, float] = (1.0, 1.0),
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        if len(weights) != 2:
            raise ValueError("weights must be a (bm25, dense) pair")
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.weights = tuple(float(w) for w in weights)
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self.dense = DenseIndex()
        self.chunks: list[Chunk] = []
        self._position: dict[str, int] = {}

    @property
    def is_indexed(self) -> bool:
        return bool(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def index(self, chunks: Iterable[Chunk]) -> HybridRetriever:
        chunks = list(chunks)
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        dupes = sorted(i for i, n in Counter(c.chunk_id for c in chunks).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate chunk ids: {dupes[:5]}")
        texts = [c.text for c in chunks]
        self.bm25.build(texts)
        self.dense.build(self.embedder.encode(texts, is_query=False))
        self.chunks = chunks
        self._position = {c.chunk_id: i for i, c in enumerate(chunks)}
        logger.info("indexed %d chunks (dense dim=%s)", len(chunks), self.dense.dim)
        return self

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.is_indexed:
            raise RuntimeError("retriever has no index; call index() or load() first")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query or not query.strip():
            return []

        pool = max(self.candidate_pool, top_k)
        bm25_hits = self.bm25.search(query, pool)
        query_vec = self.embedder.encode([query], is_query=True)[0]
        dense_hits = self.dense.search(query_vec, pool)

        bm25_info = {i: (r, s) for r, (i, s) in enumerate(bm25_hits, start=1)}
        dense_info = {i: (r, s) for r, (i, s) in enumerate(dense_hits, start=1)}
        ids = [c.chunk_id for c in self.chunks]
        fused = reciprocal_rank_fusion(
            [[ids[i] for i, _ in bm25_hits], [ids[i] for i, _ in dense_hits]],
            k=self.rrf_k,
            weights=self.weights,
        )

        results = []
        for rank, (chunk_id, score) in enumerate(fused[:top_k], start=1):
            i = self._position[chunk_id]
            b_rank, b_score = bm25_info.get(i, (None, None))
            d_rank, d_score = dense_info.get(i, (None, None))
            results.append(
                RetrievalResult(
                    chunk=self.chunks[i],
                    score=score,
                    rank=rank,
                    bm25_rank=b_rank,
                    bm25_score=b_score,
                    dense_rank=d_rank,
                    dense_score=d_score,
                )
            )
        return results

    # -- persistence (no pickle: FAISS binary + JSON/JSONL only) -----------------

    def save(self, directory: Path | str) -> Path:
        if not self.is_indexed:
            raise RuntimeError("nothing to save: retriever is not indexed")
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.dense.save(out / self.INDEX_FILE)
        save_chunks_jsonl(self.chunks, out / self.CHUNKS_FILE)
        config = {
            "embedder": getattr(self.embedder, "name", type(self.embedder).__name__),
            "dim": self.dense.dim,
            "rrf_k": self.rrf_k,
            "candidate_pool": self.candidate_pool,
            "weights": list(self.weights),
            "bm25_k1": self.bm25.k1,
            "bm25_b": self.bm25.b,
            "num_chunks": len(self.chunks),
        }
        (out / self.CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: Path | str, embedder: Embedder) -> HybridRetriever:
        src = Path(directory)
        config = json.loads((src / cls.CONFIG_FILE).read_text(encoding="utf-8"))
        name = getattr(embedder, "name", type(embedder).__name__)
        if config["embedder"] != name:
            logger.warning("index built with %s but loading with %s", config["embedder"], name)
        retriever = cls(
            embedder,
            rrf_k=config["rrf_k"],
            candidate_pool=config["candidate_pool"],
            weights=tuple(config["weights"]),
            bm25_k1=config["bm25_k1"],
            bm25_b=config["bm25_b"],
        )
        retriever.chunks = load_chunks_jsonl(src / cls.CHUNKS_FILE)
        retriever._position = {c.chunk_id: i for i, c in enumerate(retriever.chunks)}
        retriever.dense = DenseIndex.load(src / cls.INDEX_FILE)
        if len(retriever.dense) != len(retriever.chunks):
            raise ValueError("corrupt index: vector count does not match chunk count")
        retriever.bm25.build([c.text for c in retriever.chunks])  # cheap and deterministic
        return retriever
