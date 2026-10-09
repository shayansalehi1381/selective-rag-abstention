"""Second-stage reranking: re-score first-stage candidates with a cross-encoder.

::

    query ──► HybridRetriever (top retrieve_k, e.g. 50) ──► Reranker (query, passage) pairs ──► top_k

A cross-encoder reads the query and the passage *together*, so it can model term
interactions (proximity, order, which entity a number belongs to). A bi-encoder or
BM25 only compares independent representations. Its raw logit is also a per-pair
relevance score, which makes it a key confidence signal for the abstention layer.

Backends:

* ``cross-encoder:<model>``: ``sentence_transformers.CrossEncoder`` (default
  ``cross-encoder/ms-marco-MiniLM-L6-v2``). It is loaded lazily, and raw logits are
  requested explicitly (``activation_fn=Identity``) so the probability is always
  ``sigmoid(logit)``, whatever the model's configured activation.
* ``mock-lexical-v1``: ``MockCrossEncoder``, a deterministic, dependency-free scorer
  for CI and sandboxes without model weights. When ``allow_fallback=True`` it is used
  automatically if the real model cannot be loaded. A WARNING is logged and the backend
  name is recorded, so mock results are never mistaken for real-model results.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.data_loader import Chunk
from src.retriever import HybridRetriever, RetrievalResult, tokenize

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass(frozen=True)
class RerankedDocument:
    index: int  # position in the input list
    chunk_id: str | None  # None when plain strings were reranked
    text: str
    score: float  # raw logit
    probability: float  # sigmoid(score)
    rank: int  # 1-based


# ---------------------------------------------------------------------------
# Deterministic offline scorer
# ---------------------------------------------------------------------------


def trigrams(token: str) -> set[str]:
    padded = f"#{token}#"
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class MockCrossEncoder:
    """Lexical *interaction* scorer that stands in for a neural cross-encoder.

    The features deliberately go beyond bag-of-words matching, so the reranked track
    is not a copy of BM25:

    * ``coverage``: share of the query's IDF mass found in the passage. IDF is computed
      over the candidate set of each query, so no corpus state is needed (a real
      cross-encoder has none either).
    * ``proximity``: ``(m / |q|) · (m / span)``, where ``m`` is the number of distinct
      matched query terms and ``span`` the shortest token window containing all of
      them. Requires ``m ≥ 2``. Rewards passages that state the asked-about entities
      *together*.
    * ``bigram``: share of ordered query bigrams that occur contiguously in the passage.
    * ``fuzzy``: mean over query terms of 1 (exact match) or the best character-trigram
      Jaccard similarity to a passage term (morphological variants).

    ``logit = 6·coverage + 3·proximity + 2·bigram + 1·fuzzy − 5``. The weights are fixed
    a priori and never fitted on evaluation data. Every feature lies in [0, 1], so the
    logit lies in [−5, 7]. It is still a heuristic and does not understand semantics.
    """

    name = "mock-lexical-v1"
    WEIGHTS = {"coverage": 6.0, "proximity": 3.0, "bigram": 2.0, "fuzzy": 1.0}
    BIAS = -5.0

    @staticmethod
    def _idf(query_terms: set[str], docs: Sequence[list[str]]) -> dict[str, float]:
        n = len(docs)
        doc_sets = [set(d) for d in docs]
        return {t: math.log1p((n - sum(t in d for d in doc_sets) + 0.5) /
                              (sum(t in d for d in doc_sets) + 0.5)) for t in query_terms}

    @staticmethod
    def _min_span(doc: list[str], terms: set[str]) -> int:
        """Shortest window (in tokens) containing every term in ``terms`` (all must occur)."""
        need, have, best, left = len(terms), {}, len(doc), 0
        for right, tok in enumerate(doc):
            if tok in terms:
                have[tok] = have.get(tok, 0) + 1
            while len(have) == need:
                best = min(best, right - left + 1)
                lt = doc[left]
                if lt in have:
                    have[lt] -= 1
                    if not have[lt]:
                        del have[lt]
                left += 1
        return best

    def features(self, query: str, doc: str, idf: dict[str, float] | None = None) -> dict[str, float]:
        q = tokenize(query)
        d = tokenize(doc)
        q_set, d_set = set(q), set(d)
        if not q_set or not d:
            return {k: 0.0 for k in self.WEIGHTS}
        idf = idf or self._idf(q_set, [d])
        # Sum in sorted order: set iteration order varies across processes (hash
        # randomisation), which would make float sums differ in the last bits.
        q_terms = sorted(q_set)
        matched = q_set & d_set
        total = sum(idf[t] for t in q_terms)
        coverage = sum(idf[t] for t in q_terms if t in matched) / total if total > 0 else 0.0
        m = len(matched)
        proximity = (m / len(q_set)) * (m / self._min_span(d, matched)) if m >= 2 else 0.0
        q_bigrams = {(a, b) for a, b in zip(q, q[1:])}
        d_bigrams = set(zip(d, d[1:]))
        bigram = len(q_bigrams & d_bigrams) / len(q_bigrams) if q_bigrams else 0.0
        d_grams = [trigrams(t) for t in sorted(d_set)]
        fuzzy_scores = []
        for t in q_terms:
            if t in d_set:
                fuzzy_scores.append(1.0)
                continue
            tg = trigrams(t)
            fuzzy_scores.append(max((len(tg & g) / len(tg | g) for g in d_grams), default=0.0))
        fuzzy = float(np.mean(fuzzy_scores))
        return {"coverage": coverage, "proximity": proximity, "bigram": bigram, "fuzzy": fuzzy}

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        """Logits for (query, passage) pairs. IDF is shared by pairs with the same query."""
        by_query: dict[str, list[int]] = {}
        for i, (q, _) in enumerate(pairs):
            by_query.setdefault(q, []).append(i)
        logits = np.zeros(len(pairs), dtype=np.float64)
        for q, idxs in by_query.items():
            q_set = set(tokenize(q))
            idf = self._idf(q_set, [tokenize(pairs[i][1]) for i in idxs])
            for i in idxs:
                f = self.features(q, pairs[i][1], idf)
                logits[i] = self.BIAS + sum(self.WEIGHTS[k] * v for k, v in f.items())
        return logits


# ---------------------------------------------------------------------------
# Reranker facade
# ---------------------------------------------------------------------------


class Reranker:
    """Cross-encoder reranker with an explicit, logged offline fallback.

    Args:
        model_name: Hugging Face id of a sentence-transformers cross-encoder.
        device: e.g. ``"cpu"`` or ``"cuda"``; ``None`` lets sentence-transformers choose.
        mock: use ``MockCrossEncoder`` directly (no import or download attempt).
        allow_fallback: if the real model cannot be loaded (missing packages, no
            network, unknown model), fall back to the mock with a WARNING instead of
            raising. Set it to ``False`` for runs whose numbers will be reported.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str | None = None,
        mock: bool = False,
        allow_fallback: bool = True,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.allow_fallback = allow_fallback
        self.batch_size = batch_size
        self.max_length = max_length
        self.fallback_reason: str | None = None
        self._model: Any = MockCrossEncoder() if mock else None
        self._identity: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            self._identity = torch.nn.Identity()
            logger.info("loaded cross-encoder %s", self.model_name)
        except Exception as exc:  # ImportError, OSError, HTTP/proxy errors from the hub, ...
            if not self.allow_fallback:
                raise RuntimeError(f"could not load cross-encoder {self.model_name!r}: {exc}") from exc
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("cross-encoder %s unavailable (%s); falling back to %s. Scores are a lexical "
                           "heuristic, not a neural model.", self.model_name, self.fallback_reason,
                           MockCrossEncoder.name)
            self._model = MockCrossEncoder()
        return self._model

    @property
    def is_mock(self) -> bool:
        return isinstance(self._load(), MockCrossEncoder)

    @property
    def backend(self) -> str:
        return MockCrossEncoder.name if self.is_mock else f"cross-encoder:{self.model_name}"

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        """Raw logits for each (query, text) pair, computed in batches."""
        if not texts:
            return np.zeros(0, dtype=np.float64)
        model = self._load()
        pairs = [(query, t) for t in texts]
        if isinstance(model, MockCrossEncoder):
            return model.predict(pairs, batch_size=self.batch_size)
        logits = model.predict(pairs, batch_size=self.batch_size, activation_fn=self._identity,
                               convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(logits, dtype=np.float64).reshape(-1)

    def rerank(self, query: str, documents: Sequence[Chunk | str], top_n: int | None = None) -> list[RerankedDocument]:
        """Rank ``documents`` by cross-encoder logit (ties keep input order)."""
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be positive")
        if not documents or not query or not query.strip():
            return []
        texts = [d.text if isinstance(d, Chunk) else str(d) for d in documents]
        logits = self.score(query, texts)
        probs = sigmoid(logits)
        order = sorted(range(len(texts)), key=lambda i: (-logits[i], i))
        if top_n is not None:
            order = order[:top_n]
        return [
            RerankedDocument(
                index=i,
                chunk_id=documents[i].chunk_id if isinstance(documents[i], Chunk) else None,
                text=texts[i],
                score=float(logits[i]),
                probability=float(probs[i]),
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
        ]


# ---------------------------------------------------------------------------
# Two-stage retriever
# ---------------------------------------------------------------------------


class RerankingRetriever:
    """Composes a first-stage ``HybridRetriever`` with a ``Reranker``.

    ``retrieve`` fetches ``max(retrieve_k, top_k)`` candidates with ``first_stage_mode``,
    re-scores them, and returns the top ``top_k``. In each ``RetrievalResult``,
    ``score``/``rank`` refer to the reranked order (``score`` is the cross-encoder logit).
    The first-stage rank, score and per-retriever fields are preserved for analysis
    and abstention features.
    """

    def __init__(self, first_stage: HybridRetriever, reranker: Reranker, *, retrieve_k: int = 50,
                 first_stage_mode: str = "hybrid") -> None:
        if retrieve_k <= 0:
            raise ValueError("retrieve_k must be positive")
        self.first_stage = first_stage
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.first_stage_mode = first_stage_mode

    @property
    def chunks(self) -> list[Chunk]:
        return self.first_stage.chunks

    @property
    def embedder(self):
        return self.first_stage.embedder

    def candidates(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """The first-stage pool: ``retrieve_k`` results, widened to ``top_k`` if that is larger."""
        k = self.retrieve_k if top_k is None else max(self.retrieve_k, top_k)
        return self.first_stage.retrieve(query, top_k=k, mode=self.first_stage_mode)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.retrieve_with_candidates(query, top_k)[1]

    def retrieve_with_candidates(self, query: str, top_k: int = 10) -> tuple[list[RetrievalResult],
                                                                           list[RetrievalResult]]:
        """``(first-stage candidates, reranked top_k)`` from a single first-stage call."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = self.candidates(query, top_k)
        if not candidates:
            return [], []
        reranked = self.reranker.rerank(query, [c.chunk for c in candidates], top_n=top_k)
        return candidates, [
            dataclasses.replace(
                candidates[r.index],
                score=r.score,
                rank=r.rank,
                first_stage_rank=candidates[r.index].rank,
                first_stage_score=candidates[r.index].score,
                rerank_score=r.score,
                rerank_probability=r.probability,
            )
            for r in reranked
        ]
