"""End-to-end selective RAG.

::

    query ─► HybridRetriever ─► Reranker ─► abstention features ─► policy g(x) ≥ τ ?
                                                                    │ no  → abstain (stage "policy")
                                                                    ▼ yes
                                              Generator(top-k_ctx context) ─► validated answer
                                                                    │ abstains / ungrounded → abstain (stage "generator")
                                                                    ▼
                                                     answer + chunk citations + verbatim quotes

There are two abstention points. The calibrated pre-generation gate (Phase 4) skips the
generator call entirely. The reader can also decline when the context lacks support.
``policy=None`` gives standard RAG, which always generates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.abstention import AbstentionPolicy, extract_features
from src.constants import ABSTENTION_ANSWER
from src.generator import GeneratedAnswer, Generator
from src.reranker import RerankingRetriever
from src.retriever import HybridRetriever, RetrievalResult


@dataclass(frozen=True)
class PipelineResponse:
    query: str
    answer: str
    abstained: bool
    abstention_stage: str | None  # "policy" | "generator" | None
    confidence: float | None  # g(x) from the policy; None without a policy
    citations: list[str]
    evidence_quotes: list[str]
    evidence: list[RetrievalResult]  # the context passed to the reader
    features: dict[str, float]
    backends: dict[str, str | None]
    latency_ms: dict[str, float] = field(default_factory=dict)
    generation: GeneratedAnswer | None = None


class SelectiveRAGPipeline:
    def __init__(self, retriever: RerankingRetriever | HybridRetriever, generator: Generator,
                 policy: AbstentionPolicy | None = None, *, k_ctx: int = 3, top_k: int = 10,
                 forced: bool = False) -> None:
        """``forced=True`` uses the no-abstention reader prompt (the standard-RAG reader)."""
        self.retriever, self.generator, self.policy = retriever, generator, policy
        self.k_ctx, self.top_k, self.forced = k_ctx, top_k, forced

    def backends(self) -> dict[str, str | None]:
        reranker = self.retriever.reranker.backend if isinstance(self.retriever, RerankingRetriever) else None
        first = self.retriever.first_stage if isinstance(self.retriever, RerankingRetriever) else self.retriever
        return {"embedder": getattr(first.embedder, "name", None), "reranker": reranker,
                "generator": self.generator.backend}

    def answer(self, query: str) -> PipelineResponse:
        t0 = time.perf_counter()
        ev = extract_features(query, self.retriever, k_ctx=self.k_ctx, top_k=self.top_k)
        t1 = time.perf_counter()
        latency = {"retrieve": (t1 - t0) * 1e3}
        confidence = None
        if self.policy is not None:
            decision = self.policy.decide(ev.features)
            confidence = decision.confidence
            gate_open = decision.answer
        else:
            gate_open = True
        common = dict(query=query, confidence=confidence, evidence=ev.context, features=ev.features,
                      backends=self.backends())
        if not ev.results or not gate_open:
            latency["generate"] = 0.0
            latency["total"] = (time.perf_counter() - t0) * 1e3
            return PipelineResponse(answer=ABSTENTION_ANSWER, abstained=True, abstention_stage="policy",
                                    citations=[], evidence_quotes=[], latency_ms=latency, **common)
        gen = self.generator.generate(query, [r.chunk for r in ev.context], forced=self.forced)
        t2 = time.perf_counter()
        latency["generate"] = (t2 - t1) * 1e3
        latency["total"] = (t2 - t0) * 1e3
        if gen.abstain:
            return PipelineResponse(answer=ABSTENTION_ANSWER, abstained=True, abstention_stage="generator",
                                    citations=[], evidence_quotes=[], latency_ms=latency, generation=gen, **common)
        return PipelineResponse(answer=gen.answer, abstained=False, abstention_stage=None, citations=gen.citations,
                                evidence_quotes=gen.evidence_quotes, latency_ms=latency, generation=gen, **common)
