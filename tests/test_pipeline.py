"""Tests for the end-to-end selective RAG pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.abstention import FEATURE_NAMES, AbstentionPolicy, LogisticCalibrator
from src.constants import ABSTENTION_ANSWER
from src.generator import Generator
from src.pipeline import SelectiveRAGPipeline
from src.reranker import Reranker, RerankingRetriever
from src.retriever import HybridRetriever
from tests.conftest import make_chunks

TEXTS = [
    "The backbone of ZenoRank is a pretrained RoBERTa encoder.",
    "We train ZenoRank for 10 epochs with a batch size of 32.",
    "On the SciClaimQA test set, the full ZenoRank model attains 78.4 macro-F1.",
    "Conformal prediction provides distribution-free coverage guarantees.",
    "Reciprocal rank fusion merges ranked lists from several retrievers.",
]
QUESTION = "For how many epochs is ZenoRank trained?"


class CountingGenerator(Generator):
    def __init__(self):
        super().__init__("mock")
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return super().generate(*args, **kwargs)


class CountingRetriever(HybridRetriever):
    calls = 0

    def retrieve(self, *args, **kwargs):
        CountingRetriever.calls += 1
        return super().retrieve(*args, **kwargs)


@pytest.fixture
def retriever(hashing_embedder):
    first = CountingRetriever(hashing_embedder).index(make_chunks(TEXTS, arxiv_id="p1", title="ZenoRank"))
    return RerankingRetriever(first, Reranker(mock=True), retrieve_k=5)


def policy(tau: float) -> AbstentionPolicy:
    rng = np.random.default_rng(0)
    rows = [{f: float(v) for f in FEATURE_NAMES} for v in rng.normal(size=40)]
    labels = [int(r["ce_top1"] > 0) for r in rows]
    return AbstentionPolicy(LogisticCalibrator(list(FEATURE_NAMES)).fit(rows, labels), tau)


class TestPipeline:
    def test_standard_rag_answers_with_citations(self, retriever):
        out = SelectiveRAGPipeline(retriever, Generator("mock")).answer(QUESTION)
        assert not out.abstained and out.abstention_stage is None and "10 epochs" in out.answer
        assert out.citations and set(out.citations) <= {r.chunk.chunk_id for r in out.evidence}
        assert out.evidence_quotes and out.confidence is None
        assert len(out.evidence) == 3 and list(out.features) == list(FEATURE_NAMES)
        assert out.backends == {"embedder": "hashing-256", "reranker": "mock-lexical-v1",
                                "generator": "mock-extractive-v1"}
        assert {"retrieve", "generate", "total"} <= set(out.latency_ms)
        assert out.latency_ms["total"] >= out.latency_ms["retrieve"]

    def test_policy_abstention_skips_the_generator(self, retriever):
        gen = CountingGenerator()
        out = SelectiveRAGPipeline(retriever, gen, policy(math.inf)).answer(QUESTION)
        assert out.abstained and out.abstention_stage == "policy" and out.answer == ABSTENTION_ANSWER
        assert gen.calls == 0 and out.citations == [] and 0 <= out.confidence <= 1
        assert out.latency_ms["generate"] == 0.0

    def test_open_gate_always_generates(self, retriever):
        gen = CountingGenerator()
        out = SelectiveRAGPipeline(retriever, gen, policy(-math.inf)).answer(QUESTION)
        assert gen.calls == 1 and not out.abstained and out.confidence is not None

    def test_generator_abstention_is_reported(self, retriever):
        out = SelectiveRAGPipeline(retriever, Generator("mock")).answer("Which grape varieties dominate Rioja wines?")
        assert out.abstained and out.abstention_stage == "generator" and out.generation.reason.startswith("low_support")

    def test_forced_pipeline_never_self_abstains(self, retriever):
        out = SelectiveRAGPipeline(retriever, Generator("mock"), forced=True).answer(
            "Which grape varieties dominate Rioja wines?")
        assert not out.abstained and out.answer  # standard RAG hallucinates an answer

    def test_one_first_stage_retrieval_per_query(self, retriever):
        CountingRetriever.calls = 0
        SelectiveRAGPipeline(retriever, Generator("mock"), policy(-math.inf)).answer(QUESTION)
        assert CountingRetriever.calls == 1

    def test_empty_query(self, retriever):
        gen = CountingGenerator()
        out = SelectiveRAGPipeline(retriever, gen).answer("")
        assert out.abstained and out.abstention_stage == "policy" and gen.calls == 0

    def test_works_without_reranker(self, hashing_embedder):
        first = HybridRetriever(hashing_embedder).index(make_chunks(TEXTS))
        out = SelectiveRAGPipeline(first, Generator("mock")).answer(QUESTION)
        assert out.backends["reranker"] is None and "10 epochs" in out.answer
