"""Tests for the cross-encoder reranking stage (all offline; the real model is opt-in)."""

from __future__ import annotations

import logging
import sys
from types import ModuleType

import numpy as np
import pytest

from src.reranker import (
    DEFAULT_RERANKER_MODEL,
    MockCrossEncoder,
    RerankedDocument,
    Reranker,
    RerankingRetriever,
    sigmoid,
)
from src.retriever import HybridRetriever
from tests.conftest import CORPUS_TEXTS, make_chunks

# The benchmark phrasing: "test-set" + "full model" disambiguate it from the dev/ablation numbers.
QUERY = "What test-set macro-F1 does the full ZenoRank model reach on SciClaimQA?"
RELEVANT = "On the SciClaimQA test set, the full ZenoRank model reaches 78.4 macro-F1."
DEV = "On the SciClaimQA development split, ZenoRank records 80.1 macro-F1."
UNRELATED = "Beekeepers treat varroa mites during the winter months."


@pytest.fixture
def mock_reranker():
    return Reranker(mock=True)


# ---------------------------------------------------------------------------
# MockCrossEncoder behaviour
# ---------------------------------------------------------------------------


class TestMockCrossEncoder:
    def test_deterministic_across_calls_and_instances(self):
        pairs = [(QUERY, RELEVANT), (QUERY, DEV), (QUERY, UNRELATED)]
        a = MockCrossEncoder().predict(pairs)
        assert np.array_equal(a, MockCrossEncoder().predict(pairs))
        assert np.array_equal(a, MockCrossEncoder().predict(pairs))

    def test_relevant_beats_hard_negative_beats_unrelated(self):
        logits = MockCrossEncoder().predict([(QUERY, RELEVANT), (QUERY, DEV), (QUERY, UNRELATED)])
        assert logits[0] > logits[2] and logits[1] > logits[2]
        assert sigmoid(logits[2]) < 0.5

    def test_features_bounded(self):
        m = MockCrossEncoder()
        for doc in (RELEVANT, DEV, UNRELATED, "", "the of and"):
            for name, value in m.features(QUERY, doc).items():
                assert 0.0 <= value <= 1.0, (name, doc)

    def test_logit_range(self):
        m = MockCrossEncoder()
        logits = m.predict([(QUERY, d) for d in (RELEVANT, DEV, UNRELATED, QUERY)])
        assert np.all(logits >= m.BIAS) and np.all(logits <= m.BIAS + sum(m.WEIGHTS.values()) + 1e-9)

    def test_adding_adjacent_missing_term_raises_score(self):
        base = "The full ZenoRank model reaches 78.4 macro-F1."
        extended = "The full ZenoRank model reaches 78.4 macro-F1 on the SciClaimQA test set."
        low, high = MockCrossEncoder().predict([(QUERY, base), (QUERY, extended)])
        assert high > low

    def test_compact_evidence_beats_scattered_evidence(self):
        compact = "ZenoRank reaches 78.4 macro-F1 on SciClaimQA overall in our main experiments."
        scattered = ("ZenoRank is introduced first. Many unrelated remarks about compute budgets follow here. "
                     "Later we discuss SciClaimQA briefly. Finally the macro-F1 figure appears.")
        c, s = MockCrossEncoder().predict([(QUERY, compact), (QUERY, scattered)])
        assert c > s

    def test_appending_the_query_never_lowers_a_score(self):
        """Monotonicity property over every corpus passage: doc + query >= doc."""
        m = MockCrossEncoder()
        for doc in CORPUS_TEXTS + [RELEVANT, DEV, UNRELATED]:
            plain, augmented = m.predict([(QUERY, doc), (QUERY, doc + " " + QUERY)])
            assert augmented >= plain

    def test_fuzzy_matches_morphological_variants(self):
        f = MockCrossEncoder().features("optimise the retriever", "optimisation of retrievers")
        assert f["coverage"] == 0.0 and f["fuzzy"] > 0.4

    @pytest.mark.parametrize("query", ["", "   ", "the of and"])
    def test_degenerate_queries_score_at_bias(self, query):
        (logit,) = MockCrossEncoder().predict([(query, RELEVANT)])
        assert logit == MockCrossEncoder.BIAS


# ---------------------------------------------------------------------------
# Reranker interface
# ---------------------------------------------------------------------------


class TestReranker:
    def test_rerank_strings(self, mock_reranker):
        out = mock_reranker.rerank(QUERY, [UNRELATED, DEV, RELEVANT])
        assert [d.text for d in out][0] == RELEVANT
        assert [d.rank for d in out] == [1, 2, 3]
        assert all(isinstance(d, RerankedDocument) and d.chunk_id is None for d in out)
        assert [d.score for d in out] == sorted((d.score for d in out), reverse=True)
        for d in out:
            assert 0.0 < d.probability < 1.0
            assert d.probability == pytest.approx(float(sigmoid(d.score)))
        assert out[0].index == 2  # position in the input list

    def test_rerank_chunks_keeps_ids(self, mock_reranker):
        chunks = make_chunks([UNRELATED, RELEVANT, DEV])
        out = mock_reranker.rerank(QUERY, chunks)
        assert out[0].chunk_id == chunks[1].chunk_id
        assert {d.chunk_id for d in out} == {c.chunk_id for c in chunks}

    def test_top_n(self, mock_reranker):
        docs = [UNRELATED, DEV, RELEVANT]
        assert len(mock_reranker.rerank(QUERY, docs, top_n=2)) == 2
        assert len(mock_reranker.rerank(QUERY, docs, top_n=50)) == 3
        with pytest.raises(ValueError):
            mock_reranker.rerank(QUERY, docs, top_n=0)

    @pytest.mark.parametrize("query,docs", [(QUERY, []), ("", [RELEVANT]), ("   ", [RELEVANT])])
    def test_empty_inputs(self, mock_reranker, query, docs):
        assert mock_reranker.rerank(query, docs) == []

    def test_ties_keep_input_order(self, mock_reranker):
        out = mock_reranker.rerank(QUERY, [UNRELATED, UNRELATED, UNRELATED])
        assert [d.index for d in out] == [0, 1, 2]

    def test_batching_does_not_change_scores(self):
        docs = CORPUS_TEXTS + [RELEVANT, DEV, UNRELATED]
        small = Reranker(mock=True, batch_size=2).score(QUERY, docs)
        large = Reranker(mock=True, batch_size=64).score(QUERY, docs)
        assert np.array_equal(small, large)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            Reranker(mock=True, batch_size=0)

    def test_mock_flag_skips_imports(self, no_model_packages):
        r = Reranker(mock=True)
        assert r.backend == "mock-lexical-v1" and r.is_mock and r.fallback_reason is None


# ---------------------------------------------------------------------------
# Backend selection and graceful degradation
# ---------------------------------------------------------------------------


class FakeCrossEncoder:
    instances: list = []

    def __init__(self, model_name, device=None, max_length=None):
        self.model_name, self.device, self.max_length = model_name, device, max_length
        self.predict_kwargs = None
        FakeCrossEncoder.instances.append(self)

    def predict(self, pairs, batch_size=32, activation_fn=None, convert_to_numpy=True, show_progress_bar=None):
        self.predict_kwargs = {"batch_size": batch_size, "activation_fn": activation_fn}
        return np.array([float(len(doc)) / 10 for _, doc in pairs], dtype=np.float32)


class Identity:
    pass


@pytest.fixture
def fake_backend(monkeypatch):
    st = ModuleType("sentence_transformers")
    st.CrossEncoder = FakeCrossEncoder
    torch = ModuleType("torch")
    torch.nn = ModuleType("torch.nn")
    torch.nn.Identity = Identity
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "torch", torch)
    FakeCrossEncoder.instances = []
    return FakeCrossEncoder


class TestBackends:
    def test_real_backend_wiring(self, fake_backend):
        r = Reranker(device="cpu", max_length=256, batch_size=4)
        out = r.rerank("q", ["short", "a much longer passage"])
        model = fake_backend.instances[0]
        assert r.backend == f"cross-encoder:{DEFAULT_RERANKER_MODEL}" and not r.is_mock
        assert (model.model_name, model.device, model.max_length) == (DEFAULT_RERANKER_MODEL, "cpu", 256)
        assert isinstance(model.predict_kwargs["activation_fn"], Identity)  # raw logits requested
        assert model.predict_kwargs["batch_size"] == 4
        assert out[0].text == "a much longer passage"
        assert out[0].score == pytest.approx(2.1) and out[0].probability == pytest.approx(float(sigmoid(2.1)))

    def test_model_is_loaded_once_and_lazily(self, fake_backend):
        r = Reranker()
        assert fake_backend.instances == []
        r.rerank("q", ["a"])
        r.rerank("q", ["b"])
        assert len(fake_backend.instances) == 1

    def test_falls_back_when_packages_missing(self, no_model_packages, caplog):
        with caplog.at_level(logging.WARNING, logger="src.reranker"):
            r = Reranker()
            out = r.rerank(QUERY, [UNRELATED, RELEVANT])
        assert r.is_mock and r.backend == "mock-lexical-v1"
        assert "ImportError" in r.fallback_reason
        assert any("falling back" in rec.message for rec in caplog.records)
        assert out[0].text == RELEVANT

    def test_falls_back_when_model_cannot_be_downloaded(self, fake_backend, monkeypatch):
        def offline(*args, **kwargs):
            raise OSError("We couldn't connect to 'https://huggingface.co'")

        monkeypatch.setattr(fake_backend, "__init__", offline)
        r = Reranker()
        assert r.is_mock and r.fallback_reason.startswith("OSError")

    def test_strict_mode_raises(self, no_model_packages):
        with pytest.raises(RuntimeError, match="could not load cross-encoder"):
            Reranker(allow_fallback=False).rerank(QUERY, [RELEVANT])


# ---------------------------------------------------------------------------
# HybridRetriever -> Reranker integration
# ---------------------------------------------------------------------------


@pytest.fixture
def two_stage(corpus_chunks, hashing_embedder, mock_reranker):
    first = HybridRetriever(hashing_embedder).index(corpus_chunks)
    return RerankingRetriever(first, mock_reranker, retrieve_k=5)


class TestRerankingRetriever:
    def test_pipeline_output(self, two_stage):
        results = two_stage.retrieve("How does conformal prediction guarantee coverage?", top_k=3)
        assert len(results) == 3
        assert results[0].chunk.chunk_index == 2
        assert [r.rank for r in results] == [1, 2, 3]
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
        for r in results:
            assert r.score == r.rerank_score
            assert r.rerank_probability == pytest.approx(float(sigmoid(r.rerank_score)))
            assert r.first_stage_rank is not None and r.first_stage_score is not None
            assert r.dense_rank is not None  # first-stage per-retriever signals preserved

    def test_candidates_are_the_first_stage_top_k(self, two_stage):
        query = "ranked lists and passage relevance"
        first = {r.chunk.chunk_id for r in two_stage.first_stage.retrieve(query, top_k=5)}
        assert {r.chunk.chunk_id for r in two_stage.retrieve(query, top_k=5)} == first
        assert {r.chunk.chunk_id for r in two_stage.candidates(query)} == first

    def test_top_k_larger_than_retrieve_k_widens_the_pool(self, two_stage, corpus_chunks):
        assert len(two_stage.retrieve("retrieval", top_k=len(corpus_chunks))) == len(corpus_chunks)

    def test_sparse_first_stage(self, corpus_chunks, hashing_embedder, mock_reranker):
        first = HybridRetriever(hashing_embedder).index(corpus_chunks)
        rr = RerankingRetriever(first, mock_reranker, retrieve_k=5, first_stage_mode="sparse")
        assert all(r.dense_rank is None for r in rr.retrieve("faiss vector search", top_k=3))
        assert rr.retrieve("zebra giraffe", top_k=3) == []  # no lexical candidates at all

    def test_edge_cases(self, two_stage):
        assert two_stage.retrieve("", top_k=3) == []
        with pytest.raises(ValueError):
            two_stage.retrieve("coverage", top_k=0)
        with pytest.raises(ValueError):
            RerankingRetriever(two_stage.first_stage, two_stage.reranker, retrieve_k=0)

    def test_exposes_first_stage_attributes(self, two_stage, corpus_chunks):
        assert two_stage.chunks == corpus_chunks
        assert two_stage.embedder is two_stage.first_stage.embedder


@pytest.mark.integration
def test_real_cross_encoder_ranks_relevant_passage_first():
    r = Reranker(allow_fallback=False)
    out = r.rerank("How many people live in Berlin?",
                   ["Berlin is well known for its museums.", "Berlin had a population of 3,520,031 in 2019."])
    assert r.backend.startswith("cross-encoder:")
    assert out[0].text.startswith("Berlin had a population")
    assert out[0].score > 0 > out[1].score
