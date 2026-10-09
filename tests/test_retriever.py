"""Tests for the hybrid retrieval engine: tokenizer, BM25, FAISS, RRF and the HybridRetriever."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.retriever import (
    BGE_QUERY_INSTRUCTION,
    BM25Index,
    DenseIndex,
    Embedder,
    HybridRetriever,
    SentenceTransformerEmbedder,
    reciprocal_rank_fusion,
    tokenize,
)
from tests.conftest import CORPUS_TEXTS, FixedEmbedder, HashingEmbedder, make_chunks

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize("Hybrid-Retrieval, with BM25!") == ["hybrid", "retrieval", "bm25"]

    def test_removes_stopwords(self):
        assert tokenize("what is the coverage of the model") == ["coverage", "model"]
        assert tokenize("the of and", remove_stopwords=False) == ["the", "of", "and"]

    def test_unicode_normalisation(self):
        assert tokenize("ﬁne-tuning Café") == ["fine", "tuning", "café"]

    @pytest.mark.parametrize("text", ["", "   ", "!!! ... ???", "the a of"])
    def test_degenerate_inputs(self, text):
        assert tokenize(text) == []


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class TestBM25:
    def test_indexing_and_relevance(self):
        index = BM25Index().build(CORPUS_TEXTS)
        assert len(index) == len(CORPUS_TEXTS)
        hits = index.search("reciprocal rank fusion of ranked lists", k=3)
        assert hits[0][0] == 3
        assert all(a[1] >= b[1] for a, b in zip(hits, hits[1:]))

    def test_zero_overlap_returns_nothing(self):
        assert BM25Index().build(CORPUS_TEXTS).search("zebra giraffe safari", k=5) == []

    def test_only_overlapping_documents_returned(self):
        hits = BM25Index().build(CORPUS_TEXTS).search("faiss", k=10)
        assert [i for i, _ in hits] == [1]

    def test_k_limits_results(self):
        hits = BM25Index().build(CORPUS_TEXTS).search("query passage encode", k=1)
        assert len(hits) == 1

    @pytest.mark.parametrize("query", ["", "   ", "the of and"])
    def test_empty_or_stopword_query(self, query):
        assert BM25Index().build(CORPUS_TEXTS).search(query, k=5) == []

    @pytest.mark.parametrize("corpus", [[], ["", "the of"]])
    def test_empty_corpus_is_safe(self, corpus):
        index = BM25Index().build(corpus)
        assert index.search("anything", k=5) == []

    def test_non_positive_k(self):
        assert BM25Index().build(CORPUS_TEXTS).search("faiss", k=0) == []


# ---------------------------------------------------------------------------
# Dense (FAISS)
# ---------------------------------------------------------------------------


class TestDenseIndex:
    def test_build_and_self_retrieval(self, hashing_embedder):
        vectors = hashing_embedder.encode(CORPUS_TEXTS)
        index = DenseIndex().build(vectors)
        assert len(index) == len(CORPUS_TEXTS)
        assert index.dim == hashing_embedder.dim
        for i, v in enumerate(vectors):
            top_id, top_score = index.search(v, k=1)[0]
            assert top_id == i
            assert top_score == pytest.approx(1.0, abs=1e-5)

    def test_vectors_are_normalised_for_cosine(self):
        index = DenseIndex().build(np.array([[3.0, 4.0], [0.0, 10.0]]))
        hits = dict(index.search(np.array([30.0, 40.0]), k=2))
        assert hits[0] == pytest.approx(1.0, abs=1e-6)  # magnitude-invariant
        assert hits[1] == pytest.approx(0.8, abs=1e-6)

    def test_k_larger_than_corpus(self, hashing_embedder):
        index = DenseIndex().build(hashing_embedder.encode(CORPUS_TEXTS))
        hits = index.search(hashing_embedder.encode(["dense retrieval"])[0], k=100)
        assert sorted(i for i, _ in hits) == list(range(len(CORPUS_TEXTS)))

    def test_dimension_mismatch_raises(self, hashing_embedder):
        index = DenseIndex().build(hashing_embedder.encode(CORPUS_TEXTS))
        with pytest.raises(ValueError, match="dim"):
            index.search(np.ones(7, dtype=np.float32), k=1)

    @pytest.mark.parametrize("bad", [np.zeros((0, 8)), np.zeros(8)])
    def test_rejects_invalid_matrix(self, bad):
        with pytest.raises(ValueError):
            DenseIndex().build(bad)

    def test_unbuilt_index(self):
        assert DenseIndex().search(np.ones(4), k=3) == []
        with pytest.raises(RuntimeError):
            DenseIndex().save("/nonexistent/path")


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestRRF:
    def test_hand_computed_example(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
        expected = {
            "a": 1 / 61 + 1 / 62,
            "b": 1 / 62,
            "c": 1 / 63 + 1 / 61,
            "d": 1 / 63,
        }
        assert [doc for doc, _ in fused] == ["a", "c", "b", "d"]
        for doc, score in fused:
            assert score == pytest.approx(expected[doc])

    def test_k_controls_consensus_vs_top_rank(self):
        # "a" is top-1 in one list only; "b" is rank 3 in both lists.
        lists = [["a", "c", "b"], ["d", "e", "b"]]
        assert reciprocal_rank_fusion(lists, k=0)[0][0] == "a"  # sharp: one top rank wins
        assert reciprocal_rank_fusion(lists, k=60)[0][0] == "b"  # smooth: consensus wins

    def test_default_k_is_60(self):
        (doc, score), = reciprocal_rank_fusion([["x"]])
        assert score == pytest.approx(1 / 61)

    def test_weights(self):
        lists = [["a", "b"], ["b", "a"]]
        assert reciprocal_rank_fusion(lists, weights=[1.0, 1.0])[0][1] == pytest.approx(1 / 61 + 1 / 62)
        assert reciprocal_rank_fusion(lists, weights=[2.0, 1.0])[0][0] == "a"
        assert reciprocal_rank_fusion(lists, weights=[1.0, 2.0])[0][0] == "b"
        assert [d for d, _ in reciprocal_rank_fusion(lists, weights=[0.0, 1.0])] == ["b", "a"]

    def test_disjoint_lists_interleave(self):
        fused = reciprocal_rank_fusion([["a1", "a2"], ["b1", "b2"]])
        assert [d for d, _ in fused] == ["a1", "b1", "a2", "b2"]

    def test_ties_are_deterministic_and_order_independent(self):
        assert reciprocal_rank_fusion([["y"], ["x"]]) == reciprocal_rank_fusion([["x"], ["y"]])
        assert [d for d, _ in reciprocal_rank_fusion([["y"], ["x"]])] == ["x", "y"]

    def test_duplicates_within_a_list_count_once(self):
        fused = dict(reciprocal_rank_fusion([["a", "a", "b"]]))
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 63)  # rank position is preserved

    @pytest.mark.parametrize("lists", [[], [[]], [[], []]])
    def test_empty_inputs(self, lists):
        assert reciprocal_rank_fusion(lists) == []

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], k=-1)
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"]], weights=[-1.0])

    def test_scores_are_monotone_in_rank(self):
        fused = reciprocal_rank_fusion([[str(i) for i in range(100)]])
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)
        assert math.isclose(scores[0], 1 / 61) and math.isclose(scores[-1], 1 / 160)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


@pytest.fixture
def retriever(corpus_chunks, hashing_embedder):
    return HybridRetriever(hashing_embedder).index(corpus_chunks)


class TestHybridRetriever:
    def test_hashing_embedder_satisfies_protocol(self, hashing_embedder):
        assert isinstance(hashing_embedder, Embedder)

    def test_indexing(self, retriever, corpus_chunks, hashing_embedder):
        assert len(retriever) == len(corpus_chunks)
        assert len(retriever.bm25) == len(retriever.dense) == len(corpus_chunks)
        assert hashing_embedder.calls[0] == (len(corpus_chunks), False)  # passages, not queries

    @pytest.mark.parametrize(
        "query,expected_idx",
        [
            ("How does conformal prediction guarantee coverage?", 2),
            ("vector similarity search with FAISS embeddings", 1),
            ("When should a model abstain under low confidence?", 4),
            ("cross-encoder reranker relevance", 5),
        ],
    )
    def test_mock_queries_rank_correct_chunk_first(self, retriever, query, expected_idx):
        top = retriever.retrieve(query, top_k=3)[0]
        assert top.chunk.chunk_index == expected_idx
        assert top.rank == 1
        assert top.bm25_rank is not None and top.dense_rank is not None

    def test_queries_encoded_as_queries(self, retriever, hashing_embedder):
        retriever.retrieve("coverage", top_k=1)
        assert hashing_embedder.calls[-1] == (1, True)

    def test_fused_order_matches_rrf(self, retriever):
        query = "ranked lists of documents and passage relevance"
        ids = [c.chunk_id for c in retriever.chunks]
        bm25 = [ids[i] for i, _ in retriever.bm25.search(query, 50)]
        q_vec = retriever.embedder.encode([query], is_query=True)[0]
        dense = [ids[i] for i, _ in retriever.dense.search(q_vec, 50)]
        expected = reciprocal_rank_fusion([bm25, dense], k=60)

        results = retriever.retrieve(query, top_k=len(expected))
        assert [r.chunk.chunk_id for r in results] == [d for d, _ in expected]
        for r, (_, score) in zip(results, expected):
            assert r.score == pytest.approx(score)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_dense_only_document_still_retrieved(self):
        # BM25 matches only doc 0 and dense prefers doc 1, so fusion must surface both.
        texts = ["quantum annealing schedule", "adiabatic optimisation hardware", "medieval poetry"]
        query = "quantum speedup"
        emb = FixedEmbedder({texts[0]: [0.6, 0.8, 0], texts[1]: [1, 0, 0], texts[2]: [0, 0, 1],
                             query: [1, 0, 0]})
        r = HybridRetriever(emb).index(make_chunks(texts))
        results = {res.chunk.chunk_index: res for res in r.retrieve(query, top_k=3)}
        assert results[0].bm25_rank == 1 and results[0].dense_rank == 2
        assert results[1].bm25_rank is None and results[1].dense_rank == 1
        assert results[2].bm25_rank is None
        assert results[0].rank == 1  # agreement of both retrievers wins
        assert results[1].rank == 2

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_empty_query_returns_empty(self, retriever, query):
        assert retriever.retrieve(query, top_k=5) == []

    def test_zero_lexical_overlap_falls_back_to_dense(self, retriever):
        results = retriever.retrieve("zebra giraffe safari", top_k=3)
        assert len(results) == 3
        assert all(r.bm25_rank is None and r.bm25_score is None for r in results)
        assert all(r.dense_rank is not None for r in results)

    def test_stopword_only_query(self, retriever):
        results = retriever.retrieve("what is the", top_k=2)
        assert len(results) == 2 and all(r.bm25_rank is None for r in results)

    def test_top_k_larger_than_corpus(self, retriever, corpus_chunks):
        results = retriever.retrieve("retrieval", top_k=100)
        assert len(results) == len(corpus_chunks)
        assert len({r.chunk.chunk_id for r in results}) == len(corpus_chunks)

    def test_top_k_beyond_candidate_pool(self, corpus_chunks, hashing_embedder):
        r = HybridRetriever(hashing_embedder, candidate_pool=2).index(corpus_chunks)
        assert len(r.retrieve("retrieval", top_k=5)) == 5

    @pytest.mark.parametrize("top_k", [0, -3])
    def test_invalid_top_k(self, retriever, top_k):
        with pytest.raises(ValueError):
            retriever.retrieve("coverage", top_k=top_k)

    def test_retrieve_before_index(self, hashing_embedder):
        with pytest.raises(RuntimeError):
            HybridRetriever(hashing_embedder).retrieve("coverage")

    def test_index_validation(self, hashing_embedder, corpus_chunks):
        with pytest.raises(ValueError, match="empty"):
            HybridRetriever(hashing_embedder).index([])
        with pytest.raises(ValueError, match="duplicate"):
            HybridRetriever(hashing_embedder).index(corpus_chunks + corpus_chunks[:1])

    @pytest.mark.parametrize("kwargs", [{"rrf_k": -1}, {"candidate_pool": 0}, {"weights": (1.0,)}])
    def test_invalid_config(self, hashing_embedder, kwargs):
        with pytest.raises(ValueError):
            HybridRetriever(hashing_embedder, **kwargs)

    def test_save_load_round_trip(self, retriever, tmp_path):
        retriever.save(tmp_path / "index")
        loaded = HybridRetriever.load(tmp_path / "index", HashingEmbedder())
        assert loaded.chunks == retriever.chunks
        assert loaded.rrf_k == retriever.rrf_k
        for query in ["conformal coverage", "zebra", "ranked lists fusion"]:
            assert loaded.retrieve(query, top_k=4) == retriever.retrieve(query, top_k=4)

    def test_save_requires_index(self, hashing_embedder, tmp_path):
        with pytest.raises(RuntimeError):
            HybridRetriever(hashing_embedder).save(tmp_path)


def test_bge_query_instruction_auto_detection():
    assert SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5").query_instruction == BGE_QUERY_INSTRUCTION
    assert SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2").query_instruction == ""
    assert SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", query_instruction="").query_instruction == ""


@pytest.mark.integration
def test_real_bge_embedder_end_to_end(corpus_chunks):
    embedder = SentenceTransformerEmbedder()
    retriever = HybridRetriever(embedder).index(corpus_chunks)
    vectors = embedder.encode(CORPUS_TEXTS)
    assert vectors.shape == (len(CORPUS_TEXTS), 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    # Paraphrase with little lexical overlap: dense semantics must carry it.
    top = retriever.retrieve("statistical guarantees for prediction sets", top_k=1)[0]
    assert top.chunk.chunk_index == 2


class TestRetrievalModes:
    def test_sparse_mode_matches_bm25(self, retriever):
        query = "reciprocal rank fusion of ranked lists"
        expected = retriever.bm25.search(query, 3)
        results = retriever.retrieve(query, top_k=3, mode="sparse")
        assert [(r.chunk.chunk_index, r.score) for r in results] == [(i, s) for i, s in expected]
        assert all(r.dense_rank is None and r.bm25_rank == r.rank for r in results)

    def test_dense_mode_matches_faiss(self, retriever):
        query = "conformal coverage guarantees"
        q_vec = retriever.embedder.encode([query], is_query=True)[0]
        # retrieve() searches the full candidate pool, so compare against the same depth
        # (FAISS breaks score ties differently for different k).
        expected = retriever.dense.search(q_vec, retriever.candidate_pool)[:4]
        results = retriever.retrieve(query, top_k=4, mode="dense")
        assert [r.chunk.chunk_index for r in results] == [i for i, _ in expected]
        assert [r.score for r in results] == pytest.approx([s for _, s in expected])
        assert all(r.bm25_rank is None and r.dense_rank == r.rank for r in results)

    def test_sparse_mode_skips_the_embedder(self, retriever, hashing_embedder):
        n_calls = len(hashing_embedder.calls)
        retriever.retrieve("faiss", top_k=2, mode="sparse")
        assert len(hashing_embedder.calls) == n_calls

    def test_sparse_zero_overlap_is_empty(self, retriever):
        assert retriever.retrieve("zebra giraffe", top_k=3, mode="sparse") == []

    def test_hybrid_is_default(self, retriever):
        assert retriever.retrieve("coverage", top_k=3) == retriever.retrieve("coverage", top_k=3, mode="hybrid")

    def test_invalid_mode(self, retriever):
        with pytest.raises(ValueError, match="mode"):
            retriever.retrieve("coverage", mode="colbert")
