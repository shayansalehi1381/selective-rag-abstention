"""Tests for the end-to-end benchmark: answer metrics, correctness, systems and CLI."""

from __future__ import annotations

import json

import pytest

from eval.e2e_eval import (
    exact_match,
    is_correct,
    key_fact_match,
    key_facts,
    main as e2e_main,
    normalize_answer,
    render_markdown,
    run_e2e_benchmark,
    score_items,
    token_f1,
)
from eval.schemas import EvalSet
from src.data_loader import load_chunks_jsonl
from src.generator import Generator
from src.reranker import Reranker, RerankingRetriever
from src.retriever import HashingEmbedder, HybridRetriever

EVAL_SET = "data/eval/eval_set.json"
CORPUS = "data/eval/synthetic_corpus.jsonl"


class TestAnswerMetrics:
    def test_normalize(self):
        assert normalize_answer("The  78.4 Macro-F1!") == "784 macrof1"
        assert normalize_answer("a learning rate of 2e-5") == "learning rate of 2e5"

    def test_em_and_f1(self):
        assert exact_match("78.4 macro-F1", "78.4 Macro-F1.") == 1.0
        assert token_f1("78.4", "78.4 macro-F1") == pytest.approx(2 / 3)
        assert token_f1("model attains 78.4 macro-F1", "78.4 macro-F1") == pytest.approx(2 * 0.5 * 1 / 1.5)
        assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0 and token_f1("cat", "dog") == 0.0

    def test_key_facts(self):
        assert key_facts("78.4 macro-F1") == ["78.4", "macro-F1"]
        assert key_facts("a learning rate of 3e-5") == ["3e-5"]
        assert key_facts("BioHopQA (49.4 vs 49.0, a gap of 0.4 nDCG@10)") == ["49.4", "49.0", "0.4", "BioHopQA",
                                                                               "nDCG@10"]
        assert key_facts("ELECTRA") == ["ELECTRA"] and key_facts("30 epochs") == ["30"]

    def test_key_fact_match(self):
        assert key_fact_match("reaches 78.4 macro-F1 on it", "78.4 macro-F1") == 1.0
        assert key_fact_match("reaches 78.4", "78.4 macro-F1") == 0.0
        assert key_fact_match("reaches 77.4 macro-F1", "78.4 macro-F1") == 0.0

    def test_correctness_needs_both(self):
        assert is_correct("ZenoRank model attains 78.4 macro-F1", "78.4 macro-F1")
        assert not is_correct("78.4", "78.4 macro-F1")  # F1 ok, but the metric entity is missing
        assert not is_correct("the full model was evaluated thoroughly and attains 78.4 macro-F1 on it",
                              "78.4 macro-F1")  # all key facts, but F1 < 0.5


@pytest.fixture(scope="module")
def eval_set():
    return EvalSet.load(EVAL_SET)


@pytest.fixture(scope="module")
def rows(eval_set):
    first = HybridRetriever(HashingEmbedder()).index(load_chunks_jsonl(CORPUS))
    return score_items(eval_set, RerankingRetriever(first, Reranker(mock=True)), Generator("mock"))


@pytest.fixture(scope="module")
def results(rows, eval_set):
    return run_e2e_benchmark(rows, eval_set, n_splits=5, example_alpha=0.2)


class TestBenchmark:
    def test_item_rows(self, rows):
        assert len(rows) == 100
        for r in rows:
            for mode in ("free", "forced"):
                assert r[mode]["correct"] in (True, False)
                if not r["is_answerable"]:
                    assert not r[mode]["correct"]
            assert r["forced"]["answered"]  # the forced mock reader always answers

    def test_standard_rag_hallucinates_on_every_unanswerable_item(self, results):
        s = results["systems"]["standard_rag"]
        assert s["coverage"]["mean"] == 1.0 and s["hallucination_rate"]["mean"] == 1.0

    def test_gate_reduces_hallucination(self, results):
        base = results["systems"]["standard_rag"]["hallucination_rate"]["mean"]
        for name, s in results["systems"].items():
            if name.startswith("selective_rag"):
                assert s["hallucination_rate"]["mean"] <= base
        d = results["deltas_vs_standard_rag"]["selective_rag[erm,α=0.2]"]["hallucination_rate"]
        assert d["mean"] < 0 and d["share_negative"] == 1.0

    def test_metric_ranges(self, results):
        for s in results["systems"].values():
            for key in ("coverage", "hallucination_rate", "accuracy", "wrong_answer_rate"):
                v = s[key]["mean"]
                assert v is None or 0.0 <= v <= 1.0

    def test_full_grid_and_examples(self, results):
        names = set(results["systems"])
        for a in ("0.1", "0.2", "0.3"):
            for base in ("selective_rag", "selective_rag_self"):
                assert {f"{base}[erm,α={a}]", f"{base}[ltt,α={a}]"} <= names
        assert results["example_gate"] == "selective_rag[erm,α=0.2]"
        assert all(not e["category"].startswith("in_domain") for e in results["hallucination_examples"])

    def test_by_category(self, results):
        assert results["by_category"]["forced"]["out_of_domain"]["correct"] == 0
        assert sum(c["n"] for c in results["by_category"]["free"].values()) == 100

    def test_deterministic(self, rows, eval_set, results):
        again = run_e2e_benchmark(rows, eval_set, n_splits=5, example_alpha=0.2)
        assert json.dumps(again, sort_keys=True, default=str) == json.dumps(results, sort_keys=True, default=str)


class TestCli:
    def test_writes_reports(self, tmp_path, capsys):
        out = tmp_path / "e2e.json"
        assert e2e_main(["run", "--embedder", "hashing", "--reranker", "mock", "--generator", "mock",
                         "--n-splits", "3", "--no-cache", "--output", str(out)]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))  # strict JSON
        assert report["config"]["generator_backend"] == "mock-extractive-v1"
        assert report["config"]["headline_alpha"] == 0.2 and len(report["items"]) == 100
        md = out.with_suffix(".md").read_text(encoding="utf-8")
        assert md.startswith("# End-to-End Selective RAG Benchmark") and "Headline comparison (α = 0.2" in md
        assert "**mock extractive**" in md
        assert md == render_markdown(report)

    def test_default_generator_falls_back(self, tmp_path, no_llm_sdk):
        out = tmp_path / "e2e.json"
        e2e_main(["run", "--embedder", "hashing", "--reranker", "mock", "--n-splits", "2", "--no-cache",
                  "--output", str(out)])
        cfg = json.loads(out.read_text(encoding="utf-8"))["config"]
        assert cfg["generator_backend"] == "mock-extractive-v1" and "ImportError" in cfg["generator_fallback_reason"]

    def test_strict_generator_fails_clearly(self, tmp_path, no_llm_sdk):
        with pytest.raises(SystemExit, match="strict-generator"):
            e2e_main(["run", "--embedder", "hashing", "--reranker", "mock", "--strict-generator", "--no-cache",
                      "--n-splits", "2", "--output", str(tmp_path / "e.json")])

    def test_headline_alpha_is_added_to_grid(self, tmp_path):
        out = tmp_path / "e2e.json"
        e2e_main(["run", "--embedder", "hashing", "--reranker", "mock", "--generator", "mock", "--n-splits", "2",
                  "--no-cache", "--alphas", "0.1", "--headline-alpha", "0.25", "--output", str(out)])
        assert json.loads(out.read_text(encoding="utf-8"))["config"]["alphas"] == [0.1, 0.25]
