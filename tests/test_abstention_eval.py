"""Tests for the selective-prediction metrics, splits and the abstention benchmark harness."""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest

from eval.abstention_eval import (
    aurc,
    brier,
    eaurc,
    ece,
    fit_policy,
    main as abstention_main,
    oracle_aurc,
    reliability_bins,
    render_markdown,
    risk_coverage_curve,
    run_abstention_benchmark,
    selective_accuracy_at,
    stratified_splits,
    write_figures,
)
from eval.schemas import EvalSet
from src.abstention import CE_FEATURES, AbstentionPolicy
from src.data_loader import load_chunks_jsonl
from src.reranker import Reranker, RerankingRetriever
from src.retriever import HashingEmbedder, HybridRetriever

EVAL_SET = "data/eval/eval_set.json"
CORPUS = "data/eval/synthetic_corpus.jsonl"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_risk_coverage_curve(self):
        cov, risk = risk_coverage_curve([0.9, 0.8, 0.3, 0.1], [1, 0, 1, 0])
        assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
        assert risk.tolist() == pytest.approx([0.0, 0.5, 1 / 3, 0.5])

    def test_aurc_perfect_ranking_equals_oracle(self):
        y = [1, 1, 1, 0, 0]
        perfect = [0.9, 0.8, 0.7, 0.2, 0.1]
        assert aurc(perfect, y) == pytest.approx(oracle_aurc(y))
        assert eaurc(perfect, y) == pytest.approx(0.0)
        assert eaurc(perfect[::-1], y) > 0.3

    def test_oracle_aurc_hand_computed(self):
        # n=4, 2 safe: risks at k=1..4 are 0, 0, 1/3, 1/2
        assert oracle_aurc([1, 0, 1, 0]) == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)

    def test_ties_are_not_rewarded(self):
        y = [1, 0, 1, 0]
        assert aurc([0.5] * 4, y) == pytest.approx(0.5)  # all tied: base risk everywhere

    def test_selective_accuracy(self):
        s, y = [0.9, 0.8, 0.3, 0.1], [1, 0, 1, 0]
        assert selective_accuracy_at(s, y, 1.0) == 0.5
        assert selective_accuracy_at(s, y, 0.25) == 1.0
        assert selective_accuracy_at(s, y, 0.5) == 0.5
        with pytest.raises(ValueError):
            selective_accuracy_at(s, y, 0.0)

    def test_brier_and_ece(self):
        assert brier([1.0, 0.0], [1, 0]) == 0.0
        assert brier([0.5, 0.5], [1, 0]) == 0.25
        assert ece([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0
        # two items at 0.75 (bin 7), one safe: |0.5 - 0.75| = 0.25
        assert ece([0.75, 0.75], [1, 0]) == pytest.approx(0.25)

    def test_reliability_bins(self):
        bins = reliability_bins([0.05, 0.95, 1.0], [0, 1, 1], n_bins=10)
        assert len(bins) == 10 and bins[0]["count"] == 1 and bins[9]["count"] == 2
        assert bins[5]["mean_confidence"] is None


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_set():
    return EvalSet.load(EVAL_SET)


class TestSplits:
    def test_disjoint_cover_and_stratified(self, eval_set):
        items = eval_set.items
        for train, cal, test in stratified_splits(items, n_splits=5):
            assert not set(train) & set(cal) and not set(cal) & set(test) and not set(train) & set(test)
            assert sorted(train + cal + test) == list(range(len(items)))
            assert (len(train), len(cal), len(test)) == (40, 30, 30)
            for part, frac in [(train, 0.4), (cal, 0.3), (test, 0.3)]:
                counts = Counter(items[i].category.value for i in part)
                for cat, n in eval_set.counts.items():
                    assert abs(counts.get(cat, 0) - n * frac) <= 1

    def test_deterministic_and_seed_dependent(self, eval_set):
        a = stratified_splits(eval_set.items, n_splits=3, seed=1)
        assert a == stratified_splits(eval_set.items, n_splits=3, seed=1)
        assert a != stratified_splits(eval_set.items, n_splits=3, seed=2)
        assert a[0] != a[1]

    @pytest.mark.parametrize("kwargs", [{"fractions": (0.5, 0.5)}, {"fractions": (0.5, 0.4, 0.3)},
                                        {"n_splits": 0}])
    def test_invalid(self, eval_set, kwargs):
        with pytest.raises(ValueError):
            stratified_splits(eval_set.items, **kwargs)


# ---------------------------------------------------------------------------
# Benchmark end to end (committed eval set, offline pipeline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_stage():
    first = HybridRetriever(HashingEmbedder()).index(load_chunks_jsonl(CORPUS))
    return RerankingRetriever(first, Reranker(mock=True), retrieve_k=50)


@pytest.fixture(scope="module")
def report(eval_set, two_stage):
    return run_abstention_benchmark(eval_set, two_stage, n_splits=10)


class TestBenchmark:
    def test_structure_and_label_stats(self, report):
        ls = report["label_stats"]
        assert ls["n"] == 100 and ls["safe"] + ls["unsafe"] == 100
        assert ls["unsafe"] == ls["unsafe_unanswerable"] + ls["unsafe_retrieval_miss"]
        assert ls["unsafe_unanswerable"] == 30
        assert report["config"]["reranker_backend"] == "mock-lexical-v1"
        assert report["config"]["ltt_min_answered"] == {"0.05": 45, "0.1": 22, "0.2": 11}
        assert {"combined", "platt:ce_top1", "platt:rrf_top1"} <= set(report["models"])

    def test_metric_ranges(self, report):
        for model in report["models"].values():
            m = model["metrics"]
            assert 0 <= m["auroc"]["mean"] <= 1 and 0 <= m["ece"]["mean"] <= 1
            assert m["eaurc"]["mean"] >= -1e-9
            assert m["sel_acc@1"]["mean"] == pytest.approx(0.66, abs=0.05)  # test-fold base rate
            for by_method in model["thresholds"].values():
                for t in by_method.values():
                    assert 0 <= t["test_coverage"]["mean"] <= 1

    def test_ltt_is_more_conservative_than_erm(self, report):
        for by_method in report["models"]["combined"]["thresholds"].values():
            assert by_method["ltt"]["test_coverage"]["mean"] <= by_method["erm"]["test_coverage"]["mean"] + 1e-9

    def test_infeasible_alpha_abstains_with_reason(self, report):
        ltt = report["models"]["combined"]["thresholds"]["0.05"]["ltt"]
        assert ltt["abstain_all_rate"] == 1.0 and ltt["test_coverage"]["mean"] == 0.0
        assert any("too small" in r for r in ltt["abstain_reasons"])

    def test_combined_beats_chance(self, report):
        assert report["models"]["combined"]["metrics"]["auroc"]["mean"] > 0.6

    def test_markdown(self, report):
        md = render_markdown(report)
        assert "# Abstention Benchmark" in md and "**mock**" in md and "**hashing**" in md
        assert "| 0.2 | LTT |" in md and "n_min=22" in md

    def test_retrieval_only_pipeline(self, eval_set, two_stage):
        rep = run_abstention_benchmark(eval_set, two_stage.first_stage, n_splits=3)
        assert not set(CE_FEATURES) & set(rep["config"]["features"])
        assert rep["config"]["reranker_backend"] is None
        assert "platt:ce_top1" not in rep["models"]

    def test_deterministic(self, eval_set, two_stage, report):
        again = run_abstention_benchmark(eval_set, two_stage, n_splits=10)
        assert json.dumps(again, sort_keys=True, default=str) == json.dumps(report, sort_keys=True, default=str)

    def test_fit_policy(self, eval_set, two_stage):
        policy = fit_policy(eval_set, two_stage, alpha=0.2, method="erm")
        assert 0 <= policy.tau <= 1 and policy.meta["method"] == "erm"
        ltt = fit_policy(eval_set, two_stage, alpha=0.05)
        assert ltt.tau == math.inf  # cannot be certified with 30 calibration items

    def test_figures(self, report, tmp_path):
        pytest.importorskip("matplotlib")
        paths = write_figures(report, tmp_path)
        assert {p.name for p in paths} == {"risk_coverage.png", "coverage_accuracy.png", "reliability.png"}
        assert all(p.stat().st_size > 1000 for p in paths)


class TestCli:
    def test_writes_reports_and_policy(self, tmp_path, capsys):
        out = tmp_path / "ab.json"
        policy_path = tmp_path / "policy.json"
        assert abstention_main(["run", "--embedder", "hashing", "--reranker", "mock", "--n-splits", "4",
                                "--output", str(out), "--no-figures", "--save-policy", str(policy_path),
                                "--policy-method", "erm", "--policy-alpha", "0.2"]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))  # strict JSON (no NaN / Infinity literals)
        assert report["config"]["n_splits"] == 4
        assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("# Abstention Benchmark")
        policy = AbstentionPolicy.load(policy_path)
        assert policy.meta["method"] == "erm" and np.isfinite(policy.tau)
        assert "saved policy" in capsys.readouterr().out

    def test_no_reranker_drops_ce_features(self, tmp_path):
        out = tmp_path / "ab.json"
        abstention_main(["run", "--embedder", "hashing", "--reranker", "none", "--n-splits", "2",
                         "--output", str(out), "--no-figures"])
        assert not set(CE_FEATURES) & set(json.loads(out.read_text(encoding="utf-8"))["config"]["features"])

    def test_default_reranker_falls_back(self, tmp_path, no_model_packages):
        out = tmp_path / "ab.json"
        abstention_main(["run", "--embedder", "hashing", "--n-splits", "2", "--output", str(out), "--no-figures"])
        cfg = json.loads(out.read_text(encoding="utf-8"))["config"]
        assert cfg["reranker_backend"] == "mock-lexical-v1" and "ImportError" in cfg["reranker_fallback_reason"]
