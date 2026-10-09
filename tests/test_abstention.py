"""Tests for the abstention layer: features, calibrator, thresholds (ERM / LTT) and runtime policy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.abstention import (
    CE_FEATURES,
    FEATURE_NAMES,
    RETRIEVAL_FEATURES,
    AbstentionPolicy,
    LogisticCalibrator,
    SelectiveRetriever,
    binom_cdf,
    coverage_grid,
    erm_threshold,
    extract_features,
    ltt_threshold,
    min_certifiable_n,
    platt,
    risk_profile,
    safe_to_answer,
)
from src.constants import ABSTENTION_ANSWER
from src.reranker import Reranker, RerankingRetriever
from src.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Binomial tail
# ---------------------------------------------------------------------------


class TestBinomCdf:
    def test_hand_computed(self):
        assert binom_cdf(0, 3, 0.5) == pytest.approx(0.125)
        assert binom_cdf(1, 3, 0.5) == pytest.approx(0.5)
        assert binom_cdf(0, 22, 0.1) == pytest.approx(0.9 ** 22)

    def test_edges(self):
        assert binom_cdf(0, 0, 0.3) == 1.0
        assert binom_cdf(5, 5, 0.3) == 1.0
        assert binom_cdf(-1, 5, 0.3) == 0.0

    def test_min_certifiable_n(self):
        assert min_certifiable_n(0.1, 0.1) == 22  # 0.9^21 = 0.109 > 0.1 >= 0.9^22
        assert min_certifiable_n(0.2, 0.1) == 11
        assert binom_cdf(0, 22, 0.1) <= 0.1 < binom_cdf(0, 21, 0.1)
        with pytest.raises(ValueError):
            min_certifiable_n(0.0, 0.1)


# ---------------------------------------------------------------------------
# Risk profile and ERM
# ---------------------------------------------------------------------------

SCORES = [0.9, 0.8, 0.8, 0.7, 0.6, 0.5]
LABELS = [1, 1, 0, 1, 0, 0]


class TestErm:
    def test_risk_profile_groups_ties(self):
        prof = risk_profile(SCORES, LABELS)
        assert prof.thresholds.tolist() == [0.9, 0.8, 0.7, 0.6, 0.5]
        assert prof.n_answered.tolist() == [1, 3, 4, 5, 6]
        assert prof.n_errors.tolist() == [0, 1, 1, 2, 3]
        assert prof.risk[1] == pytest.approx(1 / 3)

    def test_erm_picks_largest_feasible_coverage(self):
        tr = erm_threshold(SCORES, LABELS, alpha=0.34)
        assert (tr.tau, tr.n_answered, tr.cal_coverage, tr.cal_risk) == (0.7, 4, pytest.approx(4 / 6), 0.25)
        assert erm_threshold(SCORES, LABELS, alpha=0.5).tau == 0.5  # risk 0.5 at full coverage

    def test_erm_infeasible_abstains(self):
        tr = erm_threshold([0.9, 0.8], [0, 0], alpha=0.1)
        assert tr.abstains_on_all and tr.cal_coverage == 0 and math.isnan(tr.cal_risk)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            erm_threshold(SCORES, LABELS, alpha=1.5)
        with pytest.raises(ValueError):
            risk_profile([], [])


# ---------------------------------------------------------------------------
# Learn-then-Test
# ---------------------------------------------------------------------------


class TestLtt:
    def test_certifies_with_enough_clean_data(self):
        tr = ltt_threshold([0.9] * 25, [1] * 25, alpha=0.1, delta=0.1)
        assert tr.tau == 0.9 and tr.n_answered == 25 and tr.cal_risk == 0.0

    def test_too_little_data_abstains_with_reason(self):
        tr = ltt_threshold([0.9] * 10, [1] * 10, alpha=0.1, delta=0.1)
        assert tr.abstains_on_all and "too small" in tr.reason

    def test_never_more_permissive_than_erm(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = rng.uniform(size=60)
            y = (rng.uniform(size=60) < s).astype(int)
            erm, ltt = erm_threshold(s, y, 0.2), ltt_threshold(s, y, 0.2, 0.1)
            assert ltt.tau >= erm.tau

    def test_fixed_sequence_stops_at_first_failure(self):
        # grid: 0.9 (n=30 clean) rejects, 0.5 (adds 20 errors) fails, 0.1 would pass again but is never reached
        s = [0.95] * 30 + [0.6] * 20 + [0.2] * 200
        y = [1] * 30 + [0] * 20 + [1] * 200
        tr = ltt_threshold(s, y, alpha=0.1, delta=0.1, grid=[0.9, 0.5, 0.1])
        assert tr.tau == 0.9

    def test_grid_must_be_descending(self):
        with pytest.raises(ValueError, match="strict"):
            ltt_threshold([0.5, 0.6], [1, 1], 0.1, 0.1, grid=[0.1, 0.9])

    def test_coverage_grid(self):
        ref = np.linspace(0, 1, 101)
        grid = coverage_grid(ref, n_cal=100, alpha=0.1, delta=0.1)
        assert np.all(np.diff(grid) < 0)
        assert grid[0] == pytest.approx(np.quantile(ref, 1 - 22 / 100))  # starts at n_min / n_cal coverage
        assert coverage_grid(ref, n_cal=10, alpha=0.1, delta=0.1).size == 0

    def test_guarantee_holds_in_simulation(self):
        """Over many calibration draws, P(true selective risk > α) must stay ≤ δ (+ MC slack)."""
        # n_cal is large enough that the first grid point (exactly n_min answered, so zero
        # errors allowed) is usually passed; with small n_cal LTT is valid but often vacuous.
        alpha, delta, n_cal, trials = 0.1, 0.1, 1000, 300
        rng = np.random.default_rng(42)

        def true_risk(tau):  # score ~ U(0,1), P(error | score) = 0.3 * (1 - score)
            if tau >= 1:
                return 0.0
            return 0.3 * (1 - (1 + tau) / 2)  # E[0.3(1-s) | s >= tau]

        grid = coverage_grid(rng.uniform(size=5000), n_cal, alpha, delta)  # independent reference draw
        violations, answered = 0, 0
        for _ in range(trials):
            s = rng.uniform(size=n_cal)
            y = (rng.uniform(size=n_cal) >= 0.3 * (1 - s)).astype(int)
            tr = ltt_threshold(s, y, alpha, delta, grid=grid)
            if not tr.abstains_on_all:
                answered += 1
                violations += true_risk(tr.tau) > alpha
        assert answered > trials * 0.5  # the procedure is not vacuous here
        assert violations / trials <= delta + 3 * math.sqrt(delta * (1 - delta) / trials)


# ---------------------------------------------------------------------------
# Logistic calibrator
# ---------------------------------------------------------------------------


def _toy_rows(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return [{"a": float(a), "b": float(b), "const": 1.0} for a, b in X], y


class TestLogisticCalibrator:
    def test_recovers_weight_directions(self):
        rows, y = _toy_rows()
        cal = LogisticCalibrator(["a", "b"]).fit(rows, y)
        assert cal.coef[0] > 0 > cal.coef[1]
        assert np.all((cal.predict_proba(rows) > 0) & (cal.predict_proba(rows) < 1))

    def test_deterministic_and_round_trip(self):
        rows, y = _toy_rows()
        a = LogisticCalibrator(["a", "b"]).fit(rows, y)
        b = LogisticCalibrator(["a", "b"]).fit(rows, y)
        assert np.array_equal(a.coef, b.coef)
        restored = LogisticCalibrator.from_dict(a.to_dict())
        assert np.allclose(restored.predict_proba(rows), a.predict_proba(rows))

    def test_constant_feature_is_harmless(self):
        rows, y = _toy_rows()
        cal = LogisticCalibrator(["a", "b", "const"]).fit(rows, y)
        assert cal.coef[2] == pytest.approx(0.0, abs=1e-9)

    def test_separable_data_stays_finite(self):
        rows = [{"a": float(v)} for v in range(-5, 6)]
        y = [int(v > 0) for v in range(-5, 6)]
        probs = LogisticCalibrator(["a"], l2=1.0).fit(rows, y).predict_proba(rows)
        assert np.all(np.isfinite(probs)) and probs[-1] > probs[0]

    def test_errors(self):
        rows, _ = _toy_rows(10)
        with pytest.raises(ValueError, match="single class"):
            LogisticCalibrator(["a"]).fit(rows, [1] * 10)
        with pytest.raises(KeyError):
            LogisticCalibrator(["missing"]).fit(rows, [0, 1] * 5)
        with pytest.raises(RuntimeError):
            LogisticCalibrator(["a"]).predict_proba(rows)
        with pytest.raises(ValueError):
            LogisticCalibrator([])

    def test_platt_preserves_ranking(self):
        rows, y = _toy_rows()
        probs = platt("a").fit(rows, y).predict_proba(rows)
        raw = np.array([r["a"] for r in rows])
        assert np.array_equal(np.argsort(probs, kind="stable"), np.argsort(raw, kind="stable"))


# ---------------------------------------------------------------------------
# Feature extraction and runtime policy
# ---------------------------------------------------------------------------


class CountingRetriever(HybridRetriever):
    calls = 0

    def retrieve(self, *args, **kwargs):
        CountingRetriever.calls += 1
        return super().retrieve(*args, **kwargs)


@pytest.fixture
def first_stage(corpus_chunks, hashing_embedder):
    return HybridRetriever(hashing_embedder).index(corpus_chunks)


@pytest.fixture
def two_stage(first_stage):
    return RerankingRetriever(first_stage, Reranker(mock=True), retrieve_k=6)


class TestFeatures:
    def test_full_feature_set_with_reranker(self, two_stage):
        ev = extract_features("How does conformal prediction guarantee coverage?", two_stage, k_ctx=3, top_k=5)
        assert list(ev.features) == list(FEATURE_NAMES)
        assert all(np.isfinite(v) for v in ev.features.values())
        assert ev.features["ce_margin"] >= 0 and ev.features["rrf_margin"] >= 0
        assert 0 <= ev.features["ce_entropy"] <= 1 and 0 <= ev.features["overlap@5"] <= 1
        assert len(ev.context) == 3 and ev.context[0].chunk.chunk_index == 2

    def test_retrieval_only_features_without_reranker(self, first_stage):
        ev = extract_features("conformal coverage", first_stage)
        assert list(ev.features) == list(RETRIEVAL_FEATURES)
        assert not set(CE_FEATURES) & set(ev.features)

    def test_single_first_stage_call(self, corpus_chunks, hashing_embedder):
        counting = CountingRetriever(hashing_embedder).index(corpus_chunks)
        CountingRetriever.calls = 0
        extract_features("conformal coverage", RerankingRetriever(counting, Reranker(mock=True)))
        assert CountingRetriever.calls == 1

    def test_empty_query_gives_zero_features(self, two_stage):
        ev = extract_features("", two_stage)
        assert ev.results == [] and set(ev.features.values()) == {0.0}

    def test_invalid_k(self, two_stage):
        with pytest.raises(ValueError):
            extract_features("q", two_stage, k_ctx=0)
        with pytest.raises(ValueError):
            extract_features("q", two_stage, k_ctx=5, top_k=3)

    def test_label(self, two_stage, corpus_chunks):
        ev = extract_features("How does conformal prediction guarantee coverage?", two_stage, k_ctx=1)
        assert safe_to_answer(ev, [corpus_chunks[2].chunk_id]) == 1
        assert safe_to_answer(ev, [corpus_chunks[0].chunk_id]) == 0  # gold not in the top-1 context
        assert safe_to_answer(ev, []) == 0  # unanswerable is never safe


def _policy(tau: float) -> AbstentionPolicy:
    rows, y = _toy_rows()
    rows = [{**r, **{f: r["a"] for f in FEATURE_NAMES}} for r in rows]
    return AbstentionPolicy(LogisticCalibrator(list(FEATURE_NAMES)).fit(rows, y), tau, {"note": "test"})


class TestPolicy:
    def test_extreme_thresholds(self, two_stage):
        query = "How does conformal prediction guarantee coverage?"
        always = SelectiveRetriever(two_stage, _policy(-math.inf)).query(query)
        never = SelectiveRetriever(two_stage, _policy(math.inf)).query(query)
        assert always.answered and always.reference is None and len(always.evidence) == 3
        assert not never.answered and never.reference == ABSTENTION_ANSWER and never.evidence == []
        assert always.confidence == never.confidence

    def test_empty_query_abstains(self, two_stage):
        result = SelectiveRetriever(two_stage, _policy(-math.inf)).query("")
        assert not result.answered and result.reference == ABSTENTION_ANSWER

    def test_save_load_same_decisions(self, tmp_path, two_stage):
        policy = _policy(0.5)
        loaded = AbstentionPolicy.load(policy.save(tmp_path / "p.json"))
        assert loaded.tau == 0.5 and loaded.meta == {"note": "test"}
        feats = extract_features("ranked lists fusion", two_stage).features
        assert loaded.decide(feats) == policy.decide(feats)

    def test_infinite_tau_round_trips(self, tmp_path):
        assert AbstentionPolicy.load(_policy(math.inf).save(tmp_path / "p.json")).tau == math.inf
