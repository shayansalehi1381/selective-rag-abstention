"""Calibrated abstention: answer only when the retrieved evidence is likely sufficient.

Formulation. For a query ``x`` the pipeline returns a ranked list ``R(x)``. The top
``k_ctx`` chunks are the context a generator would see. Until generation exists, the
label is retrieval-grounded:

    y(x) = 1[x is answerable  and  a gold chunk is in R_kctx(x)]      ("safe to answer")

A confidence model ``g(x) ∈ [0, 1]`` is fitted on features of the retrieval outcome,
and the policy answers iff ``g(x) ≥ τ``. With coverage ``φ(τ) = P(g ≥ τ)`` and
selective risk ``R(τ) = P(y = 0 | g ≥ τ)``, the goal is to maximise ``φ(τ)``
subject to ``R(τ) ≤ α``.

There are two ways to pick ``τ`` on a held-out calibration set:

* ``erm_threshold``: the largest coverage whose *empirical* risk is ``≤ α``. It comes
  with no guarantee.
* ``ltt_threshold``: Learn-then-Test (Angelopoulos, Bates, Candès, Jordan & Lei, 2021).
  Thresholds are tested from strict to permissive with exact binomial p-values and
  fixed-sequence testing, which gives ``P(R(τ̂) ≤ α) ≥ 1 − δ`` over the calibration
  draw.

All numerics are numpy and the standard library (no scipy or sklearn). Policies
persist as JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.constants import ABSTENTION_ANSWER
from src.reranker import RerankingRetriever
from src.retriever import HybridRetriever, RetrievalResult

CE_FEATURES = ("ce_top1", "ce_margin", "ce_entropy")
RETRIEVAL_FEATURES = ("rrf_top1", "rrf_margin", "bm25_top1", "dense_top1", "rank_agreement", "overlap@5")
FEATURE_NAMES = CE_FEATURES + RETRIEVAL_FEATURES


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryEvidence:
    query: str
    features: dict[str, float]
    results: list[RetrievalResult]  # final ranked list (reranked if a reranker is used)
    k_ctx: int

    @property
    def context(self) -> list[RetrievalResult]:
        return self.results[: self.k_ctx]


def _entropy_normalised(logits: Sequence[float]) -> float:
    if len(logits) < 2:
        return 0.0
    z = np.asarray(logits, dtype=np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    h = -float(np.sum(p * np.log(np.clip(p, 1e-300, None))))
    return h / math.log(len(logits))


def extract_features(query: str, retriever: RerankingRetriever | HybridRetriever, *, k_ctx: int = 3,
                     top_k: int = 10) -> QueryEvidence:
    """Run the pipeline once and turn its outcome into abstention features.

    A ``RerankingRetriever`` gives all of ``FEATURE_NAMES``. A plain ``HybridRetriever``
    gives only ``RETRIEVAL_FEATURES``. A query with no candidates gets all-zero features.
    """
    if k_ctx <= 0 or top_k < k_ctx:
        raise ValueError("need 0 < k_ctx <= top_k")
    reranking = isinstance(retriever, RerankingRetriever)
    if reranking:
        candidates, results = retriever.retrieve_with_candidates(query, top_k)
    else:
        candidates = retriever.retrieve(query, top_k=max(top_k, retriever.candidate_pool))
        results = candidates[:top_k]
    names = FEATURE_NAMES if reranking else RETRIEVAL_FEATURES
    feats = {name: 0.0 for name in names}
    if not results:
        return QueryEvidence(query, feats, [], k_ctx)

    if reranking:
        logits = [r.rerank_score for r in results]
        feats["ce_top1"] = float(logits[0])
        feats["ce_margin"] = float(logits[0] - logits[1]) if len(logits) > 1 else 0.0
        feats["ce_entropy"] = _entropy_normalised(logits)

    fused = sorted(candidates, key=lambda r: r.rank)  # first-stage order
    feats["rrf_top1"] = float(fused[0].score)
    feats["rrf_margin"] = float(fused[0].score - fused[1].score) if len(fused) > 1 else 0.0
    bm25_top = [r.bm25_score for r in candidates if r.bm25_rank == 1]
    dense_top = [r.dense_score for r in candidates if r.dense_rank == 1]
    feats["bm25_top1"] = math.log1p(max(0.0, bm25_top[0])) if bm25_top else 0.0
    feats["dense_top1"] = float(dense_top[0]) if dense_top else 0.0
    top = results[0]
    if top.bm25_rank is not None and top.dense_rank is not None:
        feats["rank_agreement"] = 1.0 / (1.0 + abs(top.bm25_rank - top.dense_rank))
    sparse5 = {r.chunk.chunk_id for r in candidates if r.bm25_rank is not None and r.bm25_rank <= 5}
    dense5 = {r.chunk.chunk_id for r in candidates if r.dense_rank is not None and r.dense_rank <= 5}
    union = sparse5 | dense5
    feats["overlap@5"] = len(sparse5 & dense5) / len(union) if union else 0.0
    return QueryEvidence(query, feats, results, k_ctx)


def safe_to_answer(evidence: QueryEvidence, gold_chunk_ids: Sequence[str]) -> int:
    """Retrieval-grounded label: answerable and a gold chunk is in the generator context."""
    gold = set(gold_chunk_ids)
    return int(bool(gold) and any(r.chunk.chunk_id in gold for r in evidence.context))


# ---------------------------------------------------------------------------
# Confidence model: L2-regularised logistic regression (IRLS)
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable logistic


class LogisticCalibrator:
    """Logistic regression on standardised features, fitted by Newton/IRLS.

    The penalty ``l2 · ||w||²/2`` applies to the weights, not the intercept. A single
    feature gives Platt scaling. Standardisation statistics come from the fit data only.
    """

    def __init__(self, features: Sequence[str], *, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-8) -> None:
        if not features:
            raise ValueError("at least one feature is required")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.features = list(features)
        self.l2, self.max_iter, self.tol = l2, max_iter, tol
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: float = 0.0
        self.n_iter = 0

    def _matrix(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        missing = [f for f in self.features if rows and f not in rows[0]]
        if missing:
            raise KeyError(f"feature rows are missing {missing}")
        return np.array([[float(r[f]) for f in self.features] for r in rows], dtype=np.float64).reshape(
            len(rows), len(self.features))

    def fit(self, rows: Sequence[Mapping[str, float]], labels: Sequence[int]) -> LogisticCalibrator:
        X, y = self._matrix(rows), np.asarray(labels, dtype=np.float64)
        if X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("rows and labels must be non-empty and aligned")
        if np.all(y == y[0]):
            raise ValueError("cannot fit a calibrator on a single class")
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0  # constant column: leave it centred, unscaled
        Z = np.hstack([np.ones((X.shape[0], 1)), (X - self.mean) / self.std])
        penalty = np.full(Z.shape[1], self.l2)
        penalty[0] = 1e-9  # intercept: (almost) unpenalised, kept PD for separable data
        w = np.zeros(Z.shape[1])
        for self.n_iter in range(1, self.max_iter + 1):
            p = _sigmoid(Z @ w)
            grad = Z.T @ (p - y) + penalty * w
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.intercept, self.coef = float(w[0]), w[1:]
        return self

    def decision_function(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("calibrator is not fitted")
        return self.intercept + ((self._matrix(rows) - self.mean) / self.std) @ self.coef

    def predict_proba(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        return _sigmoid(self.decision_function(rows))

    def to_dict(self) -> dict[str, Any]:
        if self.coef is None:
            raise RuntimeError("calibrator is not fitted")
        return {"features": self.features, "l2": self.l2, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogisticCalibrator:
        obj = cls(data["features"], l2=data["l2"])
        obj.mean, obj.std = np.asarray(data["mean"], float), np.asarray(data["std"], float)
        obj.coef, obj.intercept = np.asarray(data["coef"], float), float(data["intercept"])
        return obj


def platt(feature: str) -> LogisticCalibrator:
    """Platt scaling of a single signal: ``σ(a·s + b)``."""
    return LogisticCalibrator([feature], l2=1e-6)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact ``P(Bin(n, p) ≤ k)``."""
    if n <= 0 or k >= n:
        return 1.0
    if k < 0:
        return 0.0
    return float(min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))))


@dataclass(frozen=True)
class RiskProfile:
    thresholds: np.ndarray  # distinct scores, descending (increasing coverage)
    n_answered: np.ndarray
    n_errors: np.ndarray
    n_total: int

    @property
    def coverage(self) -> np.ndarray:
        return self.n_answered / self.n_total

    @property
    def risk(self) -> np.ndarray:
        return self.n_errors / self.n_answered


def risk_profile(scores: Sequence[float], labels: Sequence[int]) -> RiskProfile:
    """Answered count and error count when answering iff ``score ≥ λ``, for every distinct λ."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty and aligned")
    thresholds = np.unique(s)[::-1]
    order = np.argsort(-s, kind="stable")
    errors_sorted = np.cumsum(1 - y[order])
    # number of items with score >= λ for each λ, via the sorted scores
    n_answered = np.searchsorted(-s[order], -thresholds, side="right")
    return RiskProfile(thresholds, n_answered, errors_sorted[n_answered - 1], int(s.size))


@dataclass(frozen=True)
class ThresholdResult:
    method: str
    alpha: float
    delta: float | None
    tau: float  # +inf means "abstain on everything"
    cal_coverage: float
    cal_risk: float  # nan when nothing is answered
    n_answered: int
    reason: str | None = None  # why nothing could be certified, if so

    @property
    def abstains_on_all(self) -> bool:
        return math.isinf(self.tau) and self.tau > 0

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "alpha": self.alpha, "delta": self.delta, "tau": _encode_float(self.tau),
                "cal_coverage": self.cal_coverage, "cal_risk": _encode_float(self.cal_risk),
                "n_answered": self.n_answered, "reason": self.reason}


def _result(method: str, alpha: float, delta: float | None, prof: RiskProfile, j: int | None,
            reason: str | None = None) -> ThresholdResult:
    if j is None:
        return ThresholdResult(method, alpha, delta, math.inf, 0.0, math.nan, 0, reason)
    return ThresholdResult(method, alpha, delta, float(prof.thresholds[j]), float(prof.coverage[j]),
                           float(prof.risk[j]), int(prof.n_answered[j]))


def min_certifiable_n(alpha: float, delta: float) -> int:
    """Fewest answered calibration items that can certify ``R ≤ α`` at level δ (zero errors).

    With 0 errors the p-value is ``(1 − α)^n``, so ``n ≥ log δ / log(1 − α)``. For
    example, α = δ = 0.1 needs 22 answered items, and α = 0.2, δ = 0.1 needs 11.
    """
    _check_level(alpha, "alpha")
    _check_level(delta, "delta")
    return math.ceil(math.log(delta) / math.log(1 - alpha) - 1e-12)


def coverage_grid(reference_scores: Sequence[float], n_cal: int, alpha: float, delta: float, *,
                  n_points: int = 50) -> np.ndarray:
    """Data-independent LTT grid: thresholds at target coverages, ordered strict to permissive.

    The thresholds are quantiles of ``reference_scores``. These must come from data
    independent of the calibration labels (e.g. the train split), so that the hypothesis
    family is fixed before testing. The grid starts at coverage ``n_min / n_cal``,
    because no stricter threshold could ever be certified and fixed-sequence testing
    would stop there at once. It is empty if the calibration set is too small for any
    certificate. Trade-off: at that first point exactly ``n_min`` items are answered, so a
    single calibration error stops the walk. Small calibration sets therefore often give
    a valid but vacuous (abstain-all) result.
    """
    q_min = min_certifiable_n(alpha, delta) / n_cal
    if q_min > 1:
        return np.zeros(0)
    q = np.linspace(q_min, 1.0, n_points)
    grid = np.quantile(np.asarray(reference_scores, float), 1 - q)
    return np.unique(grid)[::-1]


def erm_threshold(scores: Sequence[float], labels: Sequence[int], alpha: float) -> ThresholdResult:
    """Largest calibration coverage with empirical selective risk ``≤ α`` (no guarantee)."""
    _check_level(alpha, "alpha")
    prof = risk_profile(scores, labels)
    ok = np.flatnonzero(prof.risk <= alpha + 1e-12)
    return _result("erm", alpha, None, prof, int(ok[-1]) if ok.size else None)


def ltt_threshold(scores: Sequence[float], labels: Sequence[int], alpha: float, delta: float, *,
                  grid: Sequence[float] | None = None) -> ThresholdResult:
    """Learn-then-Test with exact binomial p-values and fixed-sequence testing.

    For each threshold ``λ_j`` in ``grid`` (strict to permissive), ``H_j: R(λ_j) > α`` is
    tested with ``p_j = P(Bin(n_j, α) ≤ e_j)``, where ``n_j`` and ``e_j`` are the answered
    and erroneous calibration items. Testing stops at the first non-rejection, and the
    result is the last rejected λ. This controls the family-wise error rate at δ, so
    ``P(R(τ̂) ≤ α) ≥ 1 − δ``.

    ``grid`` must be fixed before looking at the calibration labels (see
    ``coverage_grid``). Leading grid points that answer fewer than ``min_certifiable_n``
    items are skipped, because they are untestable. If it is omitted, the distinct calibration scores are used and
    the walk starts at the first threshold with at least ``min_certifiable_n`` answered
    items. That convenience default is data-dependent, so the guarantee is approximate.
    """
    _check_level(alpha, "alpha")
    _check_level(delta, "delta")
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty and aligned")
    n_min = min_certifiable_n(alpha, delta)
    if grid is None:
        grid = [t for t in np.unique(s)[::-1] if np.sum(s >= t) >= n_min]
    grid = np.asarray(grid, float)
    if grid.size and np.any(np.diff(grid) > 0):
        raise ValueError("grid must be ordered from strict (high) to permissive (low)")
    if grid.size == 0:
        return ThresholdResult("ltt", alpha, delta, math.inf, 0.0, math.nan, 0,
                               f"calibration set too small: need >= {n_min} answered items, have {s.size}")
    best = None
    started = False
    for lam in grid:
        answered = s >= lam
        n_j, e_j = int(answered.sum()), int((1 - y[answered]).sum())
        if not started and n_j < n_min:
            # Leading points that answer fewer than n_min items can never be rejected,
            # whatever the labels. Skipping them depends on counts only, never on labels,
            # so the sequence still controls FWER under monotone conditional risk.
            continue
        started = True
        if binom_cdf(e_j, n_j, alpha) <= delta:
            best = (float(lam), n_j, e_j)
        else:
            break
    if best is None:
        reason = ("first hypothesis not rejected: too many errors at the strictest testable threshold" if started
                  else f"no grid threshold answers >= {n_min} calibration items")
        return ThresholdResult("ltt", alpha, delta, math.inf, 0.0, math.nan, 0, reason)
    lam, n_j, e_j = best
    return ThresholdResult("ltt", alpha, delta, lam, n_j / s.size, e_j / n_j, n_j)


def _check_level(x: float, name: str) -> None:
    if not 0.0 < x < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")


def _encode_float(x: float) -> float | str | None:
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _decode_float(x: float | str | None) -> float:
    if x is None:
        return math.nan
    return float(x)


# ---------------------------------------------------------------------------
# Runtime policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    answer: bool
    confidence: float
    tau: float


@dataclass
class AbstentionPolicy:
    """``answer iff g(x) ≥ τ``, with ``g`` a fitted ``LogisticCalibrator``."""

    calibrator: LogisticCalibrator
    tau: float
    meta: dict[str, Any] = field(default_factory=dict)

    def confidence(self, features: Mapping[str, float]) -> float:
        return float(self.calibrator.predict_proba([features])[0])

    def decide(self, features: Mapping[str, float]) -> Decision:
        g = self.confidence(features)
        return Decision(answer=g >= self.tau, confidence=g, tau=self.tau)

    def to_dict(self) -> dict[str, Any]:
        return {"calibrator": self.calibrator.to_dict(), "tau": _encode_float(self.tau), "meta": self.meta}

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> AbstentionPolicy:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(LogisticCalibrator.from_dict(data["calibrator"]), _decode_float(data["tau"]), data.get("meta", {}))


@dataclass(frozen=True)
class SelectiveResult:
    query: str
    answered: bool
    confidence: float
    evidence: list[RetrievalResult]  # the generator context; empty when abstaining
    reference: str | None  # ABSTENTION_ANSWER when abstaining, None otherwise (to be generated)
    features: dict[str, float]


class SelectiveRetriever:
    """Retrieve, score confidence, then return evidence or the canonical abstention."""

    def __init__(self, retriever: RerankingRetriever | HybridRetriever, policy: AbstentionPolicy, *,
                 k_ctx: int = 3, top_k: int = 10) -> None:
        self.retriever, self.policy, self.k_ctx, self.top_k = retriever, policy, k_ctx, top_k

    def query(self, query: str) -> SelectiveResult:
        ev = extract_features(query, self.retriever, k_ctx=self.k_ctx, top_k=self.top_k)
        decision = self.policy.decide(ev.features)
        if decision.answer and ev.results:
            return SelectiveResult(query, True, decision.confidence, ev.context, None, ev.features)
        return SelectiveResult(query, False, decision.confidence, [], ABSTENTION_ANSWER, ev.features)
