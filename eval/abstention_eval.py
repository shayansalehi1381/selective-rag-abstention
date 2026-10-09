"""Selective-prediction benchmark for the abstention layer.

Protocol: features and labels are computed once per eval item. Then, for each of
``n_splits`` stratified 40/30/30 train/calibration/test splits:

1. fit each confidence model on **train** (the combined logistic model, plus Platt-scaled
   single signals);
2. choose ``τ`` on **calibration** with ERM (no guarantee) and Learn-then-Test
   (``P(R ≤ α) ≥ 1 − δ``) for every target ``α``;
3. score on **test**: AUROC, AURC/E-AURC, selective accuracy at fixed coverage, Brier,
   ECE, and the coverage and selective risk realised by each chosen ``τ``.

Results are aggregated as the mean and 5th/95th percentiles over splits. Curves and
reliability data are also pooled across the test folds, because each fold is small.

CLI::

    python -m eval.abstention_eval run --eval-set data/eval/eval_set.json
    python -m eval.abstention_eval run --embedder hashing --reranker mock --save-policy data/eval/policy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from eval.evaluate import add_pipeline_args, auroc, build_pipeline
from eval.schemas import EvalItem, EvalSet, corpus_fingerprint
from src.abstention import (
    AbstentionPolicy,
    coverage_grid,
    min_certifiable_n,
    LogisticCalibrator,
    erm_threshold,
    extract_features,
    ltt_threshold,
    platt,
    risk_profile,
    safe_to_answer,
)
from src.data_loader import load_chunks_jsonl
from src.reranker import RerankingRetriever

logger = logging.getLogger(__name__)

DEFAULT_REPORT = Path("data/eval/abstention_benchmark.json")
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)
COVERAGE_LEVELS = (0.5, 0.8, 0.9, 1.0)
SINGLE_SIGNALS = ("ce_top1", "ce_margin", "rrf_top1", "rrf_margin", "dense_top1")


# ---------------------------------------------------------------------------
# Selective-prediction metrics (pure functions; y = 1 means "safe to answer")
# ---------------------------------------------------------------------------


def risk_coverage_curve(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, selective risk) at every distinct threshold, from strict to permissive."""
    prof = risk_profile(scores, labels)
    return prof.coverage, prof.risk


def aurc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the risk-coverage curve, as a step function over coverage.

    Tied scores enter together, so ties are never resolved in the model's favour.
    """
    prof = risk_profile(scores, labels)
    widths = np.diff(np.concatenate([[0], prof.n_answered])) / prof.n_total
    return float(np.sum(widths * prof.risk))


def oracle_aurc(labels: Sequence[int]) -> float:
    """AURC of the perfect ranking: every safe item before every unsafe one."""
    y = np.asarray(labels, dtype=np.int64)
    n, n_pos = y.size, int(y.sum())
    k = np.arange(1, n + 1)
    return float(np.mean(np.maximum(0, k - n_pos) / k))


def eaurc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Excess AURC over the oracle (0 = perfect ranking)."""
    return aurc(scores, labels) - oracle_aurc(labels)


def selective_accuracy_at(scores: Sequence[float], labels: Sequence[int], coverage: float) -> float:
    """Accuracy on the ``⌈coverage·n⌉`` most confident items (ties keep input order)."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    m = max(1, math.ceil(coverage * s.size - 1e-9))
    return float(y[np.argsort(-s, kind="stable")[:m]].mean())


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2))


def reliability_bins(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> list[dict]:
    """Equal-width bins on [0, 1]; the last bin includes 1.0."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "count": int(mask.sum()),
                     "mean_confidence": float(p[mask].mean()) if mask.any() else None,
                     "fraction_safe": float(y[mask].mean()) if mask.any() else None})
    return bins


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected calibration error: ``Σ_b |B_b|/n · |acc_b − conf_b|``."""
    n = len(probs)
    return float(sum(b["count"] / n * abs(b["fraction_safe"] - b["mean_confidence"])
                     for b in reliability_bins(probs, labels, n_bins) if b["count"]))


def _auroc_labels(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    return auroc(s[y == 1], s[y == 0])


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _allocate(n: int, fractions: Sequence[float]) -> list[int]:
    """Largest-remainder allocation of ``n`` items to ``fractions`` (deterministic)."""
    raw = [n * f for f in fractions]
    counts = [math.floor(r) for r in raw]
    by_remainder = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1
    return counts


def stratified_splits(items: Sequence[EvalItem], *, fractions: Sequence[float] = (0.4, 0.3, 0.3),
                      n_splits: int = 200, seed: int = 0) -> list[tuple[list[int], list[int], list[int]]]:
    """``n_splits`` (train, calibration, test) index lists, stratified by category."""
    if len(fractions) != 3 or not math.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError("fractions must be three positive numbers summing to 1")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(item.category.value, []).append(i)
    splits = []
    for s in range(n_splits):
        rng = np.random.default_rng([seed, s])
        parts: tuple[list[int], list[int], list[int]] = ([], [], [])
        for cat in sorted(groups):
            idx = rng.permutation(groups[cat]).tolist()
            a, b, _ = _allocate(len(idx), fractions)
            parts[0].extend(idx[:a])
            parts[1].extend(idx[a:a + b])
            parts[2].extend(idx[a + b:])
        splits.append(tuple(sorted(p) for p in parts))
    return splits


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": None, "p05": None, "p95": None, "n": 0}
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "n": int(v.size)}


def model_factories(available: Sequence[str]) -> dict[str, Callable[[], LogisticCalibrator]]:
    models: dict[str, Callable[[], LogisticCalibrator]] = {"combined": lambda: LogisticCalibrator(list(available))}
    for f in SINGLE_SIGNALS:
        if f in available:
            models[f"platt:{f}"] = (lambda f=f: platt(f))
    return models


def _threshold_outcome(tr, test_probs: np.ndarray, test_y: np.ndarray) -> dict:
    answered = test_probs >= tr.tau
    n_ans = int(answered.sum())
    risk = float(1 - test_y[answered].mean()) if n_ans else math.nan
    return {"tau": tr.tau, "cal_coverage": tr.cal_coverage, "test_coverage": n_ans / test_y.size,
            "test_risk": risk, "abstain_all": tr.abstains_on_all, "reason": tr.reason}


def run_abstention_benchmark(
    eval_set: EvalSet,
    retriever,
    *,
    k_ctx: int = 3,
    top_k: int = 10,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    delta: float = 0.10,
    n_splits: int = 200,
    seed: int = 0,
) -> dict:
    items = eval_set.items
    evidence = [extract_features(item.question, retriever, k_ctx=k_ctx, top_k=top_k) for item in items]
    labels = np.array([safe_to_answer(ev, item.ground_truth_chunk_ids) for ev, item in zip(evidence, items)])
    rows = [ev.features for ev in evidence]
    available = list(rows[0])
    models = model_factories(available)

    misses = [i for i, item in enumerate(items) if item.is_answerable and not labels[i]]
    label_stats = {
        "n": len(items), "safe": int(labels.sum()), "unsafe": int(len(items) - labels.sum()),
        "unsafe_unanswerable": sum(not item.is_answerable for item in items),
        "unsafe_retrieval_miss": len(misses),
        "retrieval_miss_ids": [items[i].id for i in misses],
    }
    raw_auroc = {f: _auroc_labels([r[f] for r in rows], labels) for f in available}
    answerable = np.array([int(item.is_answerable) for item in items])
    raw_auroc_answerability = {f: _auroc_labels([r[f] for r in rows], answerable) for f in available}

    splits = stratified_splits(items, n_splits=n_splits, seed=seed)
    per_model: dict[str, dict] = {}
    for name, make in models.items():
        metrics: dict[str, list[float]] = {m: [] for m in
                                           ["auroc", "aurc", "eaurc", "brier", "ece"]
                                           + [f"sel_acc@{c:g}" for c in COVERAGE_LEVELS]}
        outcomes: dict[str, dict[str, list[dict]]] = {f"{a:g}": {"erm": [], "ltt": []} for a in alphas}
        pooled_p: list[float] = []
        pooled_y: list[int] = []
        skipped = 0
        for train, cal, test in splits:
            y_tr, y_cal, y_te = labels[train], labels[cal], labels[test]
            if y_tr.min() == y_tr.max():
                skipped += 1
                continue
            model = make().fit([rows[i] for i in train], y_tr)
            p_tr = model.predict_proba([rows[i] for i in train])
            p_cal = model.predict_proba([rows[i] for i in cal])
            p_te = model.predict_proba([rows[i] for i in test])
            pooled_p.extend(p_te.tolist())
            pooled_y.extend(y_te.tolist())
            metrics["auroc"].append(_auroc_labels(p_te, y_te))
            metrics["aurc"].append(aurc(p_te, y_te))
            metrics["eaurc"].append(eaurc(p_te, y_te))
            metrics["brier"].append(brier(p_te, y_te))
            metrics["ece"].append(ece(p_te, y_te))
            for c in COVERAGE_LEVELS:
                metrics[f"sel_acc@{c:g}"].append(selective_accuracy_at(p_te, y_te, c))
            for a in alphas:
                outcomes[f"{a:g}"]["erm"].append(_threshold_outcome(erm_threshold(p_cal, y_cal, a), p_te, y_te))
                grid = coverage_grid(p_tr, len(cal), a, delta)  # fixed before seeing calibration labels
                outcomes[f"{a:g}"]["ltt"].append(
                    _threshold_outcome(ltt_threshold(p_cal, y_cal, a, delta, grid=grid), p_te, y_te))

        thresholds = {}
        for a_key, by_method in outcomes.items():
            alpha = float(a_key)
            thresholds[a_key] = {}
            for method, outs in by_method.items():
                answered = [o for o in outs if o["test_coverage"] > 0]
                thresholds[a_key][method] = {
                    "test_coverage": _summary([o["test_coverage"] for o in outs]),
                    "test_risk": _summary([o["test_risk"] for o in answered]),
                    "cal_coverage": _summary([o["cal_coverage"] for o in outs]),
                    "validity_rate": (float(np.mean([o["test_risk"] <= alpha + 1e-12 for o in answered]))
                                      if answered else None),
                    "violation_rate": float(np.mean([o["test_coverage"] > 0 and o["test_risk"] > alpha + 1e-12
                                                     for o in outs])) if outs else None,
                    "abstain_all_rate": float(np.mean([o["abstain_all"] for o in outs])) if outs else None,
                    "abstain_reasons": sorted({o["reason"] for o in outs if o["reason"]}),
                }
        pooled = {}
        if pooled_p:
            cov, risk = risk_coverage_curve(pooled_p, pooled_y)
            pooled = {"auroc": _auroc_labels(pooled_p, pooled_y), "aurc": aurc(pooled_p, pooled_y),
                      "brier": brier(pooled_p, pooled_y), "ece": ece(pooled_p, pooled_y),
                      "risk_coverage": {"coverage": cov.tolist(), "risk": risk.tolist()},
                      "reliability": reliability_bins(pooled_p, pooled_y)}
        per_model[name] = {"metrics": {m: _summary(v) for m, v in metrics.items()}, "thresholds": thresholds,
                           "pooled": pooled, "skipped_splits": skipped}

    backend = None
    if isinstance(retriever, RerankingRetriever):
        backend = retriever.reranker.backend
    first = retriever.first_stage if isinstance(retriever, RerankingRetriever) else retriever
    return {
        "config": {
            "embedder": getattr(first.embedder, "name", type(first.embedder).__name__),
            "reranker_backend": backend,
            "reranker_fallback_reason": retriever.reranker.fallback_reason if backend else None,
            "k_ctx": k_ctx, "top_k": top_k, "alphas": list(alphas), "delta": delta,
            "n_splits": n_splits, "split_fractions": [0.4, 0.3, 0.3], "seed": seed,
            "features": available, "label": f"answerable and a gold chunk in the top-{k_ctx}",
            "ltt_min_answered": {f"{a:g}": min_certifiable_n(a, delta) for a in alphas},
            "eval_set_generator": eval_set.generator, "corpus": eval_set.corpus.model_dump(),
        },
        "label_stats": label_stats,
        "raw_feature_auroc": raw_auroc,
        "raw_feature_auroc_answerability": raw_auroc_answerability,
        "models": per_model,
    }


def fit_policy(eval_set: EvalSet, retriever, *, k_ctx: int = 3, top_k: int = 10, alpha: float = 0.10,
               delta: float = 0.10, method: str = "ltt", seed: int = 0) -> AbstentionPolicy:
    """Deployable policy: combined model fitted on train, τ chosen on calibration (split 0)."""
    items = eval_set.items
    evidence = [extract_features(item.question, retriever, k_ctx=k_ctx, top_k=top_k) for item in items]
    labels = np.array([safe_to_answer(ev, item.ground_truth_chunk_ids) for ev, item in zip(evidence, items)])
    rows = [ev.features for ev in evidence]
    train, cal, _ = stratified_splits(items, n_splits=1, seed=seed)[0]
    model = LogisticCalibrator(list(rows[0])).fit([rows[i] for i in train], labels[train])
    p_cal = model.predict_proba([rows[i] for i in cal])
    grid = coverage_grid(model.predict_proba([rows[i] for i in train]), len(cal), alpha, delta)
    tr = (ltt_threshold(p_cal, labels[cal], alpha, delta, grid=grid) if method == "ltt"
          else erm_threshold(p_cal, labels[cal], alpha))
    meta = {**tr.to_dict(), "k_ctx": k_ctx, "top_k": top_k, "seed": seed,
            "reranker_backend": retriever.reranker.backend if isinstance(retriever, RerankingRetriever) else None}
    return AbstentionPolicy(model, tr.tau, meta)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(s: dict | None, digits: int = 3) -> str:
    if not s or s.get("mean") is None:
        return "—"
    return f"{s['mean']:.{digits}f} <sub>[{s['p05']:.2f}, {s['p95']:.2f}]</sub>"


def render_markdown(report: dict) -> str:
    cfg, ls = report["config"], report["label_stats"]
    lines = [
        "# Abstention Benchmark",
        "",
        f"- Label: **safe to answer** = {cfg['label']}. Base rate: {ls['safe']}/{ls['n']} safe; unsafe = "
        f"{ls['unsafe_unanswerable']} unanswerable + {ls['unsafe_retrieval_miss']} retrieval misses.",
        f"- Protocol: {cfg['n_splits']} stratified train/cal/test splits ({'/'.join(f'{f:.0%}' for f in cfg['split_fractions'])}); "
        f"confidence models fitted on train, τ chosen on calibration, everything scored on test. "
        f"Cells: mean <sub>[5th, 95th percentile]</sub> over splits.",
        f"- Embedder `{cfg['embedder']}` · reranker `{cfg['reranker_backend']}` · features: "
        + ", ".join(f"`{f}`" for f in cfg["features"]),
    ]
    if cfg["embedder"].startswith("hashing"):
        lines.append("- ⚠️ Dense retrieval uses the offline **hashing** embedder (lexical, not semantic).")
    if cfg["reranker_backend"] and cfg["reranker_backend"].startswith("mock"):
        lines.append("- ⚠️ The reranker is the deterministic **mock**, not a neural cross-encoder"
                     + (f" (fallback: `{cfg['reranker_fallback_reason']}`)" if cfg["reranker_fallback_reason"] else "")
                     + ".")

    lines += ["", "## Discrimination and calibration (test folds)", "",
              "| Confidence model | AUROC ↑ | AURC ↓ | E-AURC ↓ | Brier ↓ | ECE ↓ | "
              + " | ".join(f"Sel. acc @{c:g} cov ↑" for c in COVERAGE_LEVELS) + " |",
              "|---" * (6 + len(COVERAGE_LEVELS)) + "|"]
    for name, m in report["models"].items():
        mm = m["metrics"]
        lines.append(f"| {name} | " + " | ".join(_fmt(mm[k]) for k in ["auroc", "aurc", "eaurc", "brier", "ece"])
                     + " | " + " | ".join(_fmt(mm[f"sel_acc@{c:g}"]) for c in COVERAGE_LEVELS) + " |")

    lines += ["", f"## Risk control (combined model, δ = {cfg['delta']})", "",
              "ERM picks the largest calibration coverage with empirical risk ≤ α (no guarantee). "
              "LTT gives P(risk ≤ α) ≥ 1 − δ, testing a coverage grid fixed on the train split. *Violation rate* = "
              "share of all splits whose **test** risk exceeds α (abstaining on everything never violates); this is "
              "what LTT bounds by δ. *Validity* = share of splits whose test risk is ≤ α, among splits that answered "
              "anything. Test folds are small, so this "
              "is itself a noisy estimate. LTT needs at least n_min answered calibration items to certify anything: "
              + ", ".join(f"α={a}: n_min={n}" for a, n in cfg["ltt_min_answered"].items())
              + f" (calibration folds have ≈{round(ls['n'] * cfg['split_fractions'][1])} items).", "",
              "| α | Method | Test coverage ↑ | Test selective risk | Violation rate (≤ δ for LTT) | Validity | "
              "Abstain-all rate |",
              "|---|---|---|---|---|---|---|"]
    combined = report["models"]["combined"]["thresholds"]
    for a_key, by_method in combined.items():
        for method, t in by_method.items():
            validity = "—" if t["validity_rate"] is None else f"{t['validity_rate']:.0%}"
            lines.append(f"| {a_key} | {method.upper()} | {_fmt(t['test_coverage'])} | {_fmt(t['test_risk'])} | "
                         f"{t['violation_rate']:.1%} | {validity} | {t['abstain_all_rate']:.0%} |")

    lines += ["", "## Raw signals (all items, no fitting)", "",
              "AUROC of each raw feature (higher value = predicted safer). Below 0.5 means the signal points the "
              "other way, e.g. entropy.", "",
              "| Feature | AUROC vs safe-to-answer | AUROC vs answerable |", "|---|---|---|"]
    for f, v in report["raw_feature_auroc"].items():
        lines.append(f"| `{f}` | {v:.3f} | {report['raw_feature_auroc_answerability'][f]:.3f} |")
    if ls["retrieval_miss_ids"]:
        lines += ["", f"Retrieval misses (answerable but no gold chunk in the top-{cfg['k_ctx']}): "
                  + ", ".join(f"`{i}`" for i in ls["retrieval_miss_ids"])]
    return "\n".join(lines) + "\n"


def write_figures(report: dict, outdir: Path | str) -> list[Path]:
    """Risk-coverage, coverage-accuracy and reliability plots (needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping figures")
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    models = {k: v for k, v in report["models"].items() if v["pooled"]}

    for fname, ylabel, transform in [("risk_coverage.png", "Selective risk", lambda r: r),
                                     ("coverage_accuracy.png", "Selective accuracy", lambda r: 1 - r)]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, m in models.items():
            rc = m["pooled"]["risk_coverage"]
            # risk_j holds on (coverage_{j-1}, coverage_j], so the step is drawn "pre"; with tied
            # scores the last jump can be wide, and "post" would hide it.
            ax.step(rc["coverage"], transform(np.asarray(rc["risk"])), where="pre",
                    label=f"{name} (AURC {m['pooled']['aurc']:.3f})", lw=2 if name == "combined" else 1)
        for a in report["config"]["alphas"]:
            if fname.startswith("risk"):
                ax.axhline(a, ls=":", color="grey", lw=0.8)
        ax.set_xlabel("Coverage")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_title("Pooled test folds")
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)
        paths.append(outdir / fname)

    rel = report["models"]["combined"]["pooled"]["reliability"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    xs = [b["mean_confidence"] for b in rel if b["count"]]
    ys = [b["fraction_safe"] for b in rel if b["count"]]
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Predicted P(safe)")
    ax.set_ylabel("Observed fraction safe")
    ax.set_title(f"Reliability (combined, ECE {report['models']['combined']['pooled']['ece']:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "reliability.png", dpi=150)
    plt.close(fig)
    paths.append(outdir / "reliability.png")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.abstention_eval", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="selective-prediction benchmark of the abstention layer")
    run.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.json"))
    run.add_argument("--corpus", type=Path, default=None)
    run.add_argument("--k-ctx", type=int, default=3, help="generator context size used by the label")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    run.add_argument("--delta", type=float, default=0.10)
    run.add_argument("--n-splits", type=int, default=200)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    run.add_argument("--figures", type=Path, default=Path("data/eval/figures"))
    run.add_argument("--no-figures", action="store_true")
    run.add_argument("--save-policy", type=Path, default=None, help="also fit and save a deployable policy (JSON)")
    run.add_argument("--policy-alpha", type=float, default=0.10)
    run.add_argument("--policy-method", choices=("ltt", "erm"), default="ltt")
    add_pipeline_args(run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    eval_set = EvalSet.load(args.eval_set)
    corpus_path = args.corpus or Path(eval_set.corpus.path)
    chunks = load_chunks_jsonl(corpus_path)
    if corpus_fingerprint(chunks) != eval_set.corpus.sha256:
        logger.warning("corpus %s differs from the one the eval set was built on", corpus_path)
    first, reranker = build_pipeline(chunks, args)
    retriever = (RerankingRetriever(first, reranker, retrieve_k=args.retrieve_k) if reranker is not None else first)

    report = run_abstention_benchmark(eval_set, retriever, k_ctx=args.k_ctx, top_k=args.top_k, alphas=args.alphas,
                                      delta=args.delta, n_splits=args.n_splits, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    figures = [] if args.no_figures else write_figures(report, args.figures)
    print(render_markdown(report))
    print(f"wrote {args.output}, {md_path}" + (f" and {len(figures)} figures in {args.figures}" if figures else ""))
    if args.save_policy:
        policy = fit_policy(eval_set, retriever, k_ctx=args.k_ctx, top_k=args.top_k, alpha=args.policy_alpha,
                            delta=args.delta, method=args.policy_method, seed=args.seed)
        policy.save(args.save_policy)
        print(f"saved policy to {args.save_policy} (τ = {policy.tau:.4f}, method {args.policy_method})")
    return 0


def _jsonable(obj):
    """Strict JSON: NaN -> null, ±inf -> "inf"/"-inf"."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
    if isinstance(obj, (np.floating, np.integer)):
        return _jsonable(obj.item())
    return obj


if __name__ == "__main__":
    raise SystemExit(main())

