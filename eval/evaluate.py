"""Retrieval benchmark harness: Hit@K, MRR@K and Recall@K for dense, sparse and hybrid retrieval.

Metrics are computed over the **answerable** items (only they have gold chunks), each
with a seeded percentile-bootstrap 95% CI. An "abstention preview" also asks how well
each retriever's top-1 score separates answerable from unanswerable questions (AUROC).
That is the signal the abstention layer (Phase 4) calibrates.

CLI::

    python -m eval.evaluate run --eval-set data/eval/eval_set.json --baseline
    python -m eval.evaluate run --eval-set data/eval/eval_set.json --baseline --embedder hashing
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from eval.schemas import EvalItem, EvalSet, corpus_fingerprint
from src.data_loader import load_chunks_jsonl
from src.reranker import DEFAULT_RERANKER_MODEL, Reranker, RerankingRetriever
from src.retriever import RETRIEVAL_MODES, HashingEmbedder, HybridRetriever, SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_REPORT = Path("data/eval/retrieval_benchmark.json")
RETRIEVER_LABELS = {"dense": "Dense (FAISS)", "sparse": "Sparse (BM25)", "hybrid": "Hybrid (RRF)",
                    "hybrid_reranked": "Hybrid + Cross-Encoder"}
RERANKED = "hybrid_reranked"

# A retrieval function returns (chunk_id, score) pairs, best first.
RetrieveFn = Callable[[str, int], list[tuple[str, float]]]


# ---------------------------------------------------------------------------
# Metrics (pure functions)
# ---------------------------------------------------------------------------


def _check(gold: Sequence[str], k: int | None = None) -> set[str]:
    if not gold:
        raise ValueError("gold set must be non-empty")
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    return set(gold)


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top-k, else 0.0."""
    g = _check(gold, k)
    return float(any(cid in g for cid in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str], k: int | None = None) -> float:
    """1 / rank of the first relevant chunk (within the top-k if given), 0.0 if none."""
    g = _check(gold, k)
    for rank, cid in enumerate(ranked[:k] if k else ranked, start=1):
        if cid in g:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top-k."""
    g = _check(gold, k)
    return len(g & set(ranked[:k])) / len(g)


def bootstrap_ci(values: Sequence[float], *, n_resamples: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (deterministic for a given seed)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000,
                           alpha: float = 0.05, seed: int = 0) -> dict[str, float | bool | list[float]]:
    """Mean of ``b - a`` over the same items, with a paired percentile-bootstrap CI.

    Resampling *items* (not tracks) keeps the pairing, so the CI reflects per-question
    differences rather than the much wider spread of two independent means.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("paired samples must be non-empty and of equal length")
    diff = y - x
    lo, hi = bootstrap_ci(diff, n_resamples=n_resamples, alpha=alpha, seed=seed)
    return {"delta": float(diff.mean()), "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0)}


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5·P(tie) (the Mann–Whitney U statistic, normalised)."""
    pos, neg = np.asarray(positive_scores, float), np.asarray(negative_scores, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def auroc_ci(positive_scores: Sequence[float], negative_scores: Sequence[float], *,
             n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Stratified bootstrap CI: positives and negatives are resampled separately."""
    pos, neg = np.asarray(positive_scores, float), np.asarray(negative_scores, float)
    if pos.size == 0 or neg.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [auroc(pos[rng.integers(0, pos.size, pos.size)], neg[rng.integers(0, neg.size, neg.size)])
             for _ in range(n_resamples)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def paired_auroc_delta(pos_a: Sequence[float], neg_a: Sequence[float], pos_b: Sequence[float],
                       neg_b: Sequence[float], *, n_resamples: int = 1000, alpha: float = 0.05,
                       seed: int = 0) -> dict[str, float | bool | list[float]]:
    """AUROC(b) − AUROC(a) on the same questions; each resample uses the same indices for both."""
    pa, na, pb, nb = (np.asarray(v, float) for v in (pos_a, neg_a, pos_b, neg_b))
    if pa.shape != pb.shape or na.shape != nb.shape or pa.size == 0 or na.size == 0:
        raise ValueError("paired AUROC needs matching, non-empty positive and negative samples")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_resamples):
        ip, ineg = rng.integers(0, pa.size, pa.size), rng.integers(0, na.size, na.size)
        deltas.append(auroc(pb[ip], nb[ineg]) - auroc(pa[ip], na[ineg]))
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {"delta": auroc(pb, nb) - auroc(pa, na), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class RetrieverReport:
    name: str
    n_answerable: int
    metrics: dict[str, float]
    ci95: dict[str, tuple[float, float]]
    by_category: dict[str, dict[str, float]]
    abstention_preview: dict[str, float]
    per_item: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_answerable": self.n_answerable,
            "metrics": self.metrics,
            "ci95": {k: list(v) for k, v in self.ci95.items()},
            "by_category": self.by_category,
            "abstention_preview": self.abstention_preview,
            "per_item": self.per_item,
        }


def metric_names(ks: Sequence[int]) -> list[str]:
    max_k = max(ks)
    return [f"hit@{k}" for k in ks] + [f"mrr@{max_k}"] + [f"recall@{k}" for k in ks]


def score_item(ranked: Sequence[str], gold: Sequence[str], ks: Sequence[int]) -> dict[str, float]:
    max_k = max(ks)
    row = {f"hit@{k}": hit_at_k(ranked, gold, k) for k in ks}
    row[f"mrr@{max_k}"] = reciprocal_rank(ranked, gold, max_k)
    row.update({f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks})
    return row


def evaluate_retriever(
    name: str,
    retrieve: RetrieveFn,
    items: Sequence[EvalItem],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> RetrieverReport:
    if not ks or min(ks) <= 0:
        raise ValueError("ks must be positive integers")
    ks = sorted(set(ks))
    max_k = max(ks)
    names = metric_names(ks)
    rows: list[dict] = []
    top1 = {True: [], False: []}
    for item in items:
        hits = retrieve(item.question, max_k)
        ranked = [cid for cid, _ in hits]
        top1[item.is_answerable].append(hits[0][1] if hits else float("-inf"))
        row: dict = {"id": item.id, "category": item.category.value, "is_answerable": item.is_answerable,
                     "retrieved": ranked, "top1_score": hits[0][1] if hits else None}
        if item.is_answerable:
            row.update(score_item(ranked, item.ground_truth_chunk_ids, ks))
        rows.append(row)

    answerable = [r for r in rows if r["is_answerable"]]
    if not answerable:
        raise ValueError("the eval set has no answerable items")
    metrics = {m: float(np.mean([r[m] for r in answerable])) for m in names}
    ci = {m: bootstrap_ci([r[m] for r in answerable], n_resamples=n_bootstrap, seed=seed) for m in names}
    by_category: dict[str, dict[str, float]] = {}
    for cat in sorted({r["category"] for r in answerable}):
        sub = [r for r in answerable if r["category"] == cat]
        by_category[cat] = {"n": len(sub), **{m: float(np.mean([r[m] for r in sub])) for m in names}}

    finite = lambda xs: [x for x in xs if np.isfinite(x)]  # noqa: E731
    preview = {
        "mean_top1_answerable": float(np.mean(finite(top1[True]))) if finite(top1[True]) else float("nan"),
        "mean_top1_unanswerable": float(np.mean(finite(top1[False]))) if finite(top1[False]) else float("nan"),
        "auroc_top1": auroc(top1[True], top1[False]),
        "auroc_top1_ci95": list(auroc_ci(top1[True], top1[False], n_resamples=n_bootstrap, seed=seed)),
    }
    return RetrieverReport(name, len(answerable), metrics, ci, by_category, preview, rows)


def retriever_fn(retriever: HybridRetriever | RerankingRetriever, mode: str | None = None) -> RetrieveFn:
    def _fn(query: str, k: int) -> list[tuple[str, float]]:
        kwargs = {} if mode is None else {"mode": mode}
        return [(r.chunk.chunk_id, r.score) for r in retriever.retrieve(query, top_k=k, **kwargs)]

    return _fn


def _top1_by_answerability(report: RetrieverReport) -> tuple[list[float], list[float]]:
    score = lambda r: r["top1_score"] if r["top1_score"] is not None else float("-inf")  # noqa: E731
    return ([score(r) for r in report.per_item if r["is_answerable"]],
            [score(r) for r in report.per_item if not r["is_answerable"]])


def compare_tracks(base: RetrieverReport, other: RetrieverReport, ks: Sequence[int], *,
                   n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Paired comparison of ``other`` against ``base`` on identical questions."""
    base_rows = [r for r in base.per_item if r["is_answerable"]]
    other_rows = [r for r in other.per_item if r["is_answerable"]]
    if [r["id"] for r in base_rows] != [r["id"] for r in other_rows]:
        raise ValueError("tracks were not evaluated on the same items")
    metrics = {m: paired_bootstrap_delta([r[m] for r in base_rows], [r[m] for r in other_rows],
                                         n_resamples=n_bootstrap, seed=seed) for m in metric_names(ks)}
    pa, na = _top1_by_answerability(base)
    pb, nb = _top1_by_answerability(other)
    return {"baseline": base.name, "candidate": other.name, "metrics": metrics,
            "auroc_top1": paired_auroc_delta(pa, na, pb, nb, n_resamples=n_bootstrap, seed=seed)}


def candidate_recall(reranking: RerankingRetriever, items: Sequence[EvalItem]) -> float:
    """Recall of the first-stage candidate pool: the ceiling no reranker can exceed."""
    vals = []
    for item in items:
        if item.is_answerable:
            pool = {r.chunk.chunk_id for r in reranking.candidates(item.question)}
            vals.append(recall_at_k(list(pool), item.ground_truth_chunk_ids, max(len(pool), 1)))
    return float(np.mean(vals)) if vals else float("nan")


def run_benchmark(
    eval_set: EvalSet,
    retriever: HybridRetriever,
    *,
    modes: Sequence[str] = RETRIEVAL_MODES,
    reranker: Reranker | None = None,
    retrieve_k: int = 50,
    first_stage_mode: str = "hybrid",
    ks: Sequence[int] = DEFAULT_KS,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    eval_set.validate_against_corpus(c.chunk_id for c in retriever.chunks)
    ks = sorted(set(ks))
    reports = {m: evaluate_retriever(RETRIEVER_LABELS[m], retriever_fn(retriever, m), eval_set.items,
                                     ks=ks, n_bootstrap=n_bootstrap, seed=seed) for m in modes}
    config = {
        "embedder": getattr(retriever.embedder, "name", type(retriever.embedder).__name__),
        "rrf_k": retriever.rrf_k,
        "candidate_pool": retriever.candidate_pool,
        "ks": ks,
        "bootstrap_resamples": n_bootstrap,
        "seed": seed,
        "eval_set_generator": eval_set.generator,
        "eval_set_counts": eval_set.counts,
        "corpus": eval_set.corpus.model_dump(),
    }
    comparisons = {}
    extras: dict[str, dict] = {}
    if reranker is not None:
        if retrieve_k < max(ks):
            raise ValueError("retrieve_k must be at least max(ks)")
        two_stage = RerankingRetriever(retriever, reranker, retrieve_k=retrieve_k, first_stage_mode=first_stage_mode)
        reports[RERANKED] = evaluate_retriever(RETRIEVER_LABELS[RERANKED], retriever_fn(two_stage), eval_set.items,
                                               ks=ks, n_bootstrap=n_bootstrap, seed=seed)
        config.update({"reranker_backend": reranker.backend, "reranker_fallback_reason": reranker.fallback_reason,
                       "retrieve_k": retrieve_k, "first_stage_mode": first_stage_mode})
        extras[RERANKED] = {f"candidate_recall@{retrieve_k}": candidate_recall(two_stage, eval_set.items)}
        if first_stage_mode in reports:
            comparisons[f"{RERANKED}_vs_{first_stage_mode}"] = compare_tracks(
                reports[first_stage_mode], reports[RERANKED], ks, n_bootstrap=n_bootstrap, seed=seed)
    retrievers = {m: {**r.to_dict(), **extras.get(m, {})} for m, r in reports.items()}
    return {"config": config, "retrievers": retrievers, "comparisons": comparisons}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    retrievers = report["retrievers"]
    names = metric_names(cfg["ks"])
    best = {m: max(r["metrics"][m] for r in retrievers.values()) for m in names}

    def cell(r: dict, m: str) -> str:
        v = r["metrics"][m]
        lo, hi = r["ci95"][m]
        txt = f"{v:.3f}"
        return (f"**{txt}**" if len(retrievers) > 1 and np.isclose(v, best[m]) else txt) + \
            f" <sub>[{lo:.2f}, {hi:.2f}]</sub>"

    n = next(iter(retrievers.values()))["n_answerable"]
    lines = [
        "# Retrieval Benchmark",
        "",
        f"- Embedder: `{cfg['embedder']}` · RRF k = {cfg['rrf_k']} · candidate pool = {cfg['candidate_pool']}",
        f"- Eval set: `{cfg['eval_set_generator']}`, {n} answerable items scored "
        f"(corpus `{cfg['corpus']['path']}`, {cfg['corpus']['num_chunks']} chunks)",
        f"- Cells show the mean with a 95% percentile-bootstrap CI ({cfg['bootstrap_resamples']} resamples); "
        "best value per column in bold.",
    ]
    if cfg["embedder"].startswith("hashing"):
        lines.append("- ⚠️ The dense retriever uses the offline **hashing** embedder (lexical, not semantic). "
                     "Re-run with `--embedder bge` for real dense-retrieval numbers.")
    if "reranker_backend" in cfg:
        ceiling = retrievers[RERANKED].get(f"candidate_recall@{cfg['retrieve_k']}")
        lines.append(f"- Reranker: `{cfg['reranker_backend']}` over the {cfg['first_stage_mode']} top-"
                     f"{cfg['retrieve_k']} (candidate recall ceiling = {ceiling:.3f})")
        if cfg["reranker_backend"].startswith("mock"):
            lines.append("- ⚠️ The reranker is the deterministic **mock** (lexical interaction heuristic), not a "
                         "neural cross-encoder" + (f" — fallback reason: `{cfg['reranker_fallback_reason']}`"
                                                   if cfg.get("reranker_fallback_reason") else "")
                         + ". Re-run with `--reranker cross-encoder --strict-reranker` for real numbers.")
    lines += ["", "## Overall", "", "| Retriever | " + " | ".join(names) + " |",
              "|---" * (len(names) + 1) + "|"]
    for r in retrievers.values():
        lines.append(f"| {r['name']} | " + " | ".join(cell(r, m) for m in names) + " |")

    cats = sorted({c for r in retrievers.values() for c in r["by_category"]})
    focus = [m for m in names if m.startswith(("hit@1", "mrr", "recall@5", "recall@10"))]
    lines += ["", "## By category", "", "| Retriever | Category | n | " + " | ".join(focus) + " |",
              "|---" * (len(focus) + 3) + "|"]
    for r in retrievers.values():
        for c in cats:
            row = r["by_category"].get(c)
            if row:
                lines.append(f"| {r['name']} | {c} | {row['n']} | "
                             + " | ".join(f"{row[m]:.3f}" for m in focus) + " |")

    for comp in report.get("comparisons", {}).values():
        lines += ["", f"## Δ {comp['candidate']} vs {comp['baseline']} (paired bootstrap, same questions)", "",
                  "| Metric | Δ | 95% CI | Significant |", "|---|---|---|---|"]
        for m, d in list(comp["metrics"].items()) + [("AUROC (top-1)", comp["auroc_top1"])]:
            lo, hi = d["ci95"]
            lines.append(f"| {m} | {d['delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'yes' if d['significant'] else 'no'} |")

    lines += ["", "## Abstention preview (top-1 score)", "",
              "AUROC = how well the top-1 score alone separates answerable from unanswerable questions "
              "(0.5 = chance). For the reranked track the score is the cross-encoder logit.", "",
              "| Retriever | mean top-1 (answerable) | mean top-1 (unanswerable) | AUROC | 95% CI |",
              "|---|---|---|---|---|"]
    for r in retrievers.values():
        p = r["abstention_preview"]
        lo, hi = p.get("auroc_top1_ci95", [float("nan")] * 2)
        lines.append(f"| {r['name']} | {p['mean_top1_answerable']:.4f} | "
                     f"{p['mean_top1_unanswerable']:.4f} | {p['auroc_top1']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    return "\n".join(lines) + "\n"


def write_reports(report: dict, json_path: Path | str) -> tuple[Path, Path]:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by every command that builds the retrieval pipeline."""
    parser.add_argument("--embedder", choices=("bge", "hashing"), default="bge")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--rrf-k", type=float, default=60)
    parser.add_argument("--reranker", choices=("cross-encoder", "mock", "none"), default="cross-encoder",
                        help="second stage over the hybrid candidates; 'cross-encoder' falls back to the mock "
                             "(with a warning) if the model cannot be loaded, unless --strict-reranker")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--strict-reranker", action="store_true", help="fail instead of falling back to the mock")
    parser.add_argument("--retrieve-k", type=int, default=50, help="first-stage candidates passed to the reranker")
    parser.add_argument("--device", default=None)


def build_pipeline(chunks, args: argparse.Namespace) -> tuple[HybridRetriever, Reranker | None]:
    """Index ``chunks`` and construct the (optional) reranker, failing with actionable messages."""
    embedder = HashingEmbedder() if args.embedder == "hashing" else SentenceTransformerEmbedder(args.model)
    try:
        retriever = HybridRetriever(embedder, rrf_k=args.rrf_k).index(chunks)
    except (ImportError, OSError) as exc:  # missing package, or model weights not downloadable
        raise SystemExit(f"could not load embedder {args.model!r} ({exc}). "
                         "Install sentence-transformers / check network, or pass --embedder hashing.") from exc
    reranker = None
    if args.reranker != "none":
        reranker = Reranker(args.reranker_model, device=args.device, mock=args.reranker == "mock",
                            allow_fallback=not args.strict_reranker)
        try:
            backend = reranker.backend  # triggers the (lazy) model load
        except RuntimeError as exc:
            raise SystemExit(f"{exc}\nInstall sentence-transformers + torch and check network access, "
                             "or drop --strict-reranker to use the offline mock.") from exc
        logger.info("reranker backend: %s", backend)
    return retriever, reranker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.evaluate", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="benchmark retrievers against an eval set")
    run.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.json"))
    run.add_argument("--corpus", type=Path, default=None, help="default: the corpus path recorded in the eval set")
    run.add_argument("--baseline", action="store_true",
                     help="also run the dense and sparse baselines (default: hybrid [+ reranked])")
    run.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    run.add_argument("--bootstrap", type=int, default=1000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=DEFAULT_REPORT)
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

    retriever, reranker = build_pipeline(chunks, args)
    modes = RETRIEVAL_MODES if args.baseline else ("hybrid",)
    report = run_benchmark(eval_set, retriever, modes=modes, reranker=reranker, retrieve_k=args.retrieve_k,
                           ks=args.ks, n_bootstrap=args.bootstrap, seed=args.seed)
    json_path, md_path = write_reports(report, args.output)
    print(render_markdown(report))
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
