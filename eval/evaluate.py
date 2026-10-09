"""Retrieval benchmark harness: Hit@K, MRR@K and Recall@K for dense, sparse and hybrid retrieval.

Metrics are computed over the **answerable** items (only they have gold chunks), each
with a seeded percentile-bootstrap 95% CI. An "abstention preview" also asks how well
each retriever's top-1 score separates answerable from unanswerable questions (AUROC).
That is the signal Phase 5 will calibrate.

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
from src.retriever import RETRIEVAL_MODES, HashingEmbedder, HybridRetriever, SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_REPORT = Path("data/eval/retrieval_benchmark.json")
RETRIEVER_LABELS = {"dense": "Dense (FAISS)", "sparse": "Sparse (BM25)", "hybrid": "Hybrid (RRF)"}

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


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5·P(tie) (the Mann–Whitney U statistic, normalised)."""
    pos, neg = np.asarray(positive_scores, float), np.asarray(negative_scores, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


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
    }
    return RetrieverReport(name, len(answerable), metrics, ci, by_category, preview, rows)


def retriever_fn(retriever: HybridRetriever, mode: str) -> RetrieveFn:
    def _fn(query: str, k: int) -> list[tuple[str, float]]:
        return [(r.chunk.chunk_id, r.score) for r in retriever.retrieve(query, top_k=k, mode=mode)]

    return _fn


def run_benchmark(
    eval_set: EvalSet,
    retriever: HybridRetriever,
    *,
    modes: Sequence[str] = RETRIEVAL_MODES,
    ks: Sequence[int] = DEFAULT_KS,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    eval_set.validate_against_corpus(c.chunk_id for c in retriever.chunks)
    reports = {m: evaluate_retriever(RETRIEVER_LABELS[m], retriever_fn(retriever, m), eval_set.items,
                                     ks=ks, n_bootstrap=n_bootstrap, seed=seed) for m in modes}
    return {
        "config": {
            "embedder": getattr(retriever.embedder, "name", type(retriever.embedder).__name__),
            "rrf_k": retriever.rrf_k,
            "candidate_pool": retriever.candidate_pool,
            "ks": sorted(set(ks)),
            "bootstrap_resamples": n_bootstrap,
            "seed": seed,
            "eval_set_generator": eval_set.generator,
            "eval_set_counts": eval_set.counts,
            "corpus": eval_set.corpus.model_dump(),
        },
        "retrievers": {m: r.to_dict() for m, r in reports.items()},
    }


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

    lines += ["", "## Abstention preview (top-1 retrieval score)", "",
              "AUROC = how well the top-1 score alone separates answerable from unanswerable questions "
              "(0.5 = chance).", "",
              "| Retriever | mean top-1 (answerable) | mean top-1 (unanswerable) | AUROC |", "|---|---|---|---|"]
    for r in retrievers.values():
        p = r["abstention_preview"]
        lines.append(f"| {r['name']} | {p['mean_top1_answerable']:.4f} | "
                     f"{p['mean_top1_unanswerable']:.4f} | {p['auroc_top1']:.3f} |")
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.evaluate", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="benchmark retrievers against an eval set")
    run.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.json"))
    run.add_argument("--corpus", type=Path, default=None, help="default: the corpus path recorded in the eval set")
    run.add_argument("--baseline", action="store_true", help="compare dense, sparse and hybrid (default: hybrid)")
    run.add_argument("--embedder", choices=("bge", "hashing"), default="bge")
    run.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    run.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    run.add_argument("--rrf-k", type=float, default=60)
    run.add_argument("--bootstrap", type=int, default=1000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=DEFAULT_REPORT)
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

    embedder = HashingEmbedder() if args.embedder == "hashing" else SentenceTransformerEmbedder(args.model)
    try:
        retriever = HybridRetriever(embedder, rrf_k=args.rrf_k).index(chunks)
    except (ImportError, OSError) as exc:  # missing package, or model weights not downloadable
        raise SystemExit(f"could not load embedder {args.model!r} ({exc}). "
                         "Install sentence-transformers / check network, or pass --embedder hashing.") from exc

    modes = RETRIEVAL_MODES if args.baseline else ("hybrid",)
    report = run_benchmark(eval_set, retriever, modes=modes, ks=args.ks,
                           n_bootstrap=args.bootstrap, seed=args.seed)
    json_path, md_path = write_reports(report, args.output)
    print(render_markdown(report))
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
