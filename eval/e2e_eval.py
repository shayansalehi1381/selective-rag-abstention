"""End-to-end benchmark: standard RAG vs selective RAG, scored on answer correctness.

For every eval item the pipeline runs **once**: retrieval and abstention features, then
two reader calls on the same top-k context, one abstain-allowed and one forced. Those
cached outcomes are reused across all splits, so a real LLM costs about two calls per item.

**Correctness** of an answer to an answerable item is ``token-F1 ≥ 0.5 AND key-fact
match`` (every number and entity of the reference appears in the prediction). An
unanswerable item is handled correctly only by abstaining.

Systems (all evaluated on the same test folds):

==============================  ==========================  ====================
system                          pre-generation gate (τ)     reader
==============================  ==========================  ====================
standard_rag                    none                        forced
rag_self_abstain                none                        abstain-allowed
selective_rag[erm|ltt]          g(x) ≥ τ                    forced
selective_rag_self[erm|ltt]     g(x) ≥ τ                    abstain-allowed
==============================  ==========================  ====================

For the gated systems, a confidence model is refitted on each **train** fold with that
system's own correctness label. ``τ`` is chosen on the **calibration** fold (ERM, and
LTT on a train-derived grid), and everything is scored on the **test** fold. This
replaces Phase 4's retrieval-proxy label with end-to-end answer correctness.

CLI::

    python -m eval.e2e_eval run --eval-set data/eval/eval_set.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from eval.abstention_eval import _jsonable, _summary, stratified_splits
from eval.evaluate import add_pipeline_args, build_pipeline
from eval.schemas import EvalSet, corpus_fingerprint
from src.abstention import (
    AbstentionPolicy,
    LogisticCalibrator,
    coverage_grid,
    erm_threshold,
    extract_features,
    ltt_threshold,
    min_certifiable_n,
)
from src.data_loader import load_chunks_jsonl
from src.generator import Generator, read_both
from src.reranker import RerankingRetriever

logger = logging.getLogger(__name__)

DEFAULT_REPORT = Path("data/eval/e2e_benchmark.json")
DEFAULT_CACHE = Path("data/eval/generation_cache.jsonl")
DEFAULT_ALPHAS = (0.1, 0.2, 0.3)
F1_THRESHOLD = 0.5
SYSTEM_LABELS = {
    "standard_rag": "Standard RAG (always answers)",
    "rag_self_abstain": "RAG + reader self-abstention",
    "selective_rag": "Selective RAG (τ gate)",
    "selective_rag_self": "Selective RAG (τ gate) + self-abstention",
}


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_FACT_NUMBER = re.compile(r"(?<![\w@.\-])\d+(?:\.\d+)?(?:e-?\d+)?%?(?![\w@])")


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def key_facts(reference: str) -> list[str]:
    """Numbers and entity-like tokens of a reference (mixed case, all caps, or with ``@``)."""
    facts = _FACT_NUMBER.findall(reference)
    for raw in reference.split():
        tok = raw.strip(string.punctuation.replace("@", "").replace("-", ""))
        if tok and (any(c.isupper() for c in tok[1:]) or "@" in tok or (tok.isupper() and len(tok) > 1)):
            facts.append(tok)
    return list(dict.fromkeys(facts))


def key_fact_match(prediction: str, reference: str) -> float:
    """1.0 iff every key fact of the reference occurs (as normalised tokens) in the prediction."""
    pred_tokens = set(normalize_answer(prediction).split())
    for fact in key_facts(reference):
        if not set(normalize_answer(fact).split()) <= pred_tokens:
            return 0.0
    return 1.0


def is_correct(prediction: str, reference: str) -> bool:
    return token_f1(prediction, reference) >= F1_THRESHOLD and key_fact_match(prediction, reference) == 1.0


# ---------------------------------------------------------------------------
# Per-item cache
# ---------------------------------------------------------------------------


def score_items(eval_set: EvalSet, retriever, generator: Generator, *, k_ctx: int = 3, top_k: int = 10) -> list[dict]:
    """Retrieval, features and both reader calls for every item, run once."""
    rows = []
    for item in eval_set.items:
        ev = extract_features(item.question, retriever, k_ctx=k_ctx, top_k=top_k)
        context = [r.chunk for r in ev.context]
        pair = read_both(generator, item.question, context)
        gold = set(item.ground_truth_chunk_ids)
        row = {"id": item.id, "category": item.category.value, "is_answerable": item.is_answerable,
               "question": item.question, "reference": item.reference_answer, "features": ev.features,
               "context_ids": [c.chunk_id for c in context]}
        for mode, ans in (("free", pair.free), ("forced", pair.forced)):
            answered = not ans.abstain
            entry = {"answer": ans.answer, "answered": answered, "grounded": ans.grounded, "reason": ans.reason,
                     "citations": ans.citations}
            if item.is_answerable and answered:
                entry.update(em=exact_match(ans.answer, item.reference_answer),
                             f1=token_f1(ans.answer, item.reference_answer),
                             key_fact=key_fact_match(ans.answer, item.reference_answer),
                             citation_precision=(len(set(ans.citations) & gold) / len(ans.citations)
                                                 if ans.citations else 0.0))
                entry["correct"] = bool(entry["f1"] >= F1_THRESHOLD and entry["key_fact"] == 1.0)
            else:
                entry["correct"] = False  # an answer to an unanswerable item, or no answer
            row[mode] = entry
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# System evaluation
# ---------------------------------------------------------------------------


def system_metrics(rows: Sequence[dict], answered: np.ndarray, reader: str) -> dict[str, float]:
    """Metrics of one system on a set of items, given which items it answered."""
    ans_items = np.array([r["is_answerable"] for r in rows])
    correct = np.array([r[reader]["correct"] for r in rows])
    n = len(rows)
    out = {"coverage": float(answered.mean()) if n else math.nan,
           "selective_risk": float(1 - correct[answered].mean()) if answered.any() else math.nan,
           "hallucination_rate": float(answered[~ans_items].mean()) if (~ans_items).any() else math.nan,
           "wrong_answer_rate": float((answered & ~correct)[ans_items].mean()) if ans_items.any() else math.nan,
           "accuracy": float(np.mean(np.where(ans_items, answered & correct, ~answered)))}
    should_abstain = ~(ans_items & correct)
    abstained = ~answered
    out["abstention_precision"] = float(should_abstain[abstained].mean()) if abstained.any() else math.nan
    out["abstention_recall"] = float(abstained[should_abstain].mean()) if should_abstain.any() else math.nan
    answered_ans = [r for r, a in zip(rows, answered) if a and r["is_answerable"]]
    for m in ("em", "f1", "key_fact", "citation_precision"):
        out[m] = float(np.mean([r[reader][m] for r in answered_ans])) if answered_ans else math.nan
    return out


def _labels(rows: Sequence[dict], reader: str) -> np.ndarray:
    """Training/calibration label for a gated system: its reader answers *and* is correct."""
    return np.array([int(r[reader]["answered"] and r[reader]["correct"]) for r in rows])


def _gate_scores(probs: np.ndarray, rows: Sequence[dict], reader: str) -> np.ndarray:
    """A reader abstention can never be 'answered', so it gets a score below any threshold."""
    return np.where([r[reader]["answered"] for r in rows], probs, -1.0)


def fit_e2e_policy(rows: Sequence[dict], eval_set: EvalSet, *, reader: str = "free", method: str = "erm",
                   alpha: float = 0.2, delta: float = 0.10, seed: int = 0) -> AbstentionPolicy:
    """Deployable gate calibrated on end-to-end correctness, using the protocol of the benchmark.

    The model is fitted on the train fold of split 0 and τ is chosen on its calibration
    fold, with ERM (no guarantee) or LTT (``P(risk ≤ α) ≥ 1 − δ``). ``reader`` is the
    reader mode the gate will sit in front of: ``"free"`` (abstain-allowed) or
    ``"forced"``.
    """
    if method not in ("erm", "ltt"):
        raise ValueError("method must be 'erm' or 'ltt'")
    train, cal, _ = stratified_splits(eval_set.items, n_splits=1, seed=seed)[0]
    tr, ca = [rows[i] for i in train], [rows[i] for i in cal]
    features = list(rows[0]["features"])
    model = LogisticCalibrator(features).fit([r["features"] for r in tr], _labels(tr, reader))
    s_ca = _gate_scores(model.predict_proba([r["features"] for r in ca]), ca, reader)
    y_ca = _labels(ca, reader)
    if method == "erm":
        th = erm_threshold(s_ca, y_ca, alpha)
    else:
        grid = coverage_grid(model.predict_proba([r["features"] for r in tr]), len(ca), alpha, delta)
        th = ltt_threshold(s_ca, y_ca, alpha, delta, grid=grid)
    meta = {**th.to_dict(), "reader": reader, "label": "end-to-end correctness (F1>=0.5 and key-fact match)",
            "guarantee": f"P(risk <= {alpha:g}) >= {1 - delta:g}" if method == "ltt" else None,
            "corpus_sha256": eval_set.corpus.sha256, "seed": seed, "n_train": len(tr), "n_cal": len(ca)}
    return AbstentionPolicy(model, th.tau, meta)


def run_e2e_benchmark(rows: Sequence[dict], eval_set: EvalSet, *, alphas: Sequence[float] = DEFAULT_ALPHAS,
                      delta: float = 0.10, n_splits: int = 200, seed: int = 0,
                      example_alpha: float | None = None) -> dict:
    example_alpha = max(alphas) if example_alpha is None else example_alpha
    features = list(rows[0]["features"])
    splits = stratified_splits(eval_set.items, n_splits=n_splits, seed=seed)
    systems: dict[str, list[dict]] = {}
    gated_outcomes: dict[str, list[dict]] = {}
    caught = Counter()
    seen_in_test = Counter()

    def record(name: str, value: dict) -> None:
        systems.setdefault(name, []).append(value)

    for train, cal, test in splits:
        tr, ca, te = ([rows[i] for i in idx] for idx in (train, cal, test))
        record("standard_rag", system_metrics(te, np.array([r["forced"]["answered"] for r in te]), "forced"))
        record("rag_self_abstain", system_metrics(te, np.array([r["free"]["answered"] for r in te]), "free"))
        for base, reader in (("selective_rag", "forced"), ("selective_rag_self", "free")):
            y_tr, y_ca = _labels(tr, reader), _labels(ca, reader)
            if y_tr.min() == y_tr.max():
                continue
            model = LogisticCalibrator(features).fit([r["features"] for r in tr], y_tr)
            p_tr = model.predict_proba([r["features"] for r in tr])
            s_ca = _gate_scores(model.predict_proba([r["features"] for r in ca]), ca, reader)
            s_te = _gate_scores(model.predict_proba([r["features"] for r in te]), te, reader)
            for a in alphas:
                for method in ("erm", "ltt"):
                    if method == "erm":
                        th = erm_threshold(s_ca, y_ca, a)
                    else:
                        th = ltt_threshold(s_ca, y_ca, a, delta, grid=coverage_grid(p_tr, len(ca), a, delta))
                    answered = s_te >= th.tau
                    name = f"{base}[{method},α={a:g}]"
                    m = system_metrics(te, answered, reader)
                    m["abstain_all"] = float(th.abstains_on_all)
                    m["valid"] = (float(m["selective_risk"] <= a + 1e-12) if answered.any() else math.nan)
                    record(name, m)
                    gated_outcomes.setdefault(name, []).append({"tau": th.tau, "reason": th.reason})
                    if base == "selective_rag" and method == "erm" and math.isclose(a, example_alpha):
                        for r, ans in zip(te, answered):
                            if not r["is_answerable"]:
                                seen_in_test[r["id"]] += 1
                                caught[r["id"]] += int(not ans)

    summary = {name: {k: _summary([v[k] for v in vals]) for k in vals[0]} for name, vals in systems.items()}
    for name, vals in systems.items():
        if "valid" in vals[0]:
            answered = [v for v in vals if v["coverage"] > 0]
            summary[name]["validity_rate"] = (float(np.mean([v["valid"] for v in answered])) if answered else None)
            # What LTT actually bounds: P(test risk > α) over all splits, where abstaining on all is never a violation.
            summary[name]["violation_rate"] = float(np.mean([v["coverage"] > 0 and not v["valid"] for v in vals]))
            summary[name]["abstain_all_rate"] = float(np.mean([v["abstain_all"] for v in vals]))
            summary[name]["abstain_reasons"] = sorted({o["reason"] for o in gated_outcomes[name] if o["reason"]})

    # per-split paired differences against standard RAG
    deltas = {}
    base_vals = systems["standard_rag"]
    for name, vals in systems.items():
        if name == "standard_rag" or len(vals) != len(base_vals):
            continue
        deltas[name] = {}
        for k in ("hallucination_rate", "selective_risk", "coverage", "accuracy"):
            d = [v[k] - b[k] for v, b in zip(vals, base_vals) if np.isfinite(v[k]) and np.isfinite(b[k])]
            deltas[name][k] = {**_summary(d), "share_negative": float(np.mean([x < 0 for x in d])) if d else None}

    by_category = {}
    for reader in ("free", "forced"):
        by_category[reader] = {}
        for cat in sorted({r["category"] for r in rows}):
            sub = [r for r in rows if r["category"] == cat]
            answered = [r for r in sub if r[reader]["answered"]]
            by_category[reader][cat] = {
                "n": len(sub), "answered": len(answered),
                "correct": sum(r[reader]["correct"] for r in sub),
                "mean_f1_answered": (float(np.mean([r[reader]["f1"] for r in answered]))
                                     if answered and sub[0]["is_answerable"] else None),
            }
    examples = []
    for r in rows:
        if not r["is_answerable"] and r["forced"]["answered"]:
            n_seen = seen_in_test[r["id"]]
            examples.append({"id": r["id"], "category": r["category"], "question": r["question"],
                             "standard_rag_answer": r["forced"]["answer"],
                             "reader_abstains": not r["free"]["answered"],
                             "gate_catch_rate": caught[r["id"]] / n_seen if n_seen else None})
    return {"systems": summary, "deltas_vs_standard_rag": deltas, "by_category": by_category,
            "hallucination_examples": examples, "example_gate": f"selective_rag[erm,α={example_alpha:g}]",
            "n_splits": n_splits}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

HEADLINE_METRICS = [("coverage", "Coverage"), ("selective_risk", "Selective risk ↓"),
                    ("hallucination_rate", "Hallucination rate ↓"), ("wrong_answer_rate", "Wrong-answer rate ↓"),
                    ("accuracy", "Accuracy ↑"), ("em", "EM"), ("f1", "Token-F1"), ("key_fact", "Key-fact match")]


def _md_safe(text: str, limit: int = 120) -> str:
    text = text.replace("`", "'").replace("|", "/").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(s: dict | None) -> str:
    if not s or s.get("mean") is None:
        return "—"
    return f"{s['mean']:.3f} <sub>[{s['p05']:.2f}, {s['p95']:.2f}]</sub>"


def render_markdown(report: dict) -> str:
    cfg, res = report["config"], report["results"]
    a = cfg["headline_alpha"]
    lines = ["# End-to-End Selective RAG Benchmark", "",
             f"- Correctness: token-F1 ≥ {F1_THRESHOLD} **and** key-fact match (all reference numbers/entities "
             "present). Unanswerable items are correct only if the system abstains.",
             f"- Protocol: {res['n_splits']} stratified 40/30/30 splits. Gated systems refit g(x) on train with "
             "end-to-end correctness, choose τ on calibration, and are scored on test. "
             "Cells: mean <sub>[5th, 95th pct]</sub>.",
             f"- Backends: embedder `{cfg['embedder']}` · reranker `{cfg['reranker_backend']}` · reader "
             f"`{cfg['generator_backend']}` · k_ctx = {cfg['k_ctx']}"]
    if cfg["embedder"].startswith("hashing"):
        lines.append("- ⚠️ Dense retrieval uses the offline **hashing** embedder.")
    if cfg["reranker_backend"] and cfg["reranker_backend"].startswith("mock"):
        lines.append("- ⚠️ The reranker is the deterministic **mock**.")
    if cfg["generator_backend"].startswith("mock"):
        lines.append("- ⚠️ The reader is the deterministic **mock extractive** reader, not an LLM"
                     + (f" (fallback: {_md_safe(cfg['generator_fallback_reason'])})"
                        if cfg["generator_fallback_reason"] else "")
                     + ". It cannot do arithmetic, so reasoning items are mostly wrong by construction.")

    headline = ["standard_rag", "rag_self_abstain", f"selective_rag[erm,α={a:g}]", f"selective_rag[ltt,α={a:g}]",
                f"selective_rag_self[erm,α={a:g}]", f"selective_rag_self[ltt,α={a:g}]"]
    lines += ["", f"## Headline comparison (α = {a:g}, δ = {cfg['delta']})", "",
              "| System | " + " | ".join(h for _, h in HEADLINE_METRICS) + " | Violation rate | Validity (answering splits) |",
              "|---" * (len(HEADLINE_METRICS) + 3) + "|"]
    for name in headline:
        s = res["systems"].get(name)
        if not s:
            continue
        validity, violation = s.get("validity_rate"), s.get("violation_rate")
        lines.append(f"| {name} | " + " | ".join(_cell(s[k]) for k, _ in HEADLINE_METRICS) + " | "
                     + ("—" if violation is None else f"{violation:.1%}") + " | "
                     + ("—" if validity is None else f"{validity:.0%}") + " |")
    lines += ["", "Selective risk = errors among answered items. Hallucination rate = share of unanswerable items "
              "that got an answer. **Violation rate** = share of *all* splits whose test selective risk exceeds α "
              "(abstaining on everything is never a violation). That is the quantity LTT bounds by δ. "
              "Validity = share of the splits that answered anything whose test risk is ≤ α; it is conditional "
              "and noisy when few splits answer. "
              f"LTT needs ≥ {cfg['ltt_min_answered'][f'{a:g}']} answered calibration items at α = {a:g}."]

    lines += ["", "## Δ vs standard RAG (per-split paired difference)", "",
              "| System | Δ hallucination rate | Δ selective risk | Δ coverage | splits with lower hallucination |",
              "|---|---|---|---|---|"]
    for name, d in res["deltas_vs_standard_rag"].items():
        if f"α={a:g}]" in name or "[" not in name:
            share = d["hallucination_rate"]["share_negative"]
            lines.append(f"| {name} | {_cell(d['hallucination_rate'])} | {_cell(d['selective_risk'])} | "
                         f"{_cell(d['coverage'])} | {'—' if share is None else f'{share:.0%}'} |")

    lines += ["", "## Full α grid (gated systems)", "",
              "| System | Coverage | Selective risk | Hallucination | Violation rate | Validity | Abstain-all |",
              "|---|---|---|---|---|---|---|"]
    for name, s in res["systems"].items():
        if "[" in name:
            v = s.get("validity_rate")
            lines.append(f"| {name} | {_cell(s['coverage'])} | {_cell(s['selective_risk'])} | "
                         f"{_cell(s['hallucination_rate'])} | {s['violation_rate']:.1%} | "
                         f"{'—' if v is None else f'{v:.0%}'} | "
                         f"{s['abstain_all_rate']:.0%} |")

    lines += ["", "## Reader outcomes by category (all items, no gate)", "",
              "| Category | n | forced: correct | free: answered | free: correct | free: mean F1 (answered) |",
              "|---|---|---|---|---|---|"]
    for cat, f in res["by_category"]["free"].items():
        fo = res["by_category"]["forced"][cat]
        f1 = "—" if f["mean_f1_answered"] is None else f"{f['mean_f1_answered']:.3f}"
        lines.append(f"| {cat} | {f['n']} | {fo['correct']} | {f['answered']} | {f['correct']} | {f1} |")

    ex = res["hallucination_examples"][:8]
    if ex:
        lines += ["", "## Hallucinations of standard RAG and who catches them", "",
                  f"Gate = `{res['example_gate']}`; catch rate = share of test appearances "
                  "where the gate abstained.", "",
                  "| Item | Category | Question | Standard RAG answer | Reader abstains | Gate catch rate |",
                  "|---|---|---|---|---|---|"]
        for e in ex:
            rate = "—" if e["gate_catch_rate"] is None else f"{e['gate_catch_rate']:.0%}"
            lines.append(f"| {e['id']} | {e['category']} | {e['question'][:70]} | {e['standard_rag_answer'][:45]} | "
                         f"{'yes' if e['reader_abstains'] else 'no'} | {rate} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.e2e_eval", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="end-to-end standard vs selective RAG benchmark")
    run.add_argument("--eval-set", type=Path, default=Path("data/eval/eval_set.json"))
    run.add_argument("--corpus", type=Path, default=None)
    run.add_argument("--generator", choices=("anthropic", "openai", "mock"), default="anthropic",
                     help="reader backend; LLM backends fall back to the mock (with a warning) unless "
                          "--strict-generator")
    run.add_argument("--generator-model", default=None, help="default: claude-opus-5-5 (anthropic)")
    run.add_argument("--base-url", default=None, help="OpenAI-compatible server URL")
    run.add_argument("--strict-generator", action="store_true")
    run.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="generation cache (JSONL)")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--k-ctx", type=int, default=3)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    run.add_argument("--headline-alpha", type=float, default=0.2)
    run.add_argument("--delta", type=float, default=0.10)
    run.add_argument("--n-splits", type=int, default=200)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    add_pipeline_args(run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # per-request INFO lines drown the real output
    if args.headline_alpha not in args.alphas:
        args.alphas = sorted(set(args.alphas) | {args.headline_alpha})
    eval_set = EvalSet.load(args.eval_set)
    chunks = load_chunks_jsonl(args.corpus or Path(eval_set.corpus.path))
    if corpus_fingerprint(chunks) != eval_set.corpus.sha256:
        logger.warning("corpus differs from the one the eval set was built on")
    first, reranker = build_pipeline(chunks, args)
    retriever = RerankingRetriever(first, reranker, retrieve_k=args.retrieve_k) if reranker is not None else first
    try:
        generator = Generator(args.generator, args.generator_model, base_url=args.base_url,
                              allow_fallback=not args.strict_generator,
                              cache_path=None if args.no_cache else args.cache)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        rows = score_items(eval_set, retriever, generator, k_ctx=args.k_ctx, top_k=args.top_k)
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nSet ANTHROPIC_API_KEY / OPENAI_API_KEY (or `ant auth login`), or drop "
                         "--strict-generator to use the offline mock reader.") from exc
    results = run_e2e_benchmark(rows, eval_set, alphas=args.alphas, delta=args.delta, n_splits=args.n_splits,
                                seed=args.seed, example_alpha=args.headline_alpha)
    report = {
        "config": {
            "embedder": getattr(first.embedder, "name", "unknown"),
            "reranker_backend": reranker.backend if reranker else None,
            "generator_backend": generator.backend, "generator_fallback_reason": generator.fallback_reason,
            "cache_hits": generator.cache_hits, "k_ctx": args.k_ctx, "alphas": args.alphas,
            "headline_alpha": args.headline_alpha, "delta": args.delta, "seed": args.seed,
            "f1_threshold": F1_THRESHOLD,
            "ltt_min_answered": {f"{a:g}": min_certifiable_n(a, args.delta) for a in args.alphas},
            "eval_set_generator": eval_set.generator, "corpus": eval_set.corpus.model_dump(),
        },
        "results": results,
        "items": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"wrote {args.output} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
