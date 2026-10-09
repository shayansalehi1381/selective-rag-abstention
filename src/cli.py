"""Interactive command-line interface for the selective RAG pipeline.

::

    python -m src.cli ask "For how many epochs is SefiLens trained?"
    python -m src.cli ask "..." --json            # machine-readable output
    python -m src.cli chat                         # interactive loop (:help for commands)

Each answer shows three panels: (1) retrieval and rerank scores, (2) the abstention
gate's verdict ``g(x)`` vs ``τ``, with the signals that drove it, and (3) the cited
answer, or the stage that abstained.

The gate comes from ``--policy PATH`` (a saved ``AbstentionPolicy``). If no policy is
given, it is calibrated at startup on the committed eval set with
``--calibrate {erm,ltt}`` (default ERM at α = 0.2, which has **no** statistical
guarantee; LTT has one). ``--no-gate`` runs standard RAG.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np

from src.abstention import AbstentionPolicy
from src.pipeline import PipelineResponse, SelectiveRAGPipeline

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET = Path("data/eval/eval_set.json")
OFFLINE_BACKENDS = ("hashing", "mock")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build(args: argparse.Namespace) -> tuple[SelectiveRAGPipeline, list[str]]:
    """Build the pipeline and its gate. Returns ``(pipeline, warnings)``."""
    from eval.e2e_eval import fit_e2e_policy, score_items
    from eval.evaluate import build_pipeline
    from eval.schemas import EvalSet, corpus_fingerprint
    from src.data_loader import load_chunks_jsonl
    from src.generator import Generator
    from src.reranker import RerankingRetriever

    warnings: list[str] = []
    eval_set = EvalSet.load(args.eval_set) if args.eval_set.exists() else None
    corpus_path = args.corpus or (Path(eval_set.corpus.path) if eval_set else None)
    if corpus_path is None:
        raise SystemExit("no corpus: pass --corpus PATH (a chunks.jsonl) or keep data/eval/eval_set.json")
    chunks = load_chunks_jsonl(corpus_path)
    fingerprint = corpus_fingerprint(chunks)

    def make_retriever(cs):
        first, reranker = build_pipeline(cs, args)
        return RerankingRetriever(first, reranker, retrieve_k=args.retrieve_k) if reranker is not None else first

    retriever = make_retriever(chunks)
    try:
        generator = Generator(args.generator, args.generator_model, base_url=args.base_url,
                              allow_fallback=not args.strict_generator,
                              cache_path=None if args.no_cache else args.cache)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    policy = None
    if args.no_gate:
        pass
    elif args.policy:
        policy = AbstentionPolicy.load(args.policy)
        if policy.meta.get("corpus_sha256") not in (None, fingerprint):
            warnings.append("the policy was calibrated on a different corpus: its threshold τ does not transfer")
    else:
        if eval_set is None:
            raise SystemExit("cannot calibrate the gate without an eval set: pass --policy PATH or --no-gate")
        if fingerprint == eval_set.corpus.sha256:
            cal_retriever = retriever
        else:
            warnings.append(f"the gate is calibrated on {eval_set.corpus.path}, not on the queried corpus; "
                            "its threshold τ does not transfer, so treat verdicts as indicative only")
            cal_retriever = make_retriever(load_chunks_jsonl(eval_set.corpus.path))
        logger.info("calibrating the gate (%s, α=%g) on %d eval items", args.calibrate, args.alpha,
                    len(eval_set.items))
        rows = score_items(eval_set, cal_retriever, generator, k_ctx=args.k_ctx)
        policy = fit_e2e_policy(rows, eval_set, reader="forced" if args.forced_reader else "free",
                                method=args.calibrate, alpha=args.alpha, delta=args.delta)
    if policy is not None and math.isinf(policy.tau) and policy.tau > 0:
        warnings.append("τ = ∞: no threshold could be certified on the calibration data, so every query "
                        "will be abstained. Use --calibrate erm, a larger α, or more calibration data")
    pipeline = SelectiveRAGPipeline(retriever, generator, policy, k_ctx=args.k_ctx, top_k=max(args.show, args.k_ctx),
                                    forced=args.forced_reader)
    return pipeline, warnings


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class Style:
    def __init__(self, color: bool) -> None:
        self.color = color

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)


def gate_description(policy: AbstentionPolicy | None) -> str:
    if policy is None:
        return "no gate (standard RAG)"
    meta = policy.meta
    method, alpha = meta.get("method"), meta.get("alpha")
    if method == "ltt":
        return f"LTT, α={alpha:g}, δ={meta.get('delta'):g}: P(risk ≤ α) ≥ {1 - meta.get('delta'):g}"
    if method == "erm":
        return f"ERM, α={alpha:g}: empirical threshold, no statistical guarantee"
    return "loaded policy"


def top_signals(policy: AbstentionPolicy, features: dict[str, float], n: int = 3) -> list[tuple[str, float, float]]:
    """Features with the largest contribution ``coef · z`` to the gate's logit."""
    cal = policy.calibrator
    x = np.array([features[f] for f in cal.features])
    z = (x - cal.mean) / cal.std
    contrib = cal.coef * z
    order = np.argsort(-np.abs(contrib), kind="stable")[:n]
    return [(cal.features[i], float(z[i]), float(contrib[i])) for i in order]


def to_dict(resp: PipelineResponse, policy: AbstentionPolicy | None) -> dict:
    return {
        "query": resp.query,
        "answer": resp.answer,
        "abstained": resp.abstained,
        "abstention_stage": resp.abstention_stage,
        "gate": None if policy is None else {
            "confidence": resp.confidence, "tau": policy.tau if math.isfinite(policy.tau) else str(policy.tau),
            "method": policy.meta.get("method"), "alpha": policy.meta.get("alpha"),
            "guarantee": policy.meta.get("guarantee")},
        "reader_reason": resp.generation.reason if resp.generation else None,
        "citations": resp.citations,
        "evidence_quotes": resp.evidence_quotes,
        "retrieval": [{"rank": r.rank, "chunk_id": r.chunk.chunk_id, "title": r.chunk.title,
                       "rerank_logit": r.rerank_score, "rerank_probability": r.rerank_probability,
                       "first_stage_rank": r.first_stage_rank, "first_stage_score": r.first_stage_score,
                       "bm25_rank": r.bm25_rank, "dense_rank": r.dense_rank, "score": r.score}
                      for r in resp.ranked],
        "features": resp.features,
        "backends": resp.backends,
        "latency_ms": resp.latency_ms,
    }


def _fmt(v: float | int | None, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def render(resp: PipelineResponse, policy: AbstentionPolicy | None, *, style: Style, show: int = 5,
           scores: bool = True) -> str:
    s = style
    lines = [s.bold("Query: ") + resp.query]
    backends = resp.backends
    offline = any(str(v).startswith(OFFLINE_BACKENDS) for v in backends.values() if v)
    lines.append(s.dim("Backends: " + " · ".join(f"{k} {v}" for k, v in backends.items() if v))
                 + (" " + s.yellow("⚠ offline stand-ins") if offline else ""))
    if scores:
        lines += ["", s.bold(f"① Retrieval + rerank (top-{min(show, len(resp.ranked))})")]
        reranked = any(r.rerank_score is not None for r in resp.ranked)
        header = (f"  {'#':>2}  {'chunk':<20} {'rerank':>7} {'P(rel)':>6} {'1st-stage':>9} {'BM25#':>5} "
                  f"{'dense#':>6} {'1st#':>4}") if reranked else f"  {'#':>2}  {'chunk':<20} {'RRF':>8} {'BM25#':>5} {'dense#':>6}"
        lines.append(s.dim(header))
        for r in resp.ranked[:show]:
            if reranked:
                lines.append(f"  {r.rank:>2}  {r.chunk.chunk_id:<20} {_fmt(r.rerank_score, '7.2f')} "
                             f"{_fmt(r.rerank_probability, '6.2f')} {_fmt(r.first_stage_score, '9.4f')} "
                             f"{_fmt(r.bm25_rank, '>5')} {_fmt(r.dense_rank, '>6')} {_fmt(r.first_stage_rank, '>4')}")
            else:
                lines.append(f"  {r.rank:>2}  {r.chunk.chunk_id:<20} {_fmt(r.score, '8.4f')} "
                             f"{_fmt(r.bm25_rank, '>5')} {_fmt(r.dense_rank, '>6')}")
        if not resp.ranked:
            lines.append("  (no candidates)")

    lines += ["", s.bold("② Abstention gate  ") + s.dim(f"({gate_description(policy)})")]
    if policy is None:
        lines.append("   gate disabled: the reader always runs")
    else:
        passed = resp.abstention_stage != "policy"
        verdict = s.green("PASS") if passed else s.red("ABSTAIN (stage: policy)")
        op = "≥" if passed else "<"
        tau = "∞" if math.isinf(policy.tau) else f"{policy.tau:.3f}"
        lines.append(f"   g(x) = {resp.confidence:.3f}  {op}  τ = {tau}   → {verdict}")
        if scores and resp.features:
            sig = ", ".join(f"{name} z={z:+.1f} {'↑' if c >= 0 else '↓'}" for name, z, c in
                            top_signals(policy, resp.features))
            lines.append(s.dim(f"   strongest signals (z-score vs train; ↑ raises / ↓ lowers confidence): {sig}"))

    lines += ["", s.bold("③ Answer")]
    if not resp.abstained:
        lines.append("   " + s.green(f'"{resp.answer}"'))
        for cid, quote in zip(resp.citations, resp.evidence_quotes + [""] * len(resp.citations)):
            lines.append(f"   cited: {cid}" + (f'  ❝{quote}❞' if quote else ""))
        if resp.generation and not resp.generation.grounded:
            lines.append(s.yellow("   ⚠ ungrounded: the forced reader could not support this answer with a quote"))
    else:
        lines.append("   " + s.yellow(resp.answer))
        if resp.abstention_stage == "generator":
            lines.append(s.dim(f"   (the reader declined: {resp.generation.reason if resp.generation else '?'})"))
    lat = resp.latency_ms
    lines.append(s.dim(f"Latency: retrieve {lat.get('retrieve', 0):.1f} ms · generate {lat.get('generate', 0):.1f} ms"
                       f" · total {lat.get('total', 0):.1f} ms"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

CHAT_HELP = """Commands:
  :k N           context size passed to the reader (current: {k})
  :gate on|off   enable / disable the abstention gate (current: {gate})
  :scores on|off show the retrieval / gate score panels (current: {scores})
  :json          toggle JSON output (current: {json})
  :help          this help
  :quit          exit (also Ctrl-D)
Anything else is treated as a question."""


def run_chat(pipeline: SelectiveRAGPipeline, *, style: Style, show: int, stdin: TextIO, stdout: TextIO) -> int:
    policy = pipeline.policy
    state = {"gate": policy is not None, "scores": True, "json": False}
    print(style.bold("Selective RAG chat") + " — ask a question, or :help", file=stdout)
    while True:
        print("> ", end="", file=stdout, flush=True)
        line = stdin.readline()
        if not line:
            print(file=stdout)
            return 0
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            cmd, _, arg = line[1:].partition(" ")
            arg = arg.strip()
            if cmd in ("quit", "q", "exit"):
                return 0
            if cmd == "help":
                print(CHAT_HELP.format(k=pipeline.k_ctx, gate="on" if state["gate"] else "off",
                                       scores="on" if state["scores"] else "off",
                                       json="on" if state["json"] else "off"), file=stdout)
            elif cmd == "k" and arg.isdigit() and int(arg) > 0:
                pipeline.k_ctx = int(arg)
                pipeline.top_k = max(pipeline.top_k, pipeline.k_ctx)
                print(f"context size = {pipeline.k_ctx}", file=stdout)
            elif cmd == "gate" and arg in ("on", "off"):
                if arg == "on" and policy is None:
                    print("no policy loaded (started with --no-gate)", file=stdout)
                else:
                    state["gate"] = arg == "on"
                    pipeline.policy = policy if state["gate"] else None
                    print(f"gate {arg}", file=stdout)
            elif cmd == "scores" and arg in ("on", "off"):
                state["scores"] = arg == "on"
            elif cmd == "json":
                state["json"] = not state["json"]
            else:
                print(f"unknown command {line!r}; try :help", file=stdout)
            continue
        resp = pipeline.answer(line)
        if state["json"]:
            print(json.dumps(to_dict(resp, pipeline.policy), indent=2, default=str), file=stdout)
        else:
            print(render(resp, pipeline.policy, style=style, show=show, scores=state["scores"]), file=stdout)


def _build_parser() -> argparse.ArgumentParser:
    from eval.evaluate import add_pipeline_args

    parser = argparse.ArgumentParser(prog="selective-rag", description="Selective RAG with calibrated abstention.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("ask", "answer one question"), ("chat", "interactive question loop")):
        p = sub.add_parser(name, help=help_text)
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--json", action="store_true", help="print machine-readable JSON")
        p.add_argument("--corpus", type=Path, default=None, help="chunks.jsonl to query (default: the eval corpus)")
        p.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="used to calibrate the gate")
        gate = p.add_mutually_exclusive_group()
        gate.add_argument("--policy", type=Path, default=None, help="saved AbstentionPolicy JSON")
        gate.add_argument("--no-gate", action="store_true", help="standard RAG: never abstain before reading")
        p.add_argument("--calibrate", choices=("erm", "ltt"), default="erm",
                       help="how to choose τ when no --policy is given (default erm: no guarantee)")
        p.add_argument("--alpha", type=float, default=0.2, help="target selective risk")
        p.add_argument("--delta", type=float, default=0.1, help="LTT failure probability")
        p.add_argument("--forced-reader", action="store_true", help="reader may not abstain (standard-RAG reader)")
        p.add_argument("--k-ctx", type=int, default=3)
        p.add_argument("--show", type=int, default=5, help="rows in the retrieval panel")
        p.add_argument("--no-color", action="store_true")
        p.add_argument("--generator", choices=("anthropic", "openai", "mock"), default="anthropic")
        p.add_argument("--generator-model", default=None)
        p.add_argument("--base-url", default=None)
        p.add_argument("--strict-generator", action="store_true")
        p.add_argument("--cache", type=Path, default=Path("data/eval/generation_cache.jsonl"),
                       help="reader cache (JSONL); avoids repeat LLM calls for the gate calibration")
        p.add_argument("--no-cache", action="store_true")
        add_pipeline_args(p)
    return parser


def main(argv: Sequence[str] | None = None, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    pipeline, warnings = build(args)
    style = Style(color=not args.no_color and getattr(stdout, "isatty", lambda: False)())
    for w in warnings:
        print(style.yellow(f"⚠ {w}"), file=sys.stderr)
    if args.command == "ask":
        resp = pipeline.answer(args.question)
        if args.json:
            payload = to_dict(resp, pipeline.policy)
            payload["warnings"] = warnings
            print(json.dumps(payload, indent=2, default=str), file=stdout)
        else:
            print(render(resp, pipeline.policy, style=style, show=args.show), file=stdout)
        return 0
    return run_chat(pipeline, style=style, show=args.show, stdin=stdin, stdout=stdout)


if __name__ == "__main__":
    raise SystemExit(main())
