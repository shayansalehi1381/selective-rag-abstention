# Selective RAG with Calibrated Abstention

[![CI](https://github.com/shayansalehi1381/selective-rag-abstention/actions/workflows/ci.yml/badge.svg)](https://github.com/shayansalehi1381/selective-rag-abstention/actions/workflows/ci.yml)
![tests](https://img.shields.io/badge/tests-357%20passed%20offline-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

A retrieval-augmented QA system that **answers only when its evidence supports an answer**,
and otherwise abstains. The decision to answer is a calibrated threshold on a learned
confidence score. The threshold can be chosen with **Learn-then-Test (LTT)**, which gives a
finite-sample, distribution-free guarantee on the error rate among answered questions:
`P(selective risk ≤ α) ≥ 1 − δ`.

> **TL;DR (α = 0.2).** Standard RAG answers every unanswerable question: hallucination rate
> **1.00**, selective risk 0.71. Putting a calibrated gate in front of the reader cuts
> hallucination to **0.04–0.05** and selective risk to **0.17–0.18** (mock reader; with a local 7B LLM
> reader hallucination falls to 0.003 but selective risk stays at 0.39), with lower hallucination
> in 100% of the 200 evaluation splits. The empirical (ERM) threshold breaks its α target in
> 34–54% of splits. LTT holds its guarantee (violation rate **≤ 2%**, well under δ = 10%); with
> real retrieval models it answers 33% of questions at 0.09 test risk, but with 30 calibration
> items it cannot certify α ≤ 0.1, and it says so.
>
> **What was run with which models.** Retrieval and abstention were benchmarked both offline
> (hashing embedder, mock reranker) and with real models (bge-small-en-v1.5 and an MS MARCO
> cross-encoder). The end-to-end benchmark used the real retriever and reranker but the **mock
> extractive reader** in one table and a **local 7B LLM reader (Qwen2.5 via Ollama)** in another. With either
> reader, LTT could not certify a useful threshold at α = 0.2 on 30 calibration items; a stronger reader (or
> more calibration data) is untested. See
> [Reproducibility](#reproducibility).

---

## Contents
- [Motivation](#motivation)
- [Method](#method)
- [Components](#components)
- [Benchmark](#benchmark)
- [Findings and limitations](#findings-and-limitations)
- [Reproducibility](#reproducibility)
- [Interactive CLI](#interactive-cli)
- [Serving (REST API)](#serving-rest-api)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [References](#references)

## Motivation

A standard RAG pipeline always produces an answer. When the corpus doesn't contain the answer,
or the retriever misses it, the generator still writes something fluent and wrong. Three easy
fixes fail in instructive ways:

1. **"Let the LLM say *I don't know*."** Self-abstention helps (hallucination 1.00 → 0.27 here),
   but the generator's own judgement is uncalibrated, and it still answers about a quarter of
   unanswerable questions.
2. **"Threshold a retrieval score."** A single signal ranks safe vs unsafe queries poorly
   (AUROC 0.69–0.84 across signals), and a threshold picked by eye has no meaning on new data.
3. **"Tune the threshold on held-out data (ERM)."** This is the most common practice. On our
   benchmark, the ERM threshold's **test** risk exceeds α in **34–54% of splits**, so the
   promised error rate is a coin flip.

This project treats abstention as **selective prediction with risk control**:
- learn a confidence `g(x)` from many pipeline signals;
- choose the threshold `τ` so that the error rate among answered questions is *provably* bounded;
- measure the coverage that guarantee costs, and report it honestly.

## Method

### Architecture: two-stage abstention

```mermaid
flowchart LR
    Q([query]) --> R["Hybrid retrieval<br/>BM25 + FAISS → RRF (top-50)"]
    R --> X["Cross-encoder rerank<br/>(top-10)"]
    X --> F["Abstention features<br/>CE logit / margin / entropy,<br/>RRF, BM25, dense, agreement"]
    F --> G{"Gate<br/>g(x) ≥ τ ?"}
    G -- no --> A1(["ABSTAIN<br/>stage: policy<br/>(reader never called)"])
    G -- yes --> L["Reader on top-3 context<br/>(Claude / OpenAI-compatible / mock)"]
    L --> V{"Grounded?<br/>citations ∈ context,<br/>quotes verbatim"}
    V -- "no / reader declines" --> A2(["ABSTAIN<br/>stage: generator"])
    V -- yes --> ANS(["ANSWER<br/>+ chunk citations<br/>+ verbatim quotes"])
```

<details><summary>ASCII version</summary>

```
query ─► HybridRetriever (BM25 + FAISS, RRF, top-50) ─► CrossEncoder rerank (top-10)
      ─► 9 abstention features ─► g(x) ≥ τ ? ── no ──► ABSTAIN  (stage = policy, no LLM call)
                                              │
                                              yes
                                              ▼
                     Reader(top-3 context) ─► validate citations + verbatim quotes
                                              │
                         declines / invalid ──┴──► ABSTAIN  (stage = generator)
                                              │
                                              ▼
                                   ANSWER + citations + quotes
```
</details>

### Formulation

For a query `x`, let `y(x) = 1` if answering is *safe*: the question is answerable **and** the
pipeline's answer is correct (token-F1 ≥ 0.5 **and** every number and entity in the reference
appears in the answer). The policy answers iff `g(x) ≥ τ`.

| Quantity | Definition |
|---|---|
| Coverage | `φ(τ) = P(g(x) ≥ τ)` |
| Selective risk | `R(τ) = P(y = 0 \| g(x) ≥ τ)`: errors among answered questions |
| Goal | maximise `φ(τ)` subject to `R(τ) ≤ α` |

**Confidence model.** L2-regularised logistic regression (numpy, IRLS) on 9 features:
- cross-encoder top-1 logit, top-1 minus top-2 margin, and softmax entropy;
- RRF top-1 and margin;
- BM25 and dense top-1;
- BM25-vs-dense rank agreement;
- top-5 overlap between BM25 and dense.

Single signals are Platt-scaled baselines.

**Choosing τ on a calibration set.** Candidate thresholds are ordered from strict to permissive.
For each threshold λ_j, let `n_j` be the number of calibration items answered and `e_j` the
number of errors among them.
- **ERM:** take the largest coverage with `e_j / n_j ≤ α`. There is no guarantee.
- **Learn-then-Test** (Angelopoulos et al., 2021):
  - Test `H_j : R(λ_j) > α` with the exact binomial p-value `p_j = P(Bin(n_j, α) ≤ e_j)`.
  - Use **fixed-sequence testing** on a coverage grid fixed on the *train* split: reject while
    `p_j ≤ δ`, and stop at the first failure.
  - The result satisfies `P(R(τ̂) ≤ α) ≥ 1 − δ`.
  - **Price:** even with zero errors you need `n ≥ ⌈log δ / log(1 − α)⌉` answered calibration
    items. That's 11 at α = 0.2 and 22 at α = 0.1 (δ = 0.1). If they're not available, LTT
    abstains on everything and reports why.

**Protocol.**
- 200 stratified train / calibration / test splits (40/30/30).
- `g` is fitted on train, `τ` chosen on calibration, and everything is scored on test.
- Reader outputs are computed once per item and cached, so a real-LLM run costs about 200 calls.

## Components

| Phase | Module | What it does |
|---|---|---|
| 1 | [`src/data_loader.py`](src/data_loader.py) | arXiv downloader (idempotent, rate-limited), pypdf extraction, recursive 512/64 chunking with exact character offsets and page provenance |
| 1 | [`src/retriever.py`](src/retriever.py) | BM25 (overlap-filtered) + FAISS `IndexFlatIP` (bge-small) fused by Reciprocal Rank Fusion; per-retriever ranks kept as features |
| 2 | [`eval/synthetic_gen.py`](eval/synthetic_gen.py), [`eval/schemas.py`](eval/schemas.py) | 100-item ground truth: 70 answerable (50 factoid, 20 two-fact reasoning) + 30 unanswerable (out-of-domain, hallucinated entity, false premise). Strict Pydantic schemas. Deterministic offline engine and an LLM engine with an adversarial judge |
| 3 | [`src/reranker.py`](src/reranker.py) | Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`, raw logits) with a logged offline fallback |
| 4 | [`src/abstention.py`](src/abstention.py), [`eval/abstention_eval.py`](eval/abstention_eval.py) | Features, logistic calibrator, ERM / LTT thresholds, JSON-persisted `AbstentionPolicy`; AUROC / AURC / ECE / risk-coverage benchmark |
| 5 | [`src/generator.py`](src/generator.py), [`src/pipeline.py`](src/pipeline.py), [`eval/e2e_eval.py`](eval/e2e_eval.py) | Citation-grounded reader (validated citations and verbatim quotes, retry, abstention), end-to-end pipeline, standard vs selective RAG benchmark |
| 6 | [`src/cli.py`](src/cli.py) | Interactive CLI: scores, gate verdict, cited answer |
| 7 | [`src/server.py`](src/server.py) | FastAPI service: `/v1/query`, `/v1/calibrate` (admin), `/v1/info`, `/health`; thread-safe, one worker per process; Docker image and CI |

Every model-backed component has a deterministic offline stand-in. That keeps the whole system
testable without network access or model weights. A stand-in is only ever used through an
explicit, **logged** fallback, and it is recorded in every report. Each `--strict-*` flag
disables its fallback.

## Benchmark

**Eval set.** 100 items over a 144-chunk synthetic corpus of 16 papers. Every answer is linked to
the exact chunk(s) that state it. Dev-split and ablation numbers act as in-paper hard negatives.
The set is byte-reproducible from a seed (`data/eval/eval_set.json`).

All tables below were produced by the scripts in this repository on the committed eval set,
with **hashing embedder + mock reranker + mock extractive reader**. Brackets give the 5th–95th
percentile over 200 splits.

### 1. Retrieval (answerable items, n = 70)

Offline stand-ins (hashing embedder, mock reranker; what CI reproduces byte-for-byte):

| Retriever | Hit@1 | Hit@5 | MRR@10 | Recall@10 | Top-1 AUROC (answerable vs not) |
|---|---|---|---|---|---|
| Dense (FAISS) | 0.714 | **1.000** | 0.843 | **1.000** | 0.667 |
| Sparse (BM25) | **0.914** | 0.943 | **0.931** | 0.971 | 0.651 |
| Hybrid (RRF, k = 60) | 0.871 | 0.986 | 0.919 | **1.000** | **0.777** |
| Hybrid + cross-encoder (mock) | 0.871 | 0.943 | 0.909 | 0.957 | 0.689 |

The mock reranker doesn't help. All of its misses are vocabulary mismatch (e.g. *"backbone … based
on"* vs *"built on top of … encoder"*), which a lexical heuristic cannot bridge by construction.
Its weights were not tuned on the eval set to hide this.

#### Real models (bge-small-en-v1.5 + `cross-encoder/ms-marco-MiniLM-L6-v2`)

Measured on the author's Windows machine with `python -m eval.evaluate run --baseline --embedder bge
--reranker cross-encoder --strict-reranker` (same eval set, n = 70, so confidence intervals are wide):

| Retriever | Hit@1 | MRR@10 | Top-1 AUROC (answerable vs not) |
|---|---|---|---|
| Dense (bge-small) | 0.900 | 0.950 | **0.916** [0.858, 0.966] |
| Sparse (BM25) | 0.914 | 0.931 | 0.651 |
| Hybrid (RRF, k = 60) | **0.929** | **0.964** | 0.739 |
| Hybrid + cross-encoder (MS MARCO MiniLM) | 0.871 | 0.926 | 0.816 |

* A real embedder fixes dense retrieval (Hit@1 0.714 → 0.900) and makes its top-1 score the strongest
  answerable-vs-not signal (AUROC 0.667 → 0.916). The RRF score is rank-based, so as a confidence signal it is weak.
* The off-the-shelf MS MARCO cross-encoder does **not** improve on hybrid retrieval here: MRR −0.038 and
  Recall@3 −0.043 (both significant under the paired bootstrap), Hit@1 −0.057 (not significant). Its logit is a
  better abstention signal than RRF (AUROC 0.816 vs 0.739, difference not significant).
* Plausible cause (not tested): the cross-encoder is trained on web passages, while this corpus is numeric and
  entity-heavy with in-paper hard negatives, and hybrid is already near ceiling. With n = 70 and several
  comparisons, treat these as indications, not conclusions.

### 2. Abstention signals (label: answerable and gold evidence in the top-3; 66/100 safe)

Offline stand-ins (hashing embedder, mock reranker); the figures below are from this run. Real-model results follow.

| Confidence model | AUROC ↑ | AURC ↓ | Brier ↓ | ECE (pooled) ↓ |
|---|---|---|---|---|
| **Combined (9 features)** | **0.854** [0.75, 0.95] | 0.146 | **0.135** | 0.046 |
| Platt: CE margin | 0.837 | **0.143** | 0.165 | **0.022** |
| Platt: RRF top-1 | 0.831 | 0.177 | 0.135 | 0.032 |
| Platt: dense top-1 | 0.728 | 0.202 | 0.199 | 0.094 |
| Platt: CE top-1 | 0.689 | 0.277 | 0.188 | 0.127 |

| α | Method | Test coverage | Test selective risk | **Violation rate** (LTT bounds it by δ = 0.1) |
|---|---|---|---|---|
| 0.1 | ERM | 0.435 | 0.106 | **53.5%** |
| 0.1 | LTT | 0.000 (needs ≥ 22 answered) | — | 0.0% |
| 0.2 | ERM | 0.740 | 0.178 | **34.0%** |
| 0.2 | LTT | 0.108 | 0.133 | **2.0%** |

<p align="center">
  <img src="docs/figures/risk_coverage.png" width="58%" alt="Risk-coverage curves of each confidence model, pooled over test folds"/>
  <img src="docs/figures/reliability.png" width="38%" alt="Reliability diagram of the combined confidence model"/>
</p>

#### Real models (bge-small + MS MARCO cross-encoder; 68/100 safe, 200 splits, author's Windows run)

| Confidence model | AUROC ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
|---|---|---|---|---|
| Combined (9 features) | 0.911 [0.81, 0.98] | 0.097 | 0.114 | 0.142 |
| **Platt: dense top-1** | **0.933** [0.87, 0.99] | **0.087** | **0.104** | 0.133 |
| Platt: CE margin | 0.857 | 0.119 | 0.156 | 0.181 |
| Platt: CE top-1 | 0.831 | 0.136 | 0.142 | 0.144 |
| Platt: RRF top-1 | 0.764 | 0.197 | 0.164 | **0.070** |

| α | Method | Test coverage | Test selective risk | **Violation rate** |
|---|---|---|---|---|
| 0.05 | ERM | 0.461 | 0.057 | **49.5%** |
| 0.05 | LTT | 0.000 (needs ≥ 45 answered) | — | 0.0% |
| 0.1 | ERM | 0.637 | 0.096 | **46.5%** |
| 0.1 | LTT | 0.000 (needs ≥ 22 answered) | — | 0.0% |
| 0.2 | ERM | 0.790 | 0.174 | **34.5%** |
| 0.2 | LTT | 0.326 [0.00, 0.77] | 0.091 | **1.5%** (abstains on everything in 44% of splits) |

* With real models, the abstention signal improves a lot: AUROC 0.854 → 0.911 for the combined model, and the single
  feature **dense top-1 (AUROC 0.933) is at least as good as the 9-feature model** (0.911). The intervals overlap
  heavily, so the data cannot separate them, and the logistic model with 9 features on 40 training items probably
  overfits. A one-feature Platt model is the simpler choice here.
* The combined model is *less* well calibrated than offline (ECE 0.142 vs 0.046), again consistent with overfitting
  on small training folds. Calibration is a separate property from ranking quality.
* LTT at α = 0.2 now certifies something real: coverage 0.326 (versus 0.108 offline) at test risk 0.091 and a 1.5%
  violation rate, under δ = 0.1. ERM answers more (0.790) but exceeds the target in about 35% of splits.
* At α ≤ 0.1, 30 calibration items are still too few to certify anything, so LTT abstains on everything.
* The only two retrieval misses (`eval_031`, `eval_085`) are noted in the report; with 2 of 100 items the "unsafe
  because retrieval failed" class is too small to analyse.

### 3. End to end: standard RAG vs selective RAG (α = 0.2, δ = 0.1)

| System | Coverage | Selective risk ↓ | Hallucination ↓ | Wrong-answer ↓ | Accuracy ↑ | EM | F1 | Key-fact | Violation |
|---|---|---|---|---|---|---|---|---|---|
| Standard RAG (always answers) | 1.000 | 0.711 | 1.000 | 0.587 | 0.289 | 0.126 | 0.458 | 0.503 | — |
| RAG + reader self-abstention | 0.635 | 0.564 | 0.269 | 0.396 | 0.496 | 0.160 | 0.523 | 0.597 | — |
| **Selective RAG (ERM gate)** | 0.283 | **0.173** | **0.049** | **0.064** | 0.509 | 0.341 | 0.701 | 0.887 | 41.0% |
| Selective RAG (ERM) + self-abstention | 0.302 | 0.187 | 0.048 | 0.075 | **0.521** | 0.320 | 0.686 | 0.877 | 42.5% |
| Selective RAG (LTT gate) | 0.000 | — | 0.000 | 0.000 | 0.300 | — | — | — | **0.0%** |

- *Hallucination* = share of **unanswerable** items that got an answer.
- *Accuracy* counts a correct abstention on an unanswerable item as correct.
- EM, F1 and key-fact are over answered answerable items.
- *Violation* = share of splits whose test risk exceeds α.

#### Real retriever and reranker, mock reader

Same benchmark with `--embedder bge --reranker cross-encoder --strict-reranker --generator mock` (author's Windows run,
200 splits). The reader is still the mock extractive one, so absolute accuracy stays a floor.

| System | Coverage | Selective risk ↓ | Hallucination ↓ | Accuracy ↑ | Violation |
|---|---|---|---|---|---|
| Standard RAG | 1.000 | 0.711 | 1.000 | 0.289 | — |
| RAG + reader self-abstention | 0.708 | 0.608 | 0.304 | 0.486 | — |
| Selective RAG (ERM gate) | 0.298 | 0.181 | 0.036 | 0.524 | 43.0% |
| Selective RAG (ERM) + self-abstention | 0.309 | 0.190 | 0.008 | 0.539 | 44.0% |
| Selective RAG (LTT gate) | 0.003 | 0.333 | 0.001 | 0.301 | 1.0% |
| Selective RAG (LTT) + self-abstention | 0.002 | 0.267 | 0.000 | 0.302 | 0.5% |

The picture matches the offline run. The paired per-split drop in hallucination versus standard RAG is −0.96 for the
ERM gate (100% of splits lower). ERM still violates the α = 0.2 target in about 43% of splits. LTT keeps violations at
or below 1% but answers almost nothing at α = 0.2 (coverage 0.003, abstains on everything in 99% of splits); at
α = 0.3 it reaches coverage 0.10 with 6.5% violations, and at α = 0.1 it abstains always. With a reader that is right
on roughly 30% of questions, the certificate is simply not attainable at this calibration size. The next table
repeats the run with a real (local) LLM reader.

#### Real retriever, reranker and a local LLM reader (Qwen2.5-7B via Ollama)

Same benchmark with `--generator openai --generator-model qwen2.5:7b --base-url http://localhost:11434/v1` (author's
Windows machine, 200 splits, answers cached). Only the reader changed relative to the previous table.

| System | Coverage | Selective risk ↓ | Hallucination ↓ | Accuracy ↑ | Violation |
|---|---|---|---|---|---|
| Standard RAG (forced reader) | 0.881 | 0.772 | 0.602 | 0.320 | — |
| RAG + reader self-abstention | 0.687 | 0.679 | 0.268 | 0.440 | — |
| Selective RAG (ERM gate) | 0.093 | 0.387 [0.00, 1.00] | 0.003 | 0.351 | 44.0% |
| Selective RAG (LTT gate) | 0.000 | — | 0.000 | 0.300 | 0.0% |

Reader outcomes without a gate: 23 of 50 factoid items correct, 0 of 20 reasoning items, 0 of 30 unanswerable items
answered correctly (only abstaining counts). With the gate the hallucination rate falls from 0.60 to 0.003 and the
reader is skipped for most queries, but the gate also answers only about 9% of questions, and the few it answers are
still wrong 39% of the time (the interval is [0, 1] because a split answers about three items). **LTT abstains on
everything, at every α in {0.1, 0.2, 0.3}**: the reader is right too rarely for 30 calibration items to certify any
threshold.

What this does and does not show:

* It shows that the *gate* works as a hallucination filter even with a weak reader, and that its guarantee-free ERM
  variant still violates α in 44% of splits.
* It does **not** show what LTT can do with a strong reader. A 7B local model is a weak reader on this task, and the
  correctness rule (token-F1 ≥ 0.5 and all key facts) is strict: terse answers such as `"52.8"` fail the key-fact
  check when the reference is `52.8 nDCG@10`. I did not separate wrong answers from answers that are right but
  too terse, so part of the 0.387 risk may be formatting rather than reasoning.
* The standard-RAG row answers only 88% of questions even though its reader is forced; the remaining 12% are empty or
  unparseable outputs.

## Findings and limitations

- **Gating is what removes hallucination.** Reader self-abstention alone leaves 27% of
  unanswerable questions answered. The calibrated gate leaves 5%, and it skips the LLM call
  entirely for those queries.
- **ERM does not control risk.** Its test risk exceeds α in 34–54% of splits, across both the
  retrieval-level and end-to-end labels. Any production threshold tuned on a dev set has the
  same failure mode.
- **LTT controls risk, at a price that depends on data.** Its violation rate stays at 0–2%,
  well under δ, but certifying α needs ≥ ⌈log δ / log(1 − α)⌉ error-free answered calibration
  items. With 30 calibration items and a mock reader that is right on about 30% of questions,
  that is rarely possible, so LTT abstains and says why. **More calibration data, not a
  different method, is the fix.**
- **The best abstention signal is a simple one.** With real models, the dense top-1 score alone
  (AUROC 0.933) matches the 9-feature model (0.911) within noise, and is better calibrated. A
  learned combination fitted on 40 items overfits.
- **A stronger reranker did not help retrieval.** The off-the-shelf MS MARCO cross-encoder is
  slightly *worse* than hybrid retrieval on this numeric, entity-heavy corpus (MRR −0.038). Its
  logit is still a useful abstention signal.
- **The reader limits the guarantee.** With the mock reader (0/20 reasoning items, 30/50 factoid) and with a local
  Qwen2.5-7B reader (0/20 reasoning, 23/50 factoid) LTT abstained almost always at α = 0.2, because certifying it
  needs enough error-free answered calibration items. Both are weak readers. Whether a stronger LLM (for example
  Claude) changes this is untested here: it needs an API key I did not use.
- **The gate is a good hallucination filter regardless.** Hallucination on unanswerable questions falls from 0.60–1.00
  to 0.00–0.04 with every reader tried, at the price of low coverage.
- **Small samples.** 100 items and calibration folds of about 30 make intervals wide; treat
  differences smaller than the intervals as unresolved.
- **Synthetic data.** The ground truth comes from a fact table, which makes it exact, but
  real papers are messier. The arXiv ingestion pipeline and the LLM benchmark generator are
  implemented for that next step.

## Reproducibility

> **Windows:** create the venv with `py -m venv .venv` and activate with `.venv\Scripts\activate`; use
> `set NAME=value` (cmd) or `$env:NAME="value"` (PowerShell) instead of `export`; run every command from the
> repository root. The test suite runs on `windows-latest` in CI.


### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch
pip install -r requirements.txt
```

### Offline: no network, no API keys, no model weights
```bash
python -m pytest -q                       # 357 passed, 3 skipped (opt-in integration tests)

# regenerate the eval set (byte-identical to the committed file)
python -m eval.synthetic_gen generate --offline --output data/eval/eval_set.json

# the three benchmarks (reports land in data/eval/*.md, figures in data/eval/figures/)
python -m eval.evaluate        run --baseline --embedder hashing --reranker mock
python -m eval.abstention_eval run --embedder hashing --reranker mock
python -m eval.e2e_eval        run --embedder hashing --reranker mock --generator mock
```
Every run is deterministic: reports are identical across `PYTHONHASHSEED` values.

### With real models
```bash
export ANTHROPIC_API_KEY=...          # or `ant auth login`; OpenAI-compatible: --generator openai --generator-model M [--base-url URL]

# real arXiv corpus
python -m src.data_loader download --query "retrieval augmented generation" --max-results 15
python -m src.data_loader ingest

# real embedder + cross-encoder + Claude reader; --strict-* turns every offline fallback into an error
python -m eval.e2e_eval run --embedder bge --reranker cross-encoder --strict-reranker \
                            --generator anthropic --strict-generator
RUN_INTEGRATION=1 python -m pytest -q   # also runs the live arXiv / model tests
```
The default reader model is `claude-opus-5-5`. An end-to-end run makes about 200 reader calls
(free and forced, per item). They are cached in `data/eval/generation_cache.jsonl`, so re-runs
are free.

## Interactive CLI

```bash
python -m src.cli ask "For how many epochs is SefiLens trained?" --embedder hashing
python -m src.cli chat                                      # :help, :k N, :gate on|off, :scores on|off, :json
python -m src.cli ask "..." --json                          # machine-readable
python -m src.cli ask "..." --calibrate ltt                 # guaranteed gate (may abstain on everything)
python -m src.cli ask "..." --policy policy.json            # saved AbstentionPolicy
python -m src.cli ask "..." --no-gate --forced-reader       # standard RAG, for comparison
```
Run the commands from the repository root (the default data paths are relative to it). After `pip install -e .`, the same commands are also available as `selective-rag ask …`.

```
Query: For how many epochs is SefiLens trained?
Backends: embedder hashing-256 · reranker mock-lexical-v1 · generator mock-extractive-v1 ⚠ offline stand-ins

① Retrieval + rerank (top-5)
   #  chunk                 rerank P(rel) 1st-stage BM25# dense# 1st#
   1  synth-0007::0004       -1.96   0.12    0.0328     1      1    1
   2  synth-0007::0007       -3.82   0.02    0.0315     3      4    4
   …
② Abstention gate  (ERM, α=0.2: empirical threshold, no statistical guarantee)
   g(x) = 0.939  ≥  τ = 0.822   → PASS
   strongest signals (z-score vs train; ↑ raises / ↓ lowers confidence): ce_margin z=+1.8 ↑, …

③ Answer
   "train SefiLens for 30 epochs"
   cited: synth-0007::0004  ❝We train SefiLens for 30 epochs.❞
```
Without `--policy`, the gate is calibrated at startup on the committed eval set (ERM, α = 0.2
by default). If you query a different corpus, the CLI warns that τ does not transfer.

## Serving (REST API)

```bash
pip install -e ".[serving]"
export SRAG_ADMIN_TOKEN=change-me            # enables POST /v1/calibrate (disabled when unset)
selective-rag serve --offline                # 127.0.0.1:8000, offline stand-ins; drop --offline for real models
#   or: docker build -t selective-rag . && docker run -p 8000:8000 -e SRAG_ADMIN_TOKEN=change-me selective-rag
```
The OpenAPI docs are at `http://127.0.0.1:8000/docs`. Bind to `--host 0.0.0.0` only when you
mean to expose the service.

| Endpoint | Purpose |
|---|---|
| `GET /health` | `200 {"status":"ok"}` when ready, `503 {"status":"starting"}` while models load |
| `GET /v1/info` | version, backends (and any fallback reasons), corpus fingerprint, active policy, limits |
| `POST /v1/query` | answer or abstain: stage, gate verdict, citations with verbatim quotes, reader context, scores, latency |
| `POST /v1/calibrate` | recalibrate τ with ERM or LTT on the eval set (`X-Admin-Token` required) |

```bash
curl -s -X POST localhost:8000/v1/query -H 'Content-Type: application/json' \
     -d '{"query": "For how many epochs is SefiLens trained?", "include_scores": false}'
```
```json
{
  "answer": "train SefiLens for 30 epochs",
  "abstained": false,
  "abstention_stage": null,
  "gate": {"confidence": 0.939, "tau": 0.822, "method": "erm", "alpha": 0.2,
           "guarantee": null, "statistical_guarantee": false, "passed": true},
  "citations": [{"chunk_id": "synth-0007::0004",
                 "title": "SefiLens: Uncertainty-Aware Dense Retrieval in Legal Documents",
                 "quotes": ["We train SefiLens for 30 epochs."]}],
  "context_chunk_ids": ["synth-0007::0004", "synth-0007::0007", "synth-0007::0002"],
  "latency_ms": {"retrieve": 10.3, "generate": 0.2, "total": 10.5},
  "backends": {"embedder": "hashing-256", "reranker": "mock-lexical-v1", "generator": "mock-extractive-v1"}
}
```
(abridged; `request_id` and `reader_reason` omitted.) An out-of-domain question, such as
*"Which mortgage refinancing option minimises closing fees?"*, returns
`"abstained": true, "abstention_stage": "policy"` with `g(x) = 0.015 < τ`, and the reader is never called.

```bash
curl -s -X POST localhost:8000/v1/query -H 'Content-Type: application/json' \
     -d '{"query": "...", "gate": false, "forced_reader": true}'                 # standard RAG, for comparison
curl -s -X POST localhost:8000/v1/calibrate -H 'Content-Type: application/json' \
     -H "X-Admin-Token: $SRAG_ADMIN_TOKEN" -d '{"method": "ltt", "alpha": 0.2, "delta": 0.1}'
curl -s localhost:8000/v1/info
```
A recalibration reports whether the new τ carries a guarantee. With the 100-item eval set, LTT
often certifies nothing (`"tau": "inf"`, with a `reason`). The service then abstains on
everything rather than pretend.

**Errors** share one envelope, `{"error": {"code", "message", "request_id"}}`:

| Status | When |
|---|---|
| 422 | invalid request (empty or oversized query, unknown fields, out-of-range `k_ctx` / α) |
| 401 / 403 | missing or wrong admin token, or calibration disabled |
| 409 | a calibration is already running |
| 429 | all reader slots are busy (`Retry-After`) |
| 503 | the service is not ready yet |
| 500 | internal error; no internals in the body, the traceback is logged with the request id |

Every response carries `X-Request-ID`; an incoming one is echoed back.

**Concurrency model.**
- Endpoints run in FastAPI's thread pool. The pipeline is shared but only read, and per-request
  settings (`k_ctx`, `gate`, `forced_reader`) are arguments, never shared state.
- The active policy is an immutable object behind one reference. Each request reads it once, and
  a recalibration swaps it atomically, so no request mixes two thresholds.
- A semaphore (`--max-concurrency`, default 4) caps concurrent LLM calls. Gated queries never take
  a slot.
- Run **one worker per process**: each worker holds its own models and its own τ. Scale with
  replicas.

**Measured throughput (offline stand-ins, 4 CPUs, keep-alive client, 200 requests):**

| Clients | req/s | p50 | p95 |
|---|---|---|---|
| 1 | 65 | 14.6 ms | 23.3 ms |
| 4 | 67 | 60.1 ms | 76.6 ms |
| 16 | 67 | 235.6 ms | 288.6 ms |

Throughput saturates at about 66 req/s whatever the concurrency, because the offline mock
reranker is pure Python and holds the GIL. With real backends the profile differs: torch releases
the GIL during inference, and LLM calls are I/O-bound and capped by the reader semaphore.

## Project layout
```
src/            data_loader · retriever · reranker · abstention · llm · generator · pipeline · cli · server
eval/           schemas · synthetic_gen · evaluate · abstention_eval · e2e_eval
tests/          one test module per component (357 offline tests)
data/eval/      eval_set.json + synthetic_corpus.jsonl (committed); benchmark reports (generated)
docs/figures/   README figures (regenerate with eval.abstention_eval)
ROADMAP.md      phase-by-phase plan, results and future work
Dockerfile      CPU image (offline backends by default; --build-arg WITH_MODELS=true for real models)
.github/        CI: lint, offline tests on Python 3.10/3.12/3.13, eval-set reproducibility, Docker smoke test
```

## Roadmap
Phases 0–7 are complete; see [ROADMAP.md](ROADMAP.md). Next steps:
- Done: retrieval and abstention with real models (bge-small, MS MARCO cross-encoder). Done as well: an end-to-end run with a local 7B reader (Qwen2.5 via Ollama). Remaining: a stronger reader (Claude or a larger model) and more calibration data.
- A larger LLM-generated eval set over the arXiv corpus, with paper-disjoint splits, so LTT can
  certify α ≤ 0.1.
- Conformal risk control (E[risk] ≤ α) as a less conservative alternative.
- NLI answer–evidence entailment and self-consistency as abstention features; an LLM judge for
  semantic answer equivalence.
- Serving: share the calibrated τ across replicas (e.g. a policy store), an async LLM client,
  and metrics (Prometheus).

## References
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Cormack, Clarke & Büttcher (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Nogueira & Cho (2019). *Passage Re-ranking with BERT.*
- Geifman & El-Yaniv (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
- Angelopoulos, Bates, Candès, Jordan & Lei (2021). *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.*
- Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.*
- Kamath, Jia & Liang (2020). *Selective Question Answering under Domain Shift.* ACL.
- Rajpurkar et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text* (EM / F1).

## License
[MIT](LICENSE) © 2026 Shayan Salehi
