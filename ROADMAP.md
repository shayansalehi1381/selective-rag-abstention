# Roadmap: Selective RAG with Calibrated Abstention

> **Goal.** Build a retrieval-augmented QA system that *knows when it doesn't know*.
> It answers only when the retrieved evidence supports an answer, and otherwise abstains.
> The abstention threshold is set with **conformal prediction**, so the error rate among
> answered questions is controlled at a user-chosen level α with a finite-sample,
> distribution-free guarantee.

---

## Architecture

```
                         ┌──────────────────────────── Phase 1 ───────────────────────────┐
  arXiv API ─► PDFs ─►  │ pypdf extraction ─► recursive chunking (512 / 64, provenance)  │
                         │                     │                                          │
                         │        ┌────────────┴────────────┐                             │
                         │   BM25 (rank-bm25)       FAISS dense (bge-small-en-v1.5)       │
                         │        └──────── RRF (k=60) ─────┘                             │
                         └─────────────────────────┬──────────────────────────────────────┘
                                                   ▼
                              Phase 3: cross-encoder reranker     ◄── Phase 2: benchmark (ground truth,
                                                                        Hit@K / MRR / Recall@K) scores each stage
                                                   ▼
             Phase 4: abstention features ─► confidence g(x) ─► LTT threshold τ̂_α
                                                   ▼
                                  g(x) ≥ τ̂_α ?  ── yes ─► Phase 5: generator (LLM) ─► ANSWER + cited chunks
                                                 └─ no ──► ABSTAIN ("insufficient evidence")
                                                   ▼
             Phase 6: risk–coverage, AURC, calibration │ Phase 7: FastAPI service, Docker, CI
```

## Research questions

1. **RQ1: Retrieval.** How much does hybrid retrieval (BM25 + dense + RRF) improve Recall@k
   and MRR over either retriever alone on a scientific-paper corpus?
2. **RQ2: Signals.** Which signals best separate answerable from unanswerable queries?
   Candidates: retrieval margins, sparse/dense rank agreement, reranker confidence,
   answer–evidence entailment and self-consistency. Measured by AUROC and AURC.
3. **RQ3: Guarantees.** Does split conformal calibration keep the empirical selective risk
   at or below α across α ∈ {0.05, 0.1, 0.2}, and what coverage does that cost?
4. **RQ4: Robustness.** How does the guarantee degrade under distribution shift between
   the calibration questions and the test questions (a new topic, a new question style)?

---

## Phase 0: Project setup ✅
- [x] Modular layout: `src/`, `eval/`, `tests/`, `data/{raw_pdfs,processed,eval}`
- [x] `requirements.txt`, `pyproject.toml` (pytest config, `integration` marker)
- [x] `.gitignore` for raw PDFs, derived indices, caches, virtualenvs and secrets
- [x] This roadmap

## Phase 1: Corpus ingestion and hybrid retrieval ✅
- [x] `src/data_loader.py`: arXiv downloader (idempotent, rate-limited, writes `manifest.jsonl`)
- [x] PDF text extraction with cleaning (NFKC, dehyphenation, whitespace normalisation)
- [x] Recursive character chunking (512 chars, 64 overlap) with exact character offsets,
      page numbers, `arxiv_id`, title and chunk index
- [x] `src/retriever.py`: BM25 (Okapi) + FAISS `IndexFlatIP` (cosine) + weighted RRF (k=60)
- [x] Per-retriever ranks and scores kept on each result, to feed the Phase 4 abstention features
- [x] Offline unit tests with a deterministic hashing embedder; opt-in integration tests
- **Deliverable:** `python -m src.data_loader download && python -m src.data_loader ingest`
  produces `data/processed/chunks.jsonl`, and `HybridRetriever` indexes and retrieves it.

## Phase 2: Evaluation benchmark and ground truth ✅
- [x] `eval/schemas.py`: strict Pydantic schemas (`EvalItem`, `EvalSet`). Answerable items must cite
      evidence chunks, unanswerable items must cite none and carry the canonical abstention answer,
      and every cited id is checked against a sha256-fingerprinted corpus
- [x] `eval/synthetic_gen.py`: 100 items = 70 answerable (50 factoid + 20 two-fact reasoning) + 30
      unanswerable (10 out-of-domain, 10 unsupported/hallucinated-entity, 10 false-premise conflicts)
  - [x] Offline deterministic engine: synthetic papers rendered from a fact table, chunked with
        the real splitter, with evidence linked to the exact chunk(s). Dev-split and ablation numbers
        act as in-paper hard negatives. Byte-identical output for a given seed
  - [x] Extractive offline engine for an existing chunk corpus (`--corpus`, lower fidelity)
  - [x] LLM engine (Anthropic / OpenAI-compatible): verbatim-evidence filter, plus an adversarial
        judge over the top-5 retrieved chunks that discards "unanswerable" items that are answerable
- [x] `eval/evaluate.py`: Hit@{1,3,5,10}, MRR@10 and Recall@K with bootstrap 95% CIs, per-category
      breakdown, dense vs sparse vs hybrid baselines, and an abstention preview (top-1 score AUROC)
- [x] Reports: `data/eval/retrieval_benchmark.{json,md}`
- **Deliverable:** `python -m eval.synthetic_gen generate --offline` →
  `python -m eval.evaluate run --baseline`
- [ ] Run with real bge-small embeddings and an LLM-generated set over the arXiv corpus

## Phase 3: Cross-encoder reranking ✅
- [x] `src/reranker.py`: `Reranker` over `sentence_transformers.CrossEncoder`
      (default `cross-encoder/ms-marco-MiniLM-L6-v2`), loaded lazily. It requests raw logits
      explicitly, and the probability is `sigmoid(logit)`
- [x] Offline fallback `MockCrossEncoder` (`mock-lexical-v1`): deterministic lexical-interaction
      scorer (IDF coverage, term proximity, bigrams, character-trigram fuzzy match), with fixed weights
      that are never fitted on eval data. Fallback is automatic but logged, and recorded in reports;
      `--strict-reranker` disables it
- [x] `RerankingRetriever`: hybrid top-50 → cross-encoder → top-k. First-stage ranks and scores are kept;
      `rerank_score` / `rerank_probability` are exposed for the Phase 4 abstention features
- [x] Benchmark track `hybrid_reranked` with paired-bootstrap Δ vs hybrid for every metric, top-1
      AUROC with bootstrap CI and a paired AUROC Δ, and the candidate-recall ceiling
- **Result (offline sandbox: hashing embedder + mock reranker):** no significant change
  (Hit@1 Δ = +0.000, Recall@10 Δ = −0.043, AUROC 0.689 vs 0.777, all CIs include 0). The misses are
  vocabulary mismatch ("backbone … based on" vs "built on top of … encoder"), which a lexical
  heuristic cannot bridge. This is the case a neural cross-encoder is for.
- [ ] Run with the real model: `python -m eval.evaluate run --baseline --embedder bge
      --reranker cross-encoder --strict-reranker`
- [ ] Compare `BAAI/bge-reranker-base` and sweep `retrieve_k` ∈ {10, 25, 50, 100} (latency vs ceiling)

## Phase 4: Calibrated abstention (selective RAG) ✅
- [x] Label (until generation exists): **safe to answer** = answerable and a gold chunk in the top-3
      reranked context (`k_ctx` is configurable)
- [x] `src/abstention.py`: 9 features from one pipeline call. Reranker: top-1 logit, top-1 minus top-2
      margin, softmax entropy. Retrieval: RRF top-1 and margin, BM25 and dense top-1, sparse/dense rank
      agreement, top-5 overlap
- [x] Confidence models: L2 logistic regression (numpy IRLS) on all features, plus Platt-scaled single
      signals. JSON-persisted `AbstentionPolicy` and a runtime `SelectiveRetriever`, which returns
      evidence or the canonical abstention answer
- [x] Thresholds: **ERM** (no guarantee) and **Learn-then-Test** (exact binomial p-values,
      fixed-sequence testing on a coverage grid fixed on the train split; P(risk ≤ α) ≥ 1 − δ)
- [x] `eval/abstention_eval.py`: 200 stratified 40/30/30 splits, reporting AUROC, AURC/E-AURC,
      selective accuracy at coverage {0.5, 0.8, 0.9, 1}, Brier, ECE, risk-control outcomes per α,
      risk-coverage / coverage-accuracy / reliability figures
- **Result (offline: hashing embedder + mock reranker, 66/100 safe):**
  - The combined model reaches AUROC **0.854** [0.75, 0.95], against 0.777 for the best Phase 2/3
    single signal. Pooled ECE is 0.046.
  - **ERM breaks its target:** at α = 0.1 test risk is ≤ α in only 45% of splits, and 66% at α = 0.2.
  - **LTT at α = 0.2 is valid in 90% of splits (the 1 − δ target)**, but it certifies only 21% of
    them. At α ≤ 0.1 it cannot certify anything with ~30 calibration items (it needs ≥ 22 / ≥ 45
    answered items). A guarantee needs more calibration data, and LTT reports exactly that.
- [ ] Scale calibration data (LLM-generated set, Phase 5) so that LTT can certify α = 0.1
- [ ] Conformal risk control (E[risk] ≤ α) as a less conservative alternative; NLI / self-consistency
      features once a generator exists

## Phase 5: Generation and answer correctness
- [ ] `src/pipeline.py`: an evidence-grounded generator with chunk citations
- [ ] Scale the Phase 2 set (LLM engine) and split it into train (feature fitting) / **calibration** / test,
      with no paper shared across splits
- [ ] Replace the retrieval-grounded Phase 4 label with answer correctness, and re-calibrate
- [ ] Answer correctness labels: exact match / token-F1, plus an LLM-judge for semantic equivalence
- **Deliverable:** `data/eval/qa_{train,cal,test}.jsonl` and a data card describing how they were built

## Phase 6: Evaluation
- [ ] `eval/evaluate.py`:
  - risk–coverage curves and **AURC** / E-AURC (Geifman & El-Yaniv)
  - selective accuracy at fixed coverage (50%, 80%, 90%)
  - **ECE** and reliability diagrams for g(x)
  - empirical risk vs target α over R random calibration/test splits, with error bars
    (the guarantee check)
  - AUROC for separating answerable from unanswerable questions
- [ ] All figures produced by scripts (matplotlib), fixed seeds, results logged to JSON
- **Deliverable:** `eval/results/` holding figures and tables, and a written results section in the README

## Phase 7: Serving and engineering
- [ ] `src/server.py`: FastAPI `POST /query` → `{answer | abstained, confidence, evidence[]}`,
      `GET /health`. Pydantic schemas, with the index loaded once at startup.
- [ ] Dockerfile (CPU), `make` targets, GitHub Actions CI (lint + offline tests)
- [ ] Latency budget: retrieval p95 < 100 ms on a CPU for a corpus of about 10k chunks
- **Deliverable:** `docker run` → a working API, documented in the README

---

## Engineering principles
- **Offline-testable:** model backends sit behind protocols (`Embedder`), so tests never touch the network.
- **Reproducible:** fixed seeds, versioned data manifests, deterministic tie-breaking in RRF.
- **Traceable evidence:** every chunk keeps `arxiv_id`, title, page and exact character offsets.
- **No pickle:** indices are persisted as a FAISS binary plus JSON/JSONL.

## References
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
- Cormack, Clarke & Büttcher (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR.
- Xiao et al. (2023). *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE).
- Geifman & El-Yaniv (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
- Angelopoulos & Bates (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.*
- Angelopoulos, Bates, Fisch, Lei & Schuster (2022). *Conformal Risk Control.*
- Angelopoulos, Bates, Candès, Jordan & Lei (2021). *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.*
- Kamath, Jia & Liang (2020). *Selective Question Answering under Domain Shift.* ACL.
