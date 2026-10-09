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
                              Phase 2: cross-encoder reranker
                                                   ▼
                              Phase 3: generator (LLM, evidence-grounded prompt)
                                                   ▼
             Phase 4: abstention features ─► nonconformity score s(x) ─► conformal threshold τ̂_α
                                                   ▼
                                  s(x) ≤ τ̂_α ?  ── yes ─► ANSWER + cited chunks
                                                 └─ no ──► ABSTAIN ("insufficient evidence")
                                                   ▼
             Phase 5: risk–coverage, AURC, calibration │ Phase 6: FastAPI service, Docker, CI
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
- [x] Per-retriever ranks and scores kept on each result, to feed the Phase 4 features
- [x] Offline unit tests with a deterministic hashing embedder; opt-in integration tests
- **Deliverable:** `python -m src.data_loader download && python -m src.data_loader ingest`
  produces `data/processed/chunks.jsonl`, and `HybridRetriever` indexes and retrieves it.

## Phase 2: Reranking
- [ ] `src/reranker.py`: cross-encoder (`BAAI/bge-reranker-base`, with
      `cross-encoder/ms-marco-MiniLM-L-6-v2` as the fast baseline)
- [ ] Rerank the top 50 hybrid candidates down to the top 5. Cache scores for reproducibility.
- [ ] Ablation: BM25 vs dense vs hybrid vs hybrid+rerank (Recall@{1,5,10}, MRR@10, nDCG@10)
- **Deliverable:** a retrieval ablation table in `eval/results/retrieval.md`

## Phase 3: Generation and evaluation data
- [ ] `src/pipeline.py`: an evidence-grounded generator with chunk citations
- [ ] `eval/synthetic_gen.py`: a synthetic QA set built from the corpus
  - [ ] **Answerable** questions: an LLM writes Q/A pairs grounded in a sampled chunk
        (keeping the gold `chunk_id`), then a round-trip filter checks that the answer can be
        recovered from that chunk
  - [ ] **Unanswerable** questions: out-of-corpus topics, entity and number perturbations
        of answerable questions, and questions whose premise is false
  - [ ] Splits: train (feature fitting) / **calibration** / test, with no paper shared across splits
- [ ] Answer correctness labels: exact match / token-F1, plus an LLM-judge for semantic equivalence
- **Deliverable:** `data/eval/qa_{train,cal,test}.jsonl` and a data card describing how they were built

## Phase 4: Abstention and conformal calibration
- [ ] `src/abstention.py`, the feature extractor:
  - retrieval: top-1 score, top-1 minus top-2 margin, BM25 vs dense rank agreement, score entropy
  - reranker: maximum relevance probability
  - generation: NLI entailment between the answer and the cited evidence, and
    self-consistency across *n* samples
- [ ] Confidence model g(x) ∈ [0, 1] (logistic regression or gradient boosting) trained on the train split
- [ ] **Split conformal risk control**: on the calibration set, pick
      τ̂ = inf{τ : (n/(n+1))·R̂_n(τ) + 1/(n+1) ≤ α}, where R̂_n is the selective risk.
      This gives E[risk] ≤ α (Angelopoulos et al., *Conformal Risk Control*). A
      Learn-then-Test variant gives a high-probability (1 − δ) guarantee.
- [ ] Baselines: no abstention, a fixed retrieval-score threshold, raw LLM self-reported confidence
- **Deliverable:** a `ConformalAbstainer` with `fit(cal_scores, cal_errors, alpha)` and `decide(x)`

## Phase 5: Evaluation
- [ ] `eval/evaluate.py`:
  - risk–coverage curves and **AURC** / E-AURC (Geifman & El-Yaniv)
  - selective accuracy at fixed coverage (50%, 80%, 90%)
  - **ECE** and reliability diagrams for g(x)
  - empirical risk vs target α over R random calibration/test splits, with error bars
    (the guarantee check)
  - AUROC for separating answerable from unanswerable questions
- [ ] All figures produced by scripts (matplotlib), fixed seeds, results logged to JSON
- **Deliverable:** `eval/results/` holding figures and tables, and a written results section in the README

## Phase 6: Serving and engineering
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
