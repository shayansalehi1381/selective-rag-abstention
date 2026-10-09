"""FastAPI service for the selective RAG pipeline.

Endpoints (OpenAPI at ``/docs``):

* ``GET  /health``       liveness/readiness: 200 when ready, 503 while starting
* ``GET  /v1/info``      version, active backends (with fallback reasons), corpus, active policy
* ``POST /v1/query``     answer a question, or abstain, with citations, scores and latency
* ``POST /v1/calibrate`` recalibrate the gate τ (ERM or LTT); needs ``X-Admin-Token``

Concurrency model. Endpoints are sync functions, so FastAPI runs them in a thread pool.

* The pipeline is shared and only read. Per-request settings are passed as arguments
  (see ``SelectiveRAGPipeline.answer``).
* The active ``AbstentionPolicy`` is an immutable object behind a single reference. Each
  request reads it once. Recalibration builds a new policy off to the side and swaps the
  reference, so a request sees either the old policy or the new one, never a mix.
* A bounded semaphore caps concurrent reader (LLM) calls. If no slot frees up within
  ``reader_timeout_s``, the request gets 429.
* Recalibration is guarded by a non-blocking lock. A concurrent second request gets 409.

Run one worker per process (each worker loads its own models and holds its own τ), and
scale with replicas. Start it with ``selective-rag serve``, or build an app in code with
``create_app(service)``.
"""

from __future__ import annotations

import hmac
import logging
import math
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import __version__
from src.abstention import AbstentionPolicy
from src.pipeline import PipelineResponse, SelectiveRAGPipeline

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 2000


# ---------------------------------------------------------------------------
# Service (state + concurrency control)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerSettings:
    admin_token: str | None = None  # None disables POST /v1/calibrate
    max_concurrency: int = 4
    reader_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")


class ReaderBusy(RuntimeError):
    pass


class CalibrationInProgress(RuntimeError):
    pass


class _ReaderSlot:
    """Context manager that takes a reader slot, or raises ``ReaderBusy`` after a timeout."""

    def __init__(self, semaphore: threading.BoundedSemaphore, timeout: float) -> None:
        self.semaphore, self.timeout = semaphore, timeout

    def __enter__(self) -> None:
        if not self.semaphore.acquire(timeout=self.timeout):
            raise ReaderBusy("all reader slots are busy")

    def __exit__(self, *exc: Any) -> None:
        self.semaphore.release()


def policy_summary(policy: AbstentionPolicy | None, calibrated_at: str | None = None) -> dict[str, Any] | None:
    if policy is None:
        return None
    meta = policy.meta
    method = meta.get("method")
    return {
        "method": method,
        "alpha": meta.get("alpha"),
        "delta": meta.get("delta") if method == "ltt" else None,
        "tau": policy.tau if math.isfinite(policy.tau) else ("inf" if policy.tau > 0 else "-inf"),
        "guarantee": meta.get("guarantee") if method == "ltt" else None,
        "statistical_guarantee": method == "ltt",
        "abstains_on_everything": math.isinf(policy.tau) and policy.tau > 0,
        "reason": meta.get("reason"),
        "calibrated_at": calibrated_at,
    }


class PipelineService:
    """Owns the pipeline, the active policy and the concurrency primitives."""

    def __init__(self, pipeline: SelectiveRAGPipeline, settings: ServerSettings = ServerSettings(), *,
                 eval_set: Any = None, calibration_retriever: Any = None, corpus: dict[str, Any] | None = None,
                 warnings: list[str] | None = None) -> None:
        self.pipeline = pipeline
        self.settings = settings
        self.eval_set = eval_set
        self.calibration_retriever = calibration_retriever or pipeline.retriever
        self.corpus = corpus or {}
        self.warnings = list(warnings or [])
        self._policy: AbstentionPolicy | None = pipeline.policy
        self.calibrated_at: str | None = None
        self._reader_slots = threading.BoundedSemaphore(settings.max_concurrency)
        self._calibrate_lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_built(cls, built: Any, settings: ServerSettings = ServerSettings()) -> PipelineService:
        """From ``src.cli.build(args)``."""
        return cls(built.pipeline, settings, eval_set=built.eval_set, calibration_retriever=built.calibration_retriever,
                   corpus=built.corpus, warnings=built.warnings)

    @property
    def policy(self) -> AbstentionPolicy | None:
        return self._policy  # a single reference read is atomic

    def answer(self, query: str, *, gate: bool = True, k_ctx: int | None = None, top_k: int | None = None,
               forced_reader: bool | None = None) -> tuple[PipelineResponse, AbstentionPolicy | None]:
        policy = self.policy if gate else None  # snapshot: this request uses one policy throughout
        resp = self.pipeline.answer(query, policy=policy, k_ctx=k_ctx, top_k=top_k, forced=forced_reader,
                                    reader_guard=_ReaderSlot(self._reader_slots, self.settings.reader_timeout_s))
        return resp, policy

    def calibrate(self, *, method: str, alpha: float, delta: float, seed: int) -> tuple[AbstentionPolicy, float]:
        from eval.e2e_eval import fit_e2e_policy, score_items

        if self.eval_set is None:
            raise ValueError("no eval set is loaded, so the gate cannot be recalibrated")
        if not self._calibrate_lock.acquire(blocking=False):
            raise CalibrationInProgress("a calibration is already running")
        try:
            t0 = time.perf_counter()
            rows = score_items(self.eval_set, self.calibration_retriever, self.pipeline.generator,
                               k_ctx=self.pipeline.k_ctx)
            policy = fit_e2e_policy(rows, self.eval_set, reader="forced" if self.pipeline.forced else "free",
                                    method=method, alpha=alpha, delta=delta, seed=seed)
            self._policy = policy  # atomic swap
            self.pipeline.policy = policy  # keep the pipeline default in sync for in-process callers
            self.calibrated_at = datetime.now(timezone.utc).isoformat()
            return policy, (time.perf_counter() - t0) * 1e3
        finally:
            self._calibrate_lock.release()

    def info(self) -> dict[str, Any]:
        gen = self.pipeline.generator
        reranker = getattr(self.pipeline.retriever, "reranker", None)
        return {
            "version": __version__,
            "started_at": self.started_at,
            "backends": self.pipeline.backends(),
            "fallbacks": {"reranker": getattr(reranker, "fallback_reason", None), "generator": gen.fallback_reason},
            "corpus": self.corpus,
            "policy": policy_summary(self.policy, self.calibrated_at),
            "calibration_enabled": self.settings.admin_token is not None and self.eval_set is not None,
            "limits": {"max_concurrency": self.settings.max_concurrency,
                       "reader_timeout_s": self.settings.reader_timeout_s, "max_query_chars": MAX_QUERY_CHARS},
            "defaults": {"k_ctx": self.pipeline.k_ctx, "top_k": self.pipeline.top_k,
                         "forced_reader": self.pipeline.forced},
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    gate: bool = True
    k_ctx: int | None = Field(default=None, ge=1, le=10)
    top_k: int | None = Field(default=None, ge=1, le=50)
    forced_reader: bool | None = None
    include_scores: bool = True

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class Citation(BaseModel):
    chunk_id: str
    title: str | None = None
    quotes: list[str] = Field(default_factory=list)


class GateInfo(BaseModel):
    confidence: float | None
    tau: float | str
    method: str | None
    alpha: float | None
    guarantee: str | None
    statistical_guarantee: bool
    passed: bool


class RetrievalRow(BaseModel):
    rank: int
    chunk_id: str
    title: str
    rerank_logit: float | None
    rerank_probability: float | None
    first_stage_rank: int | None
    first_stage_score: float | None
    bm25_rank: int | None
    dense_rank: int | None


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    abstained: bool
    abstention_stage: Literal["policy", "generator"] | None
    reader_reason: str | None
    gate: GateInfo | None
    citations: list[Citation]
    context_chunk_ids: list[str]  # the chunks passed to the reader (top-k_ctx)
    retrieval: list[RetrievalRow] | None
    features: dict[str, float] | None
    latency_ms: dict[str, float]
    backends: dict[str, str | None]


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["erm", "ltt"] = "erm"
    alpha: float = Field(default=0.2, gt=0, lt=1)
    delta: float = Field(default=0.1, gt=0, lt=1)
    seed: int = Field(default=0, ge=0)


class CalibrateResponse(BaseModel):
    policy: dict[str, Any]
    cal_coverage: float
    cal_risk: float | None
    n_answered: int
    duration_ms: float


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def to_response(resp: PipelineResponse, policy: AbstentionPolicy | None, request_id: str,
                include_scores: bool) -> QueryResponse:
    by_id = {r.chunk.chunk_id: r.chunk for r in resp.evidence}
    citations = []
    for cid in resp.citations:
        chunk = by_id.get(cid)
        quotes = [q for q in resp.evidence_quotes if chunk is not None and _norm(q) in _norm(chunk.text)]
        citations.append(Citation(chunk_id=cid, title=chunk.title if chunk else None, quotes=quotes))
    gate = None
    if policy is not None:
        summary = policy_summary(policy)
        gate = GateInfo(confidence=resp.confidence, tau=summary["tau"], method=summary["method"],
                        alpha=summary["alpha"], guarantee=summary["guarantee"],
                        statistical_guarantee=summary["statistical_guarantee"],
                        passed=resp.abstention_stage != "policy")
    retrieval = None
    if include_scores:
        retrieval = [RetrievalRow(rank=r.rank, chunk_id=r.chunk.chunk_id, title=r.chunk.title,
                                  rerank_logit=r.rerank_score, rerank_probability=r.rerank_probability,
                                  first_stage_rank=r.first_stage_rank, first_stage_score=r.first_stage_score,
                                  bm25_rank=r.bm25_rank, dense_rank=r.dense_rank) for r in resp.ranked]
    return QueryResponse(
        request_id=request_id, query=resp.query, answer=resp.answer, abstained=resp.abstained,
        abstention_stage=resp.abstention_stage, reader_reason=resp.generation.reason if resp.generation else None,
        gate=gate, citations=citations, context_chunk_ids=[r.chunk.chunk_id for r in resp.evidence], retrieval=retrieval, features=resp.features if include_scores else None,
        latency_ms=resp.latency_ms, backends=resp.backends)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _error(status: int, code: str, message: str, request: Request, headers: dict | None = None) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    hdrs = {"X-Request-ID": request_id} if request_id else {}
    hdrs.update(headers or {})
    return JSONResponse(status_code=status, headers=hdrs,
                        content={"error": {"code": code, "message": message, "request_id": request_id}})


def get_service(request: Request) -> PipelineService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="the pipeline is still loading")
    return service


def create_app(service: PipelineService | None = None, *, builder: Any = None) -> FastAPI:
    """Create the app. Pass a ready ``service``, or a zero-argument ``builder`` that the
    lifespan hook calls at startup (e.g. to load models after the process has started)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None and builder is not None:
            app.state.service = builder()
        yield

    app = FastAPI(title="Selective RAG", version=__version__, lifespan=lifespan,
                  description="Retrieval-augmented QA that abstains when the evidence is insufficient.")
    app.state.service = service

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        codes = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict", 422: "validation_error",
                 429: "too_many_requests", 503: "unavailable"}
        return _error(exc.status_code, codes.get(exc.status_code, "error"), str(exc.detail), request,
                      headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
        return _error(422, "validation_error", details, request)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error (request %s)", getattr(request.state, "request_id", "?"))
        return _error(500, "internal_error", "internal server error", request)

    @app.get("/health")
    def health(request: Request):
        ready = getattr(request.app.state, "service", None) is not None
        return JSONResponse(status_code=200 if ready else 503,
                            content={"status": "ok" if ready else "starting", "ready": ready})

    @app.get("/v1/info")
    def info(svc: PipelineService = Depends(get_service)) -> dict[str, Any]:
        return svc.info()

    @app.post("/v1/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request, svc: PipelineService = Depends(get_service)) -> QueryResponse:
        try:
            resp, policy = svc.answer(body.query, gate=body.gate, k_ctx=body.k_ctx, top_k=body.top_k,
                                      forced_reader=body.forced_reader)
        except ReaderBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "1"}) from exc
        return to_response(resp, policy, request.state.request_id, body.include_scores)

    @app.post("/v1/calibrate", response_model=CalibrateResponse)
    def calibrate(body: CalibrateRequest, svc: PipelineService = Depends(get_service),
                  x_admin_token: str | None = Header(default=None)) -> CalibrateResponse:
        if svc.settings.admin_token is None:
            raise HTTPException(status_code=403, detail="calibration is disabled: set SRAG_ADMIN_TOKEN on the server")
        if x_admin_token is None:
            raise HTTPException(status_code=401, detail="missing X-Admin-Token header")
        if not hmac.compare_digest(x_admin_token.encode(), svc.settings.admin_token.encode()):
            raise HTTPException(status_code=403, detail="invalid admin token")
        try:
            policy, duration = svc.calibrate(method=body.method, alpha=body.alpha, delta=body.delta, seed=body.seed)
        except CalibrationInProgress as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:  # e.g. single-class calibration labels, no eval set
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        meta = policy.meta
        risk = meta.get("cal_risk")
        return CalibrateResponse(policy=policy_summary(policy, svc.calibrated_at), cal_coverage=meta["cal_coverage"],
                                 cal_risk=risk if isinstance(risk, float) else None, n_answered=meta["n_answered"],
                                 duration_ms=duration)

    return app
