"""Tests for the FastAPI service (offline backends; FastAPI's TestClient, no real network)."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.abstention import AbstentionPolicy  # noqa: E402
from src.cli import _build_parser, apply_offline, build  # noqa: E402
from src.constants import ABSTENTION_ANSWER  # noqa: E402
from src.pipeline import SelectiveRAGPipeline  # noqa: E402
from src.server import PipelineService, ServerSettings, create_app  # noqa: E402

ANSWERABLE = "For how many epochs is SefiLens trained?"
OOD = "Which mortgage refinancing option minimises closing fees for retirees?"
TOKEN = "test-admin-token"


@pytest.fixture(scope="module")
def built():
    args = _build_parser().parse_args(["serve", "--offline", "--no-cache"])
    apply_offline(args)
    return build(args)


class CountingGenerator:
    """Delegates to the real generator and counts reader calls."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return self.inner.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def make_service(built, **settings) -> PipelineService:
    """A fresh service (its own pipeline object and policy) sharing the expensive indices."""
    p = built.pipeline
    pipeline = SelectiveRAGPipeline(p.retriever, CountingGenerator(p.generator), p.policy, k_ctx=p.k_ctx,
                                    top_k=p.top_k, forced=p.forced)
    return PipelineService(pipeline, ServerSettings(**{"admin_token": TOKEN, **settings}), eval_set=built.eval_set,
                           calibration_retriever=built.calibration_retriever, corpus=built.corpus)


@pytest.fixture
def service(built):
    return make_service(built)


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def query(client, **body):
    return client.post("/v1/query", json={"query": ANSWERABLE, **body})


# ---------------------------------------------------------------------------
# Health and info
# ---------------------------------------------------------------------------


class TestStatus:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok", "ready": True}

    def test_info(self, client, built):
        info = client.get("/v1/info").json()
        assert info["backends"] == {"embedder": "hashing-256", "reranker": "mock-lexical-v1",
                                    "generator": "mock-extractive-v1"}
        assert info["corpus"]["sha256"] == built.eval_set.corpus.sha256 and info["corpus"]["num_chunks"] == 144
        assert info["policy"]["method"] == "erm" and info["policy"]["statistical_guarantee"] is False
        assert info["calibration_enabled"] is True and info["limits"]["max_concurrency"] == 4

    def test_not_ready(self):
        c = TestClient(create_app(None))
        assert c.get("/health").status_code == 503 and c.get("/health").json()["status"] == "starting"
        r = c.post("/v1/query", json={"query": ANSWERABLE})
        assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable"

    def test_lifespan_builder(self, service):
        with TestClient(create_app(builder=lambda: service)) as c:
            assert c.get("/health").json()["ready"] is True

    def test_openapi(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/health", "/v1/info", "/v1/query", "/v1/calibrate"} <= set(paths)


# ---------------------------------------------------------------------------
# Query flow
# ---------------------------------------------------------------------------


class TestQuery:
    def test_answer_with_citations_and_scores(self, client):
        r = query(client)
        assert r.status_code == 200
        d = r.json()
        assert not d["abstained"] and "30 epochs" in d["answer"] and d["abstention_stage"] is None
        (cit,) = d["citations"]
        assert cit["chunk_id"] in d["context_chunk_ids"] and cit["quotes"] == ["We train SefiLens for 30 epochs."]
        assert cit["title"].startswith("SefiLens")
        assert d["gate"]["passed"] and d["gate"]["method"] == "erm" and not d["gate"]["statistical_guarantee"]
        assert d["gate"]["confidence"] >= d["gate"]["tau"]
        assert len(d["retrieval"]) >= 3 and d["retrieval"][0]["rerank_logit"] is not None
        assert len(d["features"]) == 9 and {"retrieve", "generate", "total"} <= set(d["latency_ms"])

    def test_gate_rejection_skips_reader(self, client, service):
        d = query(client, query=OOD).json()
        assert d["abstained"] and d["abstention_stage"] == "policy" and d["answer"] == ABSTENTION_ANSWER
        assert not d["gate"]["passed"] and d["citations"] == [] and service.pipeline.generator.calls == 0

    def test_standard_rag_mode(self, client):
        d = query(client, query=OOD, gate=False, forced_reader=True).json()
        assert not d["abstained"] and d["gate"] is None  # answers anyway: the hallucination the gate prevents

    def test_reader_abstention(self, client):
        d = query(client, query="Which grape varieties dominate Rioja wine blends?", gate=False).json()
        assert d["abstained"] and d["abstention_stage"] == "generator" and d["reader_reason"].startswith("low_support")

    def test_include_scores_false(self, client):
        d = query(client, include_scores=False).json()
        assert d["retrieval"] is None and d["features"] is None and d["citations"]

    def test_per_request_k_ctx_does_not_leak(self, client, service):
        assert len(query(client, k_ctx=1).json()["context_chunk_ids"]) == 1
        assert len(query(client).json()["context_chunk_ids"]) == 3 and service.pipeline.k_ctx == 3

    def test_request_id_round_trip(self, client):
        r = query(client, include_scores=False)
        assert r.headers["X-Request-ID"] == r.json()["request_id"]
        r2 = client.post("/v1/query", json={"query": ANSWERABLE}, headers={"X-Request-ID": "abc-123"})
        assert r2.headers["X-Request-ID"] == "abc-123" and r2.json()["request_id"] == "abc-123"


class TestValidation:
    @pytest.mark.parametrize("body", [{"query": ""}, {"query": "   "}, {"query": "x" * 2001},
                                      {"query": "q", "k_ctx": 0}, {"query": "q", "k_ctx": 11},
                                      {"query": "q", "top_k": 99}, {"query": "q", "unknown": 1}, {}])
    def test_rejected(self, client, body):
        r = client.post("/v1/query", json=body)
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "validation_error" and err["request_id"] == r.headers["X-Request-ID"]

    def test_non_json_body(self, client):
        r = client.post("/v1/query", content=b"not json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibrate:
    def test_erm_recalibration_swaps_policy(self, client):
        before = client.get("/v1/info").json()["policy"]
        r = client.post("/v1/calibrate", json={"method": "erm", "alpha": 0.3}, headers={"X-Admin-Token": TOKEN})
        assert r.status_code == 200
        d = r.json()
        assert d["policy"]["method"] == "erm" and d["policy"]["alpha"] == 0.3 and d["n_answered"] > 0
        after = client.get("/v1/info").json()["policy"]
        assert after["alpha"] == 0.3 and after["tau"] != before["tau"] and after["calibrated_at"]

    def test_ltt_reports_guarantee_and_reason(self, client):
        d = client.post("/v1/calibrate", json={"method": "ltt", "alpha": 0.2},
                        headers={"X-Admin-Token": TOKEN}).json()
        assert d["policy"]["guarantee"] == "P(risk <= 0.2) >= 0.9" and d["policy"]["statistical_guarantee"]
        if d["policy"]["tau"] == "inf":  # with 30 calibration items LTT may certify nothing, and says why
            assert d["policy"]["abstains_on_everything"] and d["policy"]["reason"] and d["n_answered"] == 0
            assert query(client).json()["abstention_stage"] == "policy"

    def test_auth(self, client, built):
        assert client.post("/v1/calibrate", json={}).status_code == 401
        r = client.post("/v1/calibrate", json={}, headers={"X-Admin-Token": "wrong"})
        assert r.status_code == 403 and r.json()["error"]["message"] == "invalid admin token"
        disabled = TestClient(create_app(make_service(built, admin_token=None)))
        r = disabled.post("/v1/calibrate", json={}, headers={"X-Admin-Token": TOKEN})
        assert r.status_code == 403 and "SRAG_ADMIN_TOKEN" in r.json()["error"]["message"]

    def test_invalid_body(self, client):
        for body in ({"method": "magic"}, {"alpha": 0}, {"alpha": 1.5}, {"delta": -1}):
            assert client.post("/v1/calibrate", json=body, headers={"X-Admin-Token": TOKEN}).status_code == 422

    def test_concurrent_calibration_conflicts(self, client, service):
        service._calibrate_lock.acquire()
        try:
            r = client.post("/v1/calibrate", json={}, headers={"X-Admin-Token": TOKEN})
            assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
        finally:
            service._calibrate_lock.release()

    def test_without_eval_set(self, built):
        svc = make_service(built)
        svc.eval_set = None
        r = TestClient(create_app(svc)).post("/v1/calibrate", json={}, headers={"X-Admin-Token": TOKEN})
        assert r.status_code == 400 and "eval set" in r.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Concurrency and robustness
# ---------------------------------------------------------------------------


class TestConcurrency:
    QUESTIONS = [ANSWERABLE, OOD, "What batch size is used when training LumiRetr?",
                 "Which learning rate is used to optimise SefiLens?"]

    def test_parallel_queries_match_sequential(self, service):
        qs = self.QUESTIONS * 8  # 32 requests
        sequential = [service.answer(q)[0] for q in qs]
        with ThreadPoolExecutor(max_workers=16) as pool:
            parallel = [r for r, _ in pool.map(service.answer, qs)]
        key = lambda r: (r.answer, r.abstained, r.abstention_stage, r.citations, round(r.confidence, 12))  # noqa: E731
        assert [key(r) for r in parallel] == [key(r) for r in sequential]

    def test_policy_swap_is_atomic_per_request(self, service):
        a = service.policy
        b = AbstentionPolicy(a.calibrator, -math.inf, {**a.meta, "alpha": 0.99})
        stop = threading.Event()

        def flipper():
            while not stop.is_set():
                service._policy = b if service._policy is a else a

        t = threading.Thread(target=flipper)
        t.start()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(service.answer, [OOD, ANSWERABLE] * 20))
        finally:
            stop.set()
            t.join()
        for resp, policy in results:
            assert policy in (a, b)
            assert (resp.abstention_stage != "policy") == (resp.confidence >= policy.tau)

    def test_reader_slots_exhausted_returns_429(self, built):
        svc = make_service(built, max_concurrency=1, reader_timeout_s=0.05)
        svc._reader_slots.acquire()  # simulate a long-running LLM call holding the only slot
        try:
            r = TestClient(create_app(svc)).post("/v1/query", json={"query": ANSWERABLE, "gate": False})
            assert r.status_code == 429 and r.headers["Retry-After"] == "1"
            assert r.json()["error"]["code"] == "too_many_requests"
        finally:
            svc._reader_slots.release()

    def test_gated_query_needs_no_reader_slot(self, built):
        svc = make_service(built, max_concurrency=1, reader_timeout_s=0.05)
        svc._reader_slots.acquire()
        try:
            r = TestClient(create_app(svc)).post("/v1/query", json={"query": OOD})
            assert r.status_code == 200 and r.json()["abstention_stage"] == "policy"
        finally:
            svc._reader_slots.release()

    def test_unhandled_error_is_a_clean_500(self, service, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(service.pipeline, "answer", boom)
        r = TestClient(create_app(service), raise_server_exceptions=False).post("/v1/query", json={"query": "q"})
        assert r.status_code == 500
        assert r.json()["error"]["code"] == "internal_error" and "secret" not in r.text
        assert r.json()["error"]["request_id"]

    def test_settings_validation(self):
        with pytest.raises(ValueError):
            ServerSettings(max_concurrency=0)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_serve_command_hands_a_ready_app_to_uvicorn(monkeypatch):
    import uvicorn

    from src.cli import main

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
    monkeypatch.setenv("SRAG_ADMIN_TOKEN", "from-env")
    assert main(["serve", "--offline", "--no-cache", "--port", "9123", "--max-concurrency", "2"]) == 0
    assert captured["host"] == "127.0.0.1" and captured["port"] == 9123 and captured["workers"] == 1
    svc = captured["app"].state.service
    assert svc.settings.admin_token == "from-env" and svc.settings.max_concurrency == 2
    assert TestClient(captured["app"]).get("/health").json()["ready"] is True
