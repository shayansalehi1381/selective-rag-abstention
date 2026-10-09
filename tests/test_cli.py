"""Tests for the interactive CLI (offline backends throughout)."""

from __future__ import annotations

import io
import json
import math

import pytest

from src.abstention import FEATURE_NAMES, AbstentionPolicy, LogisticCalibrator
from src.cli import gate_description, main
from src.constants import ABSTENTION_ANSWER

OFFLINE = ["--embedder", "hashing", "--reranker", "mock", "--generator", "mock", "--no-color", "--no-cache"]
ANSWERABLE = "For how many epochs is SefiLens trained?"
OOD = "Which mortgage refinancing option minimises closing fees for retirees?"


def run(argv, stdin_text=""):
    out = io.StringIO()
    code = main(argv, stdin=io.StringIO(stdin_text), stdout=out)
    return code, out.getvalue()


def ask_json(question=ANSWERABLE, *extra):
    code, out = run(["ask", question, *extra, "--json", *OFFLINE])
    assert code == 0
    return json.loads(out)


@pytest.fixture(scope="module")
def answerable_json():
    return ask_json()


class TestAsk:
    def test_text_output_has_three_panels(self):
        code, out = run(["ask", ANSWERABLE, *OFFLINE])
        assert code == 0
        for marker in ("① Retrieval + rerank", "② Abstention gate", "③ Answer", "no statistical guarantee",
                       "offline stand-ins", "30 epochs", "cited: synth-"):
            assert marker in out
        assert "\033[" not in out  # --no-color

    def test_json_output(self, answerable_json):
        d = answerable_json
        assert not d["abstained"] and "30 epochs" in d["answer"] and d["citations"]
        assert d["gate"]["method"] == "erm" and d["gate"]["alpha"] == 0.2 and d["gate"]["guarantee"] is None
        assert 0 <= d["gate"]["confidence"] <= 1 and d["gate"]["confidence"] >= d["gate"]["tau"]
        assert len(d["retrieval"]) == 5 and d["retrieval"][0]["rerank_logit"] is not None
        assert set(d["features"]) == set(FEATURE_NAMES) and d["warnings"] == []
        assert d["backends"]["generator"] == "mock-extractive-v1"

    def test_gate_stops_out_of_domain_question(self):
        d = ask_json(OOD)
        assert d["abstained"] and d["abstention_stage"] == "policy" and d["answer"] == ABSTENTION_ANSWER
        assert d["citations"] == [] and d["latency_ms"]["generate"] == 0.0

    def test_no_gate_is_standard_rag(self):
        d = ask_json(OOD, "--no-gate", "--forced-reader")
        assert not d["abstained"] and d["gate"] is None  # standard RAG answers anyway

    def test_ltt_calibration_reports_its_guarantee_or_abstains(self):
        d = ask_json(ANSWERABLE, "--calibrate", "ltt")
        assert d["gate"]["method"] == "ltt" and d["gate"]["guarantee"] == "P(risk <= 0.2) >= 0.9"
        if d["gate"]["tau"] == "inf":
            assert d["abstained"] and any("τ = ∞" in w for w in d["warnings"])

    def test_policy_file(self, tmp_path):
        rows = [{f: float(i) for f in FEATURE_NAMES} for i in range(10)]
        cal = LogisticCalibrator(list(FEATURE_NAMES)).fit(rows, [0] * 5 + [1] * 5)
        path = AbstentionPolicy(cal, math.inf, {"method": "erm", "alpha": 0.1, "corpus_sha256": "0" * 64}).save(
            tmp_path / "p.json")
        d = ask_json(ANSWERABLE, "--policy", str(path))
        assert d["abstained"] and d["abstention_stage"] == "policy"
        assert any("different corpus" in w for w in d["warnings"])

    def test_reader_abstention_is_shown(self):
        code, out = run(["ask", "Which grape varieties dominate Rioja wine blends?", "--no-gate", *OFFLINE])
        assert "the reader declined" in out and "gate disabled" in out

    def test_queries_another_corpus_with_warning(self, tmp_path):
        from src.data_loader import save_chunks_jsonl
        from tests.conftest import make_chunks

        corpus = save_chunks_jsonl(make_chunks(["We train ZenoRank for 10 epochs."]), tmp_path / "c.jsonl")
        d = ask_json("For how many epochs is ZenoRank trained?", "--corpus", str(corpus))
        assert any("does not transfer" in w for w in d["warnings"])

    def test_gate_description(self):
        assert gate_description(None) == "no gate (standard RAG)"


class TestChat:
    def test_scripted_session(self):
        script = "\n".join([":help", ":k 2", ANSWERABLE, ":gate off", OOD, ":scores off", ":json", ANSWERABLE,
                            ":bogus", ":quit"]) + "\n"
        code, out = run(["chat", *OFFLINE], script)
        assert code == 0
        assert "Commands:" in out and "context size = 2" in out and "gate off" in out
        assert "30 epochs" in out and "gate disabled" in out
        assert '"abstained": false' in out  # JSON mode toggled on
        assert "unknown command" in out

    def test_eof_exits_cleanly(self):
        code, out = run(["chat", *OFFLINE], "")
        assert code == 0 and "Selective RAG chat" in out

    def test_gate_on_without_policy(self):
        _, out = run(["chat", "--no-gate", *OFFLINE], ":gate on\n:quit\n")
        assert "no policy loaded" in out
