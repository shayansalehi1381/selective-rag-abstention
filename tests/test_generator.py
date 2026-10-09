"""Tests for the citation-grounded reader: schema, mock reader, LLM reader, fallback and cache."""

from __future__ import annotations

import json
import logging

import pytest

from src.constants import ABSTENTION_ANSWER
from src.generator import (
    FORCED_PROMPT,
    MOCK_NAME,
    GeneratedAnswer,
    Generator,
    GroundingError,
    LLMGenerator,
    MockGenerator,
    format_context,
    read_both,
    split_sentences,
    validate_against_context,
)
from src.llm import LLMRefusalError
from tests.conftest import make_chunks

PAPER = [
    "The backbone of ZenoRank is a pretrained RoBERTa encoder. The encoder stack of ZenoRank has 12 transformer layers.",
    "We train ZenoRank for 10 epochs. Optimisation of ZenoRank uses AdamW with a learning rate of 2e-5.",
    "On the SciClaimQA test set, the full ZenoRank model attains 78.4 macro-F1. "
    "In comparison, BM25 scores 70.1 macro-F1 on the SciClaimQA test set.",
]


@pytest.fixture
def context():
    return make_chunks(PAPER, arxiv_id="p1", title="ZenoRank paper")


def answer(ctx, **kw):
    data = dict(answer="78.4 macro-F1", citations=[ctx[2].chunk_id],
                evidence_quotes=["the full ZenoRank model attains 78.4 macro-F1"])
    data.update(kw)
    return GeneratedAnswer(**data)


# ---------------------------------------------------------------------------
# Grounding checks
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_answer_and_abstention(self, context):
        validate_against_context(answer(context), context)
        validate_against_context(GeneratedAnswer.abstention(backend="x", reason="r"), context)

    def test_quote_matching_ignores_case_and_whitespace(self, context):
        validate_against_context(answer(context, evidence_quotes=["THE FULL   ZenoRank model\nattains 78.4"]), context)

    @pytest.mark.parametrize("kw,match", [
        ({"citations": ["other::0000"]}, "not in the context"),
        ({"citations": []}, "cite"),
        ({"evidence_quotes": []}, "quote"),
        ({"evidence_quotes": ["ZenoRank attains 99.9 macro-F1"]}, "verbatim"),
        ({"answer": "  "}, "real answer"),
        ({"answer": ABSTENTION_ANSWER}, "real answer"),
    ])
    def test_rejections(self, context, kw, match):
        with pytest.raises(GroundingError, match=match):
            validate_against_context(answer(context, **kw), context)

    def test_quote_must_be_in_a_cited_chunk(self, context):
        bad = answer(context, citations=[context[0].chunk_id])  # quote lives in chunk 2
        with pytest.raises(GroundingError, match="verbatim"):
            validate_against_context(bad, context)

    def test_abstention_must_be_clean(self, context):
        bad = GeneratedAnswer(answer=ABSTENTION_ANSWER, abstain=True, citations=[context[0].chunk_id])
        with pytest.raises(GroundingError, match="abstention"):
            validate_against_context(bad, context)
        with pytest.raises(GroundingError):
            validate_against_context(GeneratedAnswer(answer="I think 5", abstain=True), context)

    def test_format_context_lists_ids(self, context):
        text = format_context(context)
        assert all(f"[{c.chunk_id}]" in text for c in context)


# ---------------------------------------------------------------------------
# Mock reader
# ---------------------------------------------------------------------------


class TestMockGenerator:
    @pytest.mark.parametrize("question,expected", [
        ("What test-set macro-F1 does the full ZenoRank model reach on SciClaimQA?", "78.4 macro-F1"),
        ("For how many epochs is ZenoRank trained?", "10 epochs"),
        ("Which learning rate is used to optimise ZenoRank?", "learning rate of 2e-5"),
        ("Which pretrained backbone is ZenoRank based on?", "RoBERTa"),
        ("How deep is the encoder of ZenoRank, in transformer layers?", "12 transformer layers"),
    ])
    def test_extracts_typed_spans(self, context, question, expected):
        ans = MockGenerator().generate(question, context)
        assert not ans.abstain and expected in ans.answer
        validate_against_context(ans, context)  # quotes are verbatim, citations in context

    def test_vocabulary_mismatch_is_a_documented_limitation(self):
        ctx = make_chunks(["ZenoRank is built on top of a RoBERTa encoder."], arxiv_id="p2")
        ans = MockGenerator().generate("Which pretrained backbone is ZenoRank based on?", ctx)
        assert ans.abstain  # "backbone ... based on" shares no content word with "built on top of ... encoder"

    def test_soft_coverage_matches_inflections(self):
        idf = {"train": 1.0, "trained": 1.0, "epochs": 1.0, "zenorank": 1.0}
        assert MockGenerator.soft_coverage("trained epochs", "We train for 10 epochs", idf) == pytest.approx(
            (0.5 + 1.0) / 2)
        assert MockGenerator.soft_coverage("", "anything", idf) == 0.0

    def test_deterministic(self, context):
        q = "What test-set macro-F1 does the full ZenoRank model reach on SciClaimQA?"
        assert MockGenerator().generate(q, context) == MockGenerator().generate(q, context)

    def test_abstains_on_unrelated_question(self, context):
        ans = MockGenerator().generate("Which grape varieties dominate Rioja wine blends?", context)
        assert ans.abstain and ans.answer == ABSTENTION_ANSWER and ans.reason.startswith("low_support")
        validate_against_context(ans, context)

    def test_forced_never_abstains(self, context):
        ans = MockGenerator().generate("Which grape varieties dominate Rioja wine blends?", context, forced=True)
        assert not ans.abstain and ans.forced and ans.answer

    def test_empty_inputs(self, context):
        assert MockGenerator().generate("q?", []).abstain
        assert MockGenerator().generate("", context).abstain
        forced = MockGenerator().generate("q?", [], forced=True)
        assert not forced.abstain and not forced.grounded

    def test_split_sentences_keeps_decimals(self):
        assert split_sentences("It reaches 78.4 F1. Then 2e-5 is used.\nNext line") == [
            "It reaches 78.4 F1.", "Then 2e-5 is used.", "Next line"]


# ---------------------------------------------------------------------------
# LLM reader (scripted client)
# ---------------------------------------------------------------------------


class ScriptedClient:
    name = "fake:model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, prompt, *, temperature, max_tokens):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def reply(ctx, **kw):
    data = {"answer": "78.4 macro-F1", "abstain": False, "citations": [ctx[2].chunk_id],
            "evidence_quotes": ["the full ZenoRank model attains 78.4 macro-F1"]}
    data.update(kw)
    return "Here you go:\n```json\n" + json.dumps(data) + "\n```"


class TestLLMGenerator:
    def test_parses_json_in_prose(self, context):
        client = ScriptedClient([reply(context)])
        ans = LLMGenerator(client).generate("q?", context)
        assert ans.answer == "78.4 macro-F1" and ans.backend == "fake:model" and ans.grounded
        assert "abstain" in client.prompts[0] and f"[{context[0].chunk_id}]" in client.prompts[0]

    def test_retry_fixes_a_bad_citation(self, context):
        client = ScriptedClient([reply(context, citations=["ghost::0001"]), reply(context)])
        ans = LLMGenerator(client).generate("q?", context)
        assert not ans.abstain and len(client.prompts) == 2
        assert "rejected" in client.prompts[1] and "ghost::0001" in client.prompts[1]

    def test_two_failures_become_an_abstention(self, context):
        client = ScriptedClient(["no json", reply(context, evidence_quotes=["made up"])])
        ans = LLMGenerator(client).generate("q?", context)
        assert ans.abstain and ans.reason == "invalid_output" and ans.answer == ABSTENTION_ANSWER

    def test_forced_keeps_ungrounded_answer_flagged(self, context):
        bad = reply(context, evidence_quotes=["made up"])
        client = ScriptedClient([bad, bad])
        ans = LLMGenerator(client).generate("q?", context, forced=True)
        assert not ans.abstain and not ans.grounded and ans.reason.startswith("ungrounded")
        assert client.prompts[0].startswith(FORCED_PROMPT.split("{")[0][:40])
        assert "abstain\": true" not in client.prompts[0]

    def test_model_abstention_is_accepted(self, context):
        client = ScriptedClient([json.dumps({"answer": ABSTENTION_ANSWER, "abstain": True,
                                             "citations": [], "evidence_quotes": []})])
        ans = LLMGenerator(client).generate("q?", context)
        assert ans.abstain and ans.reason is None

    def test_refusal_becomes_abstention(self, context):
        ans = LLMGenerator(ScriptedClient([LLMRefusalError("declined")])).generate("q?", context)
        assert ans.abstain and ans.reason == "refusal"


# ---------------------------------------------------------------------------
# Facade: fallback, strictness and cache
# ---------------------------------------------------------------------------


class TestGeneratorFacade:
    def test_mock_backend(self, context):
        g = Generator("mock")
        assert g.backend == MOCK_NAME and g.is_mock and g.fallback_reason is None

    def test_falls_back_when_sdk_missing(self, context, no_llm_sdk, caplog):
        with caplog.at_level(logging.WARNING, logger="src.generator"):
            g = Generator("anthropic")
            ans = g.generate("What test-set macro-F1 does the full ZenoRank model reach on SciClaimQA?", context)
        assert g.is_mock and "ImportError" in g.fallback_reason and ans.backend == MOCK_NAME
        assert any("falling back" in r.message for r in caplog.records)

    def test_falls_back_when_first_call_fails(self, context):
        g = Generator("anthropic", client=ScriptedClient([ConnectionError("no route")]))
        ans = g.generate("For how many epochs is ZenoRank trained?", context)
        assert g.is_mock and "ConnectionError" in g.fallback_reason and "10 epochs" in ans.answer

    def test_later_errors_abstain_per_item(self, context):
        g = Generator("anthropic", client=ScriptedClient([reply(context), TimeoutError("slow")]))
        assert not g.generate("q1?", context).abstain
        later = g.generate("q2?", context)
        assert later.abstain and later.reason == "error:TimeoutError" and not g.is_mock

    def test_strict_mode_raises(self, context):
        g = Generator("anthropic", allow_fallback=False, client=ScriptedClient([ConnectionError("no route")]))
        with pytest.raises(RuntimeError, match="could not use the anthropic reader"):
            g.generate("q?", context)

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            Generator("gpt")
        with pytest.raises(ValueError, match="model"):
            Generator("openai")

    def test_cache_hit_skips_the_client(self, context, tmp_path):
        path = tmp_path / "cache.jsonl"
        first = Generator("anthropic", client=ScriptedClient([reply(context)]), cache_path=path)
        a = first.generate("q?", context)
        second = Generator("anthropic", client=ScriptedClient([]), cache_path=path)  # would raise if called
        assert second.generate("q?", context) == a and second.cache_hits == 1

    def test_cache_key_depends_on_mode_and_context(self, context, tmp_path):
        g = Generator("mock", cache_path=tmp_path / "c.jsonl")
        q = "For how many epochs is ZenoRank trained?"
        g.generate(q, context)
        g.generate(q, context, forced=True)
        g.generate(q, context[:2])
        assert g.cache_hits == 0 and len((tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()) == 3
        g.generate(q, context)
        assert g.cache_hits == 1

    def test_cache_survives_fallback_between_runs(self, context, tmp_path):
        path = tmp_path / "c.jsonl"
        Generator("anthropic", client=ScriptedClient([ConnectionError("x")]), cache_path=path).generate("q?", context)
        again = Generator("anthropic", client=ScriptedClient([ConnectionError("x")]), cache_path=path)
        again.generate("q?", context)
        assert again.cache_hits == 1 and len(path.read_text(encoding="utf-8").splitlines()) == 1

    def test_read_both(self, context):
        pair = read_both(Generator("mock"), "Which grape varieties dominate Rioja wine blends?", context)
        assert pair.free.abstain and not pair.forced.abstain


def test_llm_clients_are_shared_with_benchmark_generation():
    import eval.synthetic_gen as sg
    import src.llm as llm

    assert sg.AnthropicClient is llm.AnthropicClient and sg.extract_json is llm.extract_json


def test_cache_is_thread_safe(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    ctx = make_chunks(PAPER, arxiv_id="p1")
    path = tmp_path / "c.jsonl"
    g = Generator("mock", cache_path=path)
    qs = ["For how many epochs is ZenoRank trained?", "Which learning rate is used to optimise ZenoRank?"] * 25
    with ThreadPoolExecutor(max_workers=16) as pool:
        answers = list(pool.map(lambda q: g.generate(q, ctx), qs))
    assert len({a.answer for a in answers}) == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len({json.loads(line)["key"] for line in lines}) == 2  # one line per key, no torn writes
