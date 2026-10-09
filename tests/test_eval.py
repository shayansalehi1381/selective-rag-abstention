"""Tests for the evaluation suite: schemas, offline/LLM generation, metrics and the harness."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from eval.evaluate import (
    auroc,
    bootstrap_ci,
    evaluate_retriever,
    hit_at_k,
    main as evaluate_main,
    reciprocal_rank,
    recall_at_k,
    render_markdown,
    run_benchmark,
)
from eval.schemas import (
    ABSTENTION_ANSWER,
    Category,
    EvalItem,
    EvalSet,
    ItemMetadata,
    corpus_fingerprint,
    count_categories,
)
from eval.synthetic_gen import (
    AnthropicClient,
    Draft,
    LLMEngine,
    Quotas,
    SyntheticCorpusFactory,
    build_offline_eval_set,
    chunk_papers,
    corpus_vocabulary,
    extract_json,
    main as generate_main,
)
from src.data_loader import load_chunks_jsonl
from src.retriever import HashingEmbedder, HybridRetriever, tokenize
from tests.conftest import make_chunks

META = {"generator": "test"}


def answerable(**overrides):
    data = dict(id="eval_001", question="What EM does ZenoRank reach?", is_answerable=True,
                category=Category.IN_DOMAIN_FACTOID, ground_truth_chunk_ids=["p::0001"],
                reference_answer="71.2 EM", metadata=META)
    data.update(overrides)
    return EvalItem(**data)


def unanswerable(**overrides):
    data = dict(id="eval_002", question="Who composed this cantata?", is_answerable=False,
                category=Category.OUT_OF_DOMAIN, ground_truth_chunk_ids=[],
                reference_answer=ABSTENTION_ANSWER, metadata=META)
    data.update(overrides)
    return EvalItem(**data)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_valid_items(self):
        assert answerable().metadata.difficulty == "medium"
        assert unanswerable().ground_truth_chunk_ids == []

    @pytest.mark.parametrize("overrides", [
        {"ground_truth_chunk_ids": []},
        {"ground_truth_chunk_ids": ["a", "a"]},
        {"reference_answer": ABSTENTION_ANSWER},
        {"category": Category.OUT_OF_DOMAIN},
        {"id": "item_1"},
        {"id": "eval_1"},
        {"question": "   "},
        {"unexpected": True},
    ])
    def test_answerable_invariants(self, overrides):
        with pytest.raises(ValidationError):
            answerable(**overrides)

    @pytest.mark.parametrize("overrides", [
        {"ground_truth_chunk_ids": ["p::0001"]},
        {"reference_answer": "Bach"},
        {"category": Category.IN_DOMAIN_REASONING},
    ])
    def test_unanswerable_invariants(self, overrides):
        with pytest.raises(ValidationError):
            unanswerable(**overrides)

    def test_items_are_immutable(self):
        with pytest.raises(ValidationError):
            answerable().question = "changed"

    def test_metadata_allows_extra_keys_but_validates_difficulty(self):
        assert ItemMetadata(generator="g", custom="x").model_dump()["custom"] == "x"
        with pytest.raises(ValidationError):
            ItemMetadata(generator="g", difficulty="trivial")

    def _set(self, items, **overrides):
        data = dict(generator="test", corpus={"path": "c.jsonl", "num_chunks": 1, "sha256": "0" * 64},
                    counts=count_categories(items), items=items)
        data.update(overrides)
        return EvalSet(**data)

    def test_eval_set_round_trip_and_structure(self, tmp_path):
        es = self._set([answerable(), unanswerable()])
        loaded = EvalSet.load(es.save(tmp_path / "set.json"))
        assert loaded == es
        raw = json.loads((tmp_path / "set.json").read_text())
        assert set(raw) == {"schema_version", "generator", "seed", "corpus", "counts", "items"}
        assert set(raw["items"][0]) == {"id", "question", "is_answerable", "category",
                                        "ground_truth_chunk_ids", "reference_answer", "metadata"}
        assert raw["items"][1]["category"] == "out_of_domain"

    def test_eval_set_rejects_inconsistencies(self):
        with pytest.raises(ValidationError, match="unique"):
            self._set([answerable(), answerable()])
        with pytest.raises(ValidationError, match="counts"):
            self._set([answerable()], counts={"in_domain_factoid": 5})
        with pytest.raises(ValidationError, match="sorted"):
            self._set([unanswerable(), answerable()])

    def test_validate_against_corpus(self):
        es = self._set([answerable()])
        es.validate_against_corpus(["p::0001"])
        with pytest.raises(ValueError, match="not in the corpus"):
            es.validate_against_corpus(["other"])


# ---------------------------------------------------------------------------
# Offline generation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    out = tmp_path_factory.mktemp("offline") / "corpus.jsonl"
    eval_set, chunks = build_offline_eval_set(seed=42, synthetic_corpus_out=out)
    return eval_set, chunks, out


class TestOfflineGeneration:
    def test_exact_distribution(self, offline):
        eval_set, _, _ = offline
        assert len(eval_set.items) == 100
        assert len(eval_set.answerable) == 70 and len(eval_set.unanswerable) == 30
        assert eval_set.counts == {"in_domain_factoid": 50, "in_domain_reasoning": 20,
                                   "out_of_domain": 10, "unsupported_fact": 10, "subtle_conflict": 10}
        assert [i.id for i in eval_set.items] == [f"eval_{n:03d}" for n in range(1, 101)]

    def test_categories_are_interleaved(self, offline):
        cats = [i.category for i in offline[0].items[:20]]
        assert len(set(cats)) > 1

    def test_ground_truth_contains_the_evidence(self, offline):
        eval_set, chunks, _ = offline
        by_id = {c.chunk_id: c for c in chunks}
        for item in eval_set.answerable:
            texts = [by_id[cid].text for cid in item.ground_truth_chunk_ids]
            for sentence in item.metadata.model_dump()["evidence_sentences"]:
                assert any(sentence in t for t in texts), item.id

    def test_reasoning_items_need_multiple_facts(self, offline):
        for item in offline[0].items:
            if item.category is Category.IN_DOMAIN_REASONING:
                assert len(item.metadata.model_dump()["evidence_sentences"]) == 2

    def test_unanswerable_items_are_really_unanswerable(self, offline):
        eval_set, chunks, _ = offline
        vocab = corpus_vocabulary(chunks)
        corpus_text = " ".join(c.text for c in chunks)
        for item in eval_set.unanswerable:
            meta = item.metadata.model_dump()
            if item.category is Category.OUT_OF_DOMAIN:
                assert not set(tokenize(item.question)) & vocab
            if "hallucinated_entity" in meta:
                assert meta["hallucinated_entity"] not in corpus_text
            if item.category is Category.SUBTLE_CONFLICT:
                assert meta["distractor_chunk_ids"], "conflicts must point at the contradicted chunk"

    def test_corpus_file_matches_fingerprint(self, offline):
        eval_set, chunks, path = offline
        assert load_chunks_jsonl(path) == chunks
        assert eval_set.corpus.sha256 == corpus_fingerprint(chunks)
        assert eval_set.corpus.num_chunks == len(chunks)

    def test_every_fact_has_exact_evidence_chunks(self):
        papers = SyntheticCorpusFactory(seed=7).build()
        chunks = {c.chunk_id: c for c in chunk_papers(papers)}
        for paper in papers:
            for f in paper.facts:
                assert paper.text[f.start:f.end] == f.sentence
                for cid in f.chunk_ids:
                    assert f.sentence in chunks[cid].text

    def test_deterministic_for_a_seed(self, tmp_path):
        a, _ = build_offline_eval_set(seed=3, synthetic_corpus_out=tmp_path / "a.jsonl")
        b, _ = build_offline_eval_set(seed=3, synthetic_corpus_out=tmp_path / "b.jsonl")
        c, _ = build_offline_eval_set(seed=4, synthetic_corpus_out=tmp_path / "c.jsonl")
        assert b.to_json().replace("b.jsonl", "a.jsonl") == a.to_json()
        assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
        assert [i.question for i in a.items] != [i.question for i in c.items]

    def test_custom_quotas(self, tmp_path):
        es, _ = build_offline_eval_set(seed=1, synthetic_corpus_out=tmp_path / "c.jsonl",
                                       quotas=Quotas(5, 3, 2, 2, 2))
        assert len(es.items) == 14 and es.counts["in_domain_reasoning"] == 3

    @pytest.mark.parametrize("bad", [dict(in_domain_factoid=-1), dict(in_domain_factoid=0, in_domain_reasoning=0,
                                     out_of_domain=0, unsupported_fact=0, subtle_conflict=0)])
    def test_invalid_quotas(self, bad):
        with pytest.raises(ValueError):
            Quotas(**bad)

    def test_impossible_quota_fails_loudly(self, tmp_path):
        with pytest.raises(RuntimeError, match="out_of_domain"):
            build_offline_eval_set(seed=1, synthetic_corpus_out=tmp_path / "c.jsonl",
                                   quotas=Quotas(out_of_domain=500))

    def test_extractive_engine_on_local_corpus(self, tmp_path, offline):
        _, _, corpus_path = offline
        es, chunks = build_offline_eval_set(seed=5, corpus_path=corpus_path, quotas=Quotas(10, 4, 3, 3, 3))
        assert es.generator == "offline-extractive-v1"
        assert len(es.items) == 23
        es.validate_against_corpus(c.chunk_id for c in chunks)

    def test_cli(self, tmp_path, capsys):
        out = tmp_path / "eval_set.json"
        assert generate_main(["generate", "--offline", "--output", str(out),
                              "--synthetic-corpus-out", str(tmp_path / "corpus.jsonl")]) == 0
        assert len(EvalSet.load(out).items) == 100
        assert "total" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# LLM engine (scripted fake client, no network)
# ---------------------------------------------------------------------------

LLM_TEXTS = [
    "The full ZenoRank model reaches 71.2 exact match on the SciClaimQA test set. It uses eight layers.",
    "ZenoRank is trained for ten epochs with a learning rate of 2e-5 and large batches of examples.",
    "Conformal prediction gives coverage guarantees from a held-out calibration split of the data.",
]


class FakeLLM:
    name = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, system, prompt, *, temperature, max_tokens):
        self.calls.append((prompt, temperature))
        return self.script.pop(0)(prompt) if callable(self.script[0]) else self.script.pop(0)


def llm_chunks():
    return make_chunks([t * 3 for t in LLM_TEXTS], arxiv_id="p1", title="ZenoRank")


class TestLLMEngine:
    def test_extract_json(self):
        assert extract_json('Sure! ```json\n{"question": "q?"}\n``` done') == {"question": "q?"}
        assert extract_json('{"a": {"b": 1}} trailing {"c": 2}') == {"a": {"b": 1}}
        with pytest.raises(ValueError):
            extract_json("no json here {oops")

    def test_factoid_with_retry_and_evidence_filter(self):
        chunks = llm_chunks()
        good = json.dumps({"question": "What EM does ZenoRank get on SciClaimQA?", "answer": "71.2",
                           "evidence_quote": "reaches 71.2 exact match"})
        hallucinated = json.dumps({"question": "q2?", "answer": "99", "evidence_quote": "reaches 99.9 EM"})
        rng = random.Random(0)
        # Force the sampled chunk to be the one with the quote.
        engine = LLMEngine(FakeLLM(["not json", hallucinated, good]), chunks[:1], rng)
        drafts = engine.generate(Quotas(1, 0, 0, 0, 0))
        assert len(drafts) == 1 and drafts[0].ground_truth_chunk_ids == [chunks[0].chunk_id]
        assert engine.rejections == {"ValueError": 1, "evidence_not_verbatim": 1}
        assert drafts[0].metadata["generator"] == "llm:fake"

    def test_judge_discards_answerable_unanswerables(self):
        chunks = llm_chunks()
        q = lambda text: json.dumps({"question": text})  # noqa: E731
        judge_yes = json.dumps({"answerable": True, "chunk_id": chunks[0].chunk_id})
        judge_no = json.dumps({"answerable": False})
        client = FakeLLM([q("How many layers does ZenoRank use?"), judge_yes,
                          q("What dropout does ZenoRank use?"), judge_no])
        engine = LLMEngine(client, chunks, random.Random(0), retrieve=lambda question, k: chunks[:k])
        drafts = engine.generate(Quotas(0, 0, 0, 1, 0))
        assert [d.question for d in drafts] == ["What dropout does ZenoRank use?"]
        assert drafts[0].metadata["distractor_chunk_ids"] == [c.chunk_id for c in chunks]
        assert drafts[0].metadata["verified"] is True
        assert engine.rejections == {"judge_says_answerable": 1}
        assert client.calls[1][1] == 0.0  # the judge runs at temperature 0

    def test_reasoning_requires_both_quotes(self):
        chunks = llm_chunks()
        ok = json.dumps({"question": "How long and how well?", "answer": "10 epochs; 71.2 EM",
                         "evidence_quotes": ["reaches 71.2 exact match", "trained for ten epochs"]})

        def respond(prompt):
            p1 = prompt.index("Passage 1:")
            p2 = prompt.index("Passage 2:")
            first_has_em = "71.2" in prompt[p1:p2]
            quotes = ok if first_has_em else json.dumps({**json.loads(ok), "evidence_quotes": [
                "trained for ten epochs", "reaches 71.2 exact match"]})
            return quotes

        engine = LLMEngine(FakeLLM([respond]), chunks[:2], random.Random(1))
        (draft,) = engine.generate(Quotas(0, 1, 0, 0, 0))
        assert sorted(draft.ground_truth_chunk_ids) == sorted(c.chunk_id for c in chunks[:2])

    def test_gives_up_after_max_attempts(self):
        engine = LLMEngine(FakeLLM(["garbage"] * 10), llm_chunks(), random.Random(0), max_attempts_per_item=2)
        with pytest.raises(RuntimeError, match="could not fill"):
            engine.generate(Quotas(1, 0, 0, 0, 0))

    def test_anthropic_adapter_omits_temperature_by_default(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(stop_reason="end_turn", content=[
                SimpleNamespace(type="thinking", text="ignored"), SimpleNamespace(type="text", text='{"x": 1}')])

        fake_sdk = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        out = AnthropicClient(client=fake_sdk).complete("sys", "prompt", temperature=0.7, max_tokens=100)
        assert out == '{"x": 1}'
        assert "temperature" not in captured and "extra_body" not in captured
        assert captured["model"] == "claude-opus-5-5"
        assert captured["fallbacks"] == "default"
        assert captured["betas"] == ["server-side-fallback-2026-07-01"]
        captured.clear()
        legacy = AnthropicClient("legacy-model", supports_temperature=True, client=fake_sdk)
        legacy.complete("sys", "prompt", temperature=0.3, max_tokens=100)
        assert captured["extra_body"] == {"temperature": 0.3}

    def test_anthropic_adapter_matches_installed_sdk(self):
        anthropic = pytest.importorskip("anthropic")
        import inspect

        params = inspect.signature(anthropic.Anthropic(api_key="test").beta.messages.create).parameters
        assert {"model", "max_tokens", "system", "messages", "betas", "fallbacks", "extra_body"} <= set(params)

    def test_anthropic_refusal_is_rejected_not_fatal(self):
        refusal = SimpleNamespace(stop_reason="refusal", content=[])
        sdk = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **_: refusal)))
        engine = LLMEngine(AnthropicClient(client=sdk), llm_chunks(), random.Random(0), max_attempts_per_item=1)
        with pytest.raises(RuntimeError):
            engine.generate(Quotas(1, 0, 0, 0, 0))
        assert engine.rejections == {"LLMRefusalError": 1}


def test_draft_answerability():
    assert Draft("q", Category.IN_DOMAIN_REASONING, ["a"], "x", {}).is_answerable
    assert not Draft("q", Category.SUBTLE_CONFLICT, [], ABSTENTION_ANSWER, {}).is_answerable


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    ranked = ["x", "g1", "y", "g2"]
    gold = ["g1", "g2"]

    def test_toy_ranking(self):
        assert hit_at_k(self.ranked, self.gold, 1) == 0.0
        assert hit_at_k(self.ranked, self.gold, 2) == 1.0
        assert reciprocal_rank(self.ranked, self.gold) == 0.5
        assert recall_at_k(self.ranked, self.gold, 3) == 0.5
        assert recall_at_k(self.ranked, self.gold, 5) == 1.0  # k beyond the list length

    def test_no_relevant_chunk(self):
        assert reciprocal_rank(["a", "b"], ["z"]) == 0.0
        assert hit_at_k([], ["z"], 10) == 0.0
        assert recall_at_k(["a"], ["z", "y"], 1) == 0.0

    def test_rr_respects_cutoff(self):
        assert reciprocal_rank(self.ranked, ["g2"], k=3) == 0.0
        assert reciprocal_rank(self.ranked, ["g2"], k=4) == 0.25

    @pytest.mark.parametrize("fn", [hit_at_k, recall_at_k])
    def test_invalid_inputs(self, fn):
        with pytest.raises(ValueError):
            fn(["a"], [], 1)
        with pytest.raises(ValueError):
            fn(["a"], ["a"], 0)

    def test_bootstrap_ci(self):
        values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        lo, hi = bootstrap_ci(values, seed=1)
        assert lo <= sum(values) / len(values) <= hi
        assert bootstrap_ci(values, seed=1) == (lo, hi)
        assert bootstrap_ci([0.5] * 5) == (0.5, 0.5)

    def test_auroc(self):
        assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
        assert auroc([0.1], [0.9]) == 0.0
        assert auroc([0.5, 0.5], [0.5]) == 0.5
        assert auroc([0.9, 0.2], [0.5]) == 0.5

    def test_evaluate_retriever_aggregation(self):
        items = [answerable(id="eval_001", ground_truth_chunk_ids=["g1"]),
                 answerable(id="eval_002", question="other?", ground_truth_chunk_ids=["g2", "g3"]),
                 unanswerable(id="eval_003")]
        rankings = {items[0].question: [("g1", 0.9), ("x", 0.5)],
                    items[1].question: [("x", 0.8), ("g2", 0.7)],
                    items[2].question: [("x", 0.3)]}
        rep = evaluate_retriever("toy", lambda q, k: rankings[q][:k], items, ks=(1, 2), n_bootstrap=50)
        assert rep.n_answerable == 2
        assert rep.metrics["hit@1"] == 0.5
        assert rep.metrics["mrr@2"] == pytest.approx((1.0 + 0.5) / 2)
        assert rep.metrics["recall@2"] == pytest.approx((1.0 + 0.5) / 2)
        assert rep.abstention_preview["auroc_top1"] == 1.0
        assert rep.by_category["in_domain_factoid"]["n"] == 2
        assert len(rep.per_item) == 3

    def test_requires_answerable_items(self):
        with pytest.raises(ValueError, match="answerable"):
            evaluate_retriever("toy", lambda q, k: [], [unanswerable()])


# ---------------------------------------------------------------------------
# Harness end to end
# ---------------------------------------------------------------------------


class TestHarness:
    def test_benchmark_all_baselines(self, offline):
        eval_set, chunks, _ = offline
        retriever = HybridRetriever(HashingEmbedder()).index(chunks)
        report = run_benchmark(eval_set, retriever, n_bootstrap=100)
        assert set(report["retrievers"]) == {"dense", "sparse", "hybrid"}
        for r in report["retrievers"].values():
            assert r["n_answerable"] == 70
            assert all(0.0 <= v <= 1.0 for v in r["metrics"].values())
            assert r["metrics"]["hit@10"] >= r["metrics"]["hit@1"]
            assert len(r["per_item"]) == 100
        md = render_markdown(report)
        assert "| Hybrid (RRF) |" in md and "hashing" in md and "AUROC" in md

    def test_cli_writes_json_and_markdown(self, tmp_path, offline, capsys):
        eval_set, _, corpus_path = offline
        set_path = eval_set.save(tmp_path / "eval_set.json")
        out = tmp_path / "bench.json"
        assert evaluate_main(["run", "--eval-set", str(set_path), "--corpus", str(corpus_path), "--baseline",
                              "--embedder", "hashing", "--bootstrap", "50", "--output", str(out)]) == 0
        report = json.loads(out.read_text())
        assert set(report["retrievers"]) == {"dense", "sparse", "hybrid"}
        assert out.with_suffix(".md").read_text().startswith("# Retrieval Benchmark")
        assert "wrote" in capsys.readouterr().out

    def test_cli_without_baseline_runs_hybrid_only(self, tmp_path, offline):
        eval_set, _, corpus_path = offline
        set_path = eval_set.save(tmp_path / "eval_set.json")
        out = tmp_path / "bench.json"
        evaluate_main(["run", "--eval-set", str(set_path), "--corpus", str(corpus_path),
                       "--embedder", "hashing", "--bootstrap", "20", "--output", str(out)])
        assert list(json.loads(out.read_text())["retrievers"]) == ["hybrid"]
