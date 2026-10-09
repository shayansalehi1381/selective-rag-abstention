"""Benchmark generator: builds the 100-item ground truth in ``data/eval/eval_set.json``.

Two engines share one output schema (``eval.schemas.EvalSet``):

* **Offline, deterministic** (``--offline``; CI and sandboxes). By default it builds a
  corpus of synthetic technical papers from a structured *fact table*, chunks it with
  the project's real splitter, and links every question to the exact chunk(s) that
  contain its evidence sentence. ``--corpus`` switches it to extractive templates over
  an existing chunk file (e.g. the arXiv corpus); that mode has lower fidelity.
* **LLM** (``--provider anthropic|openai``). It prompts a model for factoid, reasoning
  and adversarial unanswerable questions, keeps answerable items only if their evidence
  quote appears verbatim in the cited chunk, and runs an adversarial judge over the
  top-5 retrieved chunks to discard "unanswerable" questions that are in fact answerable.

Default mix: 50 factoid + 20 reasoning (answerable) and 10 out-of-domain + 10
unsupported-fact + 10 subtle-conflict (unanswerable), 100 items in total.

CLI::

    python -m eval.synthetic_gen generate --output data/eval/eval_set.json --offline
    python -m eval.synthetic_gen generate --output data/eval/eval_set.json --provider anthropic
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, ValidationError

from eval.schemas import (
    ABSTENTION_ANSWER,
    Category,
    CorpusInfo,
    EvalItem,
    EvalSet,
    ItemMetadata,
    corpus_fingerprint,
    count_categories,
    make_item_id,
)
from src.data_loader import (
    Chunk,
    RecursiveCharacterSplitter,
    chunk_document,
    load_chunks_jsonl,
    save_chunks_jsonl,
)
from src.llm import (  # noqa: F401  (re-exported for backwards compatibility)
    AnthropicClient,
    LLMClient,
    LLMRefusalError,
    OpenAICompatibleClient,
    extract_json,
)
from src.retriever import tokenize

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/eval/eval_set.json")
DEFAULT_SYNTHETIC_CORPUS = Path("data/eval/synthetic_corpus.jsonl")
OFFLINE_GENERATOR = "offline-template-v1"
EXTRACTIVE_GENERATOR = "offline-extractive-v1"


# ---------------------------------------------------------------------------
# Quotas and draft items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quotas:
    in_domain_factoid: int = 50
    in_domain_reasoning: int = 20
    out_of_domain: int = 10
    unsupported_fact: int = 10
    subtle_conflict: int = 10

    def __post_init__(self) -> None:
        if any(n < 0 for n in self.as_dict().values()):
            raise ValueError("quotas must be non-negative")
        if self.total == 0:
            raise ValueError("at least one item is required")

    def as_dict(self) -> dict[Category, int]:
        return {c: getattr(self, c.value) for c in Category}

    @property
    def total(self) -> int:
        return sum(self.as_dict().values())

    @property
    def answerable(self) -> int:
        return self.in_domain_factoid + self.in_domain_reasoning


@dataclass
class Draft:
    """An eval item before ids are assigned."""

    question: str
    category: Category
    ground_truth_chunk_ids: list[str]
    reference_answer: str
    metadata: dict[str, Any]

    @property
    def is_answerable(self) -> bool:
        return self.category in (Category.IN_DOMAIN_FACTOID, Category.IN_DOMAIN_REASONING)


def _take(candidates: Iterable[Draft], n: int, seen: set[str], category: Category) -> list[Draft]:
    """Take the first ``n`` drafts whose question text has not been used yet."""
    out: list[Draft] = []
    for draft in candidates:
        if len(out) == n:
            break
        key = draft.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(draft)
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} unique {category.value} items available, need {n}")
    return out


# ---------------------------------------------------------------------------
# Synthetic corpus: papers rendered from a fact table
# ---------------------------------------------------------------------------

_NAME_HEADS = ["Zeno", "Vexa", "Lumi", "Quor", "Talin", "Morv", "Ostra", "Pyre", "Nivo", "Kest",
               "Dral", "Sefi", "Ulmo", "Varn", "Eira", "Brisk", "Corvi", "Tamsa", "Helio", "Opal"]
_NAME_TAILS = ["Rank", "Fuse", "Gate", "Retr", "Calib", "Prob", "Mix", "Sift", "Lens", "Path"]
DATASETS = ["SciClaimQA", "BioHopQA", "LexiNLI", "CodeSeekBench", "MedFactCheck", "FinTabQA",
            "PatentRet", "ChronoNewsQA"]
METRICS = {
    "accuracy": ["accuracy", "classification accuracy"],
    "macro-F1": ["macro-F1", "macro-averaged F1"],
    "exact match": ["exact match", "EM score"],
    "nDCG@10": ["nDCG@10", "ranking quality (nDCG@10)"],
    "Recall@20": ["Recall@20", "recall at 20"],
}
BASELINES = ["BM25", "DPR", "ColBERT", "Contriever", "SPLADE", "ANCE"]
ENCODERS = ["DeBERTa-v3", "RoBERTa", "MiniLM", "ELECTRA", "BERT-base", "T5-encoder"]
TOPICS = [
    "Calibrated Evidence Retrieval for {domain}",
    "Uncertainty-Aware Dense Retrieval in {domain}",
    "Selective Answering over {domain} Corpora",
    "Late-Interaction Reranking for {domain}",
    "Learning When to Abstain in {domain} Question Answering",
    "Hybrid Sparse-Dense Search for {domain}",
]
DOMAINS = ["Scientific Claims", "Biomedical Literature", "Legal Documents", "Source Code",
           "Clinical Notes", "Financial Tables", "Patents", "Temporal News"]

FILLER = [
    "Retrieval-augmented systems often fail silently when the supporting evidence is missing.",
    "Prior work has largely evaluated retrievers on clean benchmarks with a single relevant passage.",
    "We argue that robust systems must expose calibrated confidence to downstream components.",
    "Our design keeps the retriever lightweight so that it can be deployed on commodity hardware.",
    "We analyse failure cases and find that most errors stem from lexical mismatch between query and passage.",
    "Ablations show that each component contributes to the final performance.",
    "We release our code and evaluation scripts to support reproducibility.",
    "The approach is complementary to reranking and can be combined with any generator.",
    "Qualitative inspection suggests that retrieved passages are more focused than those of prior systems.",
    "We leave the extension to multilingual settings for future work.",
    "Negative sampling strategy has a noticeable effect on convergence speed.",
    "All experiments use the official splits released by the benchmark authors.",
    "Hyperparameters were selected on the development split only.",
    "Error analysis reveals that long multi-part questions remain challenging.",
    "These findings indicate that confidence estimation and retrieval quality are tightly coupled.",
    "We report the mean over three runs with different random initialisations.",
]

# Off-domain topics; the generator keeps only those sharing no content token with the corpus.
OOD_QUESTIONS = [
    "What is the ideal rotation schedule for alpine sheep pastures?",
    "How long should sourdough starter ferment before baking rye loaves?",
    "Which tectonic plates collided to form the Himalayan mountain range?",
    "What tax bracket applies to freelance photographers in Portugal?",
    "Who composed the baroque cantata Jauchzet Gott in allen Landen?",
    "How many wickets did the bowler take in the 1932 Bodyline series?",
    "What glaze temperature suits porcelain stoneware in a wood kiln?",
    "Which mortgage refinancing option minimises closing fees for retirees?",
    "How do you fold a traditional origami crane from washi paper?",
    "What caused the eruption of Mount Pinatubo in 1991?",
    "Which grape varieties dominate Rioja wine blends?",
    "How should beekeepers treat varroa mites during winter?",
    "What is the migration route of Arctic terns each year?",
    "Which chess opening did Capablanca favour against Alekhine?",
    "How deep must tulip bulbs be planted in clay soil?",
    "What were the main causes of the Peloponnesian War?",
    "Which yoga posture relieves lower back stiffness after cycling?",
    "How is saffron harvested from crocus flowers in Kashmir?",
    "What penalty applies to offside offences in rugby union?",
    "Which telescope first photographed the rings of Uranus?",
]
ABLATED_COMPONENTS = ["calibration head", "gating module", "hard-negative mining", "query expansion",
                      "late-interaction layer", "contrastive pre-training"]
UNSTATED_ATTRIBUTES = [
    ("What dropout rate does {M} use?", "dropout"),
    ("How many GPUs were needed to train {M}?", "gpus"),
    ("What weight decay is applied when optimising {M}?", "weight_decay"),
    ("How large is the vocabulary of the {M} tokenizer?", "vocabulary"),
]


@dataclass
class Fact:
    fact_id: str
    arxiv_id: str
    kind: str  # result | baseline | encoder | layers | hidden | epochs | lr | batch
    fields: dict[str, Any]
    sentence: str
    start: int = -1
    end: int = -1
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class SyntheticPaper:
    arxiv_id: str
    title: str
    method: str
    metric: str
    datasets: list[str]
    baselines: list[str]
    text: str
    facts: list[Fact]


def _fmt(value: float) -> str:
    return f"{value:.1f}"


class SyntheticCorpusFactory:
    """Deterministically renders papers whose every quantitative claim is a known fact."""

    def __init__(self, seed: int = 42, n_papers: int = 16) -> None:
        if not 1 <= n_papers <= len(_NAME_HEADS):
            raise ValueError(f"n_papers must be in [1, {len(_NAME_HEADS)}]")
        self.rng = random.Random(seed)
        self.n_papers = n_papers

    def _method_names(self) -> list[str]:
        heads = self.rng.sample(_NAME_HEADS, self.n_papers)
        return [h + self.rng.choice(_NAME_TAILS) for h in heads]

    def build(self) -> list[SyntheticPaper]:
        papers = []
        for idx, method in enumerate(self._method_names(), start=1):
            papers.append(self._paper(idx, method))
        return papers

    def _paper(self, idx: int, method: str) -> SyntheticPaper:
        rng = self.rng
        arxiv_id = f"synth-{idx:04d}"
        metric = rng.choice(sorted(METRICS))
        datasets = rng.sample(DATASETS, 3)
        baselines = rng.sample(BASELINES, 2)
        domain = rng.choice(DOMAINS)
        title = f"{method}: " + rng.choice(TOPICS).format(domain=domain)
        facts: list[Fact] = []

        def fact(kind: str, sentence: str, **fields: Any) -> str:
            facts.append(Fact(f"{arxiv_id}/{len(facts):02d}", arxiv_id, kind, fields, sentence))
            return sentence

        encoder = rng.choice(ENCODERS)
        layers = rng.choice([6, 8, 12, 16, 24])
        hidden = rng.choice([384, 512, 768, 1024])
        epochs = rng.choice([3, 5, 8, 10, 20, 30])
        lr = rng.choice(["1e-5", "2e-5", "3e-5", "5e-5", "1e-4"])
        batch = rng.choice([16, 32, 64, 128, 256])

        method_par = " ".join([
            fact("encoder", rng.choice([
                f"{method} is built on top of a {encoder} encoder.",
                f"The backbone of {method} is a pretrained {encoder} encoder.",
            ]), encoder=encoder),
            rng.choice(FILLER),
            fact("layers", f"The encoder stack of {method} has {layers} transformer layers.", layers=layers),
            fact("hidden", f"Each layer of {method} uses a hidden size of {hidden}.", hidden=hidden),
            rng.choice(FILLER),
        ])
        training_par = " ".join([
            fact("epochs", f"We train {method} for {epochs} epochs.", epochs=epochs),
            fact("lr", f"Optimisation of {method} uses AdamW with a learning rate of {lr}.", lr=lr),
            fact("batch", f"Training batches for {method} contain {batch} examples.", batch=batch),
            rng.choice(FILLER),
        ])

        # Results: the test-set number is the fact; dev-split and ablation numbers on the
        # same dataset are realistic hard negatives (same entities, different value).
        result_sentences, ablation_sentences = [], []
        for dataset in datasets:
            ours = round(rng.uniform(45.0, 92.0), 1)
            delta = round(rng.uniform(1.2, 9.5), 1)
            dev = round(ours + rng.uniform(0.6, 3.5), 1)
            ablated = round(ours - rng.uniform(1.5, 7.0), 1)
            base_name = rng.choice(baselines)
            theirs = round(ours - delta, 1)
            result_sentences.append(fact("result", rng.choice([
                f"On the {dataset} test set, the full {method} model attains {_fmt(ours)} {metric}.",
                f"The full {method} model reaches {_fmt(ours)} {metric} on the {dataset} test set.",
                f"For the {dataset} test split, we measure {_fmt(ours)} {metric} with the full {method} model.",
            ]), method=method, dataset=dataset, metric=metric, value=ours))
            result_sentences.append(fact("baseline", rng.choice([
                f"The {base_name} baseline obtains {_fmt(theirs)} {metric} on the {dataset} test set.",
                f"In comparison, {base_name} scores {_fmt(theirs)} {metric} on the {dataset} test set.",
            ]), method=method, baseline=base_name, dataset=dataset, metric=metric, value=theirs))
            result_sentences.append(
                f"On the {dataset} development split, {method} records {_fmt(dev)} {metric}.")
            component = rng.choice(ABLATED_COMPONENTS)
            ablation_sentences.append(
                f"Without its {component}, {method} drops to {_fmt(ablated)} {metric} on {dataset}.")
            ablation_sentences.append(rng.choice(FILLER))
        others = [d for d in DATASETS if d not in datasets]
        related = " ".join([
            f"Earlier retrievers such as {baselines[0]} and {baselines[1]} were tuned on "
            f"{others[0]} and {others[1]}.",
            rng.choice(FILLER),
            f"Unlike systems built for {others[2]}, {method} targets {domain.lower()}.",
            rng.choice(FILLER),
        ])

        abstract = " ".join([
            f"We present {method}, a retrieval model for {domain.lower()}.",
            rng.choice(FILLER), rng.choice(FILLER),
            f"{method} is evaluated on {', '.join(datasets[:-1])} and {datasets[-1]}.",
        ])
        sections = [
            title,
            "Abstract. " + abstract,
            "1 Introduction. " + " ".join(rng.sample(FILLER, 4)),
            "2 Related Work. " + related,
            "3 Method. " + method_par,
            "4 Training Setup. " + training_par,
            "5 Results. " + " ".join(result_sentences),
            "6 Ablations. " + " ".join(ablation_sentences),
            "7 Conclusion. " + " ".join(rng.sample(FILLER, 3)),
        ]
        text = "\n\n".join(sections)
        cursor = 0
        for f in facts:  # facts appear in text order; search forward to get exact spans
            f.start = text.index(f.sentence, cursor)
            f.end = cursor = f.start + len(f.sentence)
        return SyntheticPaper(arxiv_id, title, method, metric, datasets, baselines, text, facts)


def chunk_papers(papers: Sequence[SyntheticPaper], splitter: RecursiveCharacterSplitter | None = None) -> list[Chunk]:
    """Chunk with the project splitter and link each fact to every chunk fully containing it."""
    splitter = splitter or RecursiveCharacterSplitter(512, 64)
    chunks: list[Chunk] = []
    for paper in papers:
        doc_chunks = chunk_document(paper.text, arxiv_id=paper.arxiv_id, title=paper.title, splitter=splitter)
        for f in paper.facts:
            f.chunk_ids = [c.chunk_id for c in doc_chunks if c.char_start <= f.start and f.end <= c.char_end]
        chunks.extend(doc_chunks)
    return chunks


def corpus_vocabulary(chunks: Iterable[Chunk]) -> set[str]:
    vocab: set[str] = set()
    for c in chunks:
        vocab.update(tokenize(c.text))
        vocab.update(tokenize(c.title))
    return vocab


def _invented_names(rng: random.Random, vocab: set[str], n: int, suffixes: Sequence[str]) -> list[str]:
    """Plausible names guaranteed absent from the corpus (hallucinated entities)."""
    heads = ["Gral", "Hexo", "Jorvi", "Plinth", "Wexa", "Yltra", "Cobal", "Faro", "Istri", "Rondo"]
    tails = list(suffixes)
    names = [h + t for h in heads for t in tails]
    rng.shuffle(names)
    out = [nm for nm in names if not set(tokenize(nm)) & vocab]
    if len(out) < n:
        raise RuntimeError("not enough out-of-corpus names")
    return out[:n]


def _ood_drafts(vocab: set[str], generator: str) -> list[Draft]:
    drafts = []
    for q in OOD_QUESTIONS:
        if set(tokenize(q)) & vocab:
            continue  # shares a content word with the corpus: not cleanly out-of-domain
        drafts.append(Draft(q, Category.OUT_OF_DOMAIN, [], ABSTENTION_ANSWER,
                            {"difficulty": "easy", "generator": generator, "template": "ood_fixed"}))
    return drafts


class OfflineTemplateEngine:
    """Template questions over a synthetic fact table. Evidence is exact by construction."""

    def __init__(self, papers: Sequence[SyntheticPaper], chunks: Sequence[Chunk], rng: random.Random) -> None:
        self.papers = list(papers)
        self.chunks = list(chunks)
        self.rng = rng
        self.vocab = corpus_vocabulary(chunks)

    # -- answerable ----------------------------------------------------------------

    def _factoid_candidates(self) -> list[Draft]:
        rng, out = self.rng, []
        for p in self.papers:
            m = p.method
            for f in p.facts:
                if not f.chunk_ids:
                    continue  # sentence straddles a chunk boundary: no exact evidence
                fl = f.fields
                syn = rng.choice(METRICS.get(fl.get("metric", ""), [""]))
                templates: dict[str, list[tuple[str, str]]] = {
                    "result": [
                        (f"What test-set {syn} does the full {m} model achieve on {fl.get('dataset')}?", "medium"),
                        (f"How well does the complete {m} system do on the {fl.get('dataset')} test split "
                         f"in terms of {syn}?", "hard"),
                    ],
                    "baseline": [
                        (f"In the {m} paper, what test-set {syn} is reported for the {fl.get('baseline')} "
                         f"baseline on {fl.get('dataset')}?", "medium"),
                    ],
                    "encoder": [(f"Which pretrained backbone is {m} based on?", "easy")],
                    "layers": [(f"How deep is the encoder of {m}, in transformer layers?", "easy")],
                    "hidden": [(f"What is the hidden dimension of each {m} layer?", "medium")],
                    "epochs": [(f"For how many epochs is {m} trained?", "easy")],
                    "lr": [(f"Which learning rate is used to optimise {m}?", "easy")],
                    "batch": [(f"What batch size is used when training {m}?", "medium")],
                }
                question, difficulty = rng.choice(templates[f.kind])
                answer = {
                    "result": f"{_fmt(fl.get('value', 0))} {fl.get('metric')}",
                    "baseline": f"{_fmt(fl.get('value', 0))} {fl.get('metric')}",
                    "encoder": str(fl.get("encoder")),
                    "layers": f"{fl.get('layers')} transformer layers",
                    "hidden": f"a hidden size of {fl.get('hidden')}",
                    "epochs": f"{fl.get('epochs')} epochs",
                    "lr": f"a learning rate of {fl.get('lr')}",
                    "batch": f"{fl.get('batch')} examples per batch",
                }[f.kind]
                out.append(Draft(question, Category.IN_DOMAIN_FACTOID, list(f.chunk_ids), answer, {
                    "arxiv_id": p.arxiv_id, "difficulty": difficulty, "generator": OFFLINE_GENERATOR,
                    "template": f"factoid_{f.kind}", "evidence_sentences": [f.sentence],
                }))
        rng.shuffle(out)
        return out

    def _reasoning_candidates(self) -> list[Draft]:
        rng, out = self.rng, []
        for p in self.papers:
            m = p.method
            results = {f.fields["dataset"]: f for f in p.facts if f.kind == "result" and f.chunk_ids}
            baselines = {f.fields["dataset"]: f for f in p.facts if f.kind == "baseline" and f.chunk_ids}
            for d, ours in results.items():
                theirs = baselines.get(d)
                if theirs is None:
                    continue
                gap = ours.fields["value"] - theirs.fields["value"]
                b = theirs.fields["baseline"]
                out.append(self._reasoning(
                    p, [ours, theirs], "reasoning_margin",
                    f"By how many points does the full {m} model beat {b} on the {d} test set, "
                    f"measured in {p.metric}?",
                    f"{_fmt(gap)} points ({_fmt(ours.fields['value'])} vs {_fmt(theirs.fields['value'])} {p.metric})",
                ))
            names = sorted(results)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b2 = results[names[i]], results[names[j]]
                    hi, lo = (a, b2) if a.fields["value"] >= b2.fields["value"] else (b2, a)
                    out.append(self._reasoning(
                        p, [a, b2], "reasoning_compare_datasets",
                        f"Is the test-set {p.metric} of the full {m} model higher on {names[i]} or on "
                        f"{names[j]}, and by how much?",
                        f"{hi.fields['dataset']} ({_fmt(hi.fields['value'])} vs {_fmt(lo.fields['value'])}, "
                        f"a gap of {_fmt(hi.fields['value'] - lo.fields['value'])} {p.metric})",
                    ))
        rng.shuffle(out)
        return out

    def _reasoning(self, p: SyntheticPaper, facts: list[Fact], template: str, q: str, a: str) -> Draft:
        gold = list(dict.fromkeys(cid for f in facts for cid in f.chunk_ids))
        return Draft(q, Category.IN_DOMAIN_REASONING, gold, a, {
            "arxiv_id": p.arxiv_id, "difficulty": "hard", "generator": OFFLINE_GENERATOR,
            "template": template, "evidence_sentences": [f.sentence for f in facts],
        })

    # -- unanswerable --------------------------------------------------------------

    def _chunks_mentioning(self, term: str, limit: int = 5) -> list[str]:
        toks = set(tokenize(term))
        return [c.chunk_id for c in self.chunks if toks <= set(tokenize(c.text))][:limit]

    def _unsupported_candidates(self) -> list[Draft]:
        rng, out = self.rng, []
        fake_methods = _invented_names(rng, self.vocab, len(self.papers), ["Rank", "Fuse", "Sift"])
        for p, fake in zip(self.papers, fake_methods):
            d = rng.choice(p.datasets)
            out.append(self._unsupported(
                p, f"What test-set {p.metric} does {fake} achieve on {d}?", "unsupported_hallucinated_method",
                "medium", self._chunks_mentioning(d), hallucinated_entity=fake))
            unseen = [d2 for d2 in DATASETS if d2 not in p.datasets]
            d2 = rng.choice(unseen)
            result_chunks = [cid for f in p.facts if f.kind == "result" for cid in f.chunk_ids]
            out.append(self._unsupported(
                p, f"What test-set {p.metric} does the full {p.method} model achieve on {d2}?", "unsupported_unevaluated_dataset",
                "hard", list(dict.fromkeys(result_chunks))[:5]))
            q, attr = rng.choice(UNSTATED_ATTRIBUTES)
            arch_chunks = [cid for f in p.facts if f.kind in ("layers", "hidden", "epochs", "lr")
                           for cid in f.chunk_ids]
            out.append(self._unsupported(
                p, q.format(M=p.method), f"unsupported_unstated_{attr}", "hard",
                list(dict.fromkeys(arch_chunks))[:5]))
        rng.shuffle(out)
        return out

    def _unsupported(self, p: SyntheticPaper, q: str, template: str, difficulty: str,
                     distractors: list[str], **extra: Any) -> Draft:
        return Draft(q, Category.UNSUPPORTED_FACT, [], ABSTENTION_ANSWER, {
            "arxiv_id": p.arxiv_id, "difficulty": difficulty, "generator": OFFLINE_GENERATOR,
            "template": template, "distractor_chunk_ids": distractors, **extra,
        })

    def _conflict_candidates(self) -> list[Draft]:
        rng, out = self.rng, []
        for p in self.papers:
            m = p.method
            res = {f.fields["dataset"]: f for f in p.facts if f.kind == "result" and f.chunk_ids}
            base = {f.fields["dataset"]: f for f in p.facts if f.kind == "baseline" and f.chunk_ids}
            for d, ours in res.items():
                if d in base:
                    b = base[d].fields["baseline"]
                    out.append(self._conflict(
                        p, f"Why does {m} fall behind {b} on {d}?", "conflict_reversed_comparison",
                        ours.chunk_ids + base[d].chunk_ids, f"{m} outperforms {b} on {d}"))
                wrong = round(ours.fields["value"] * rng.choice([0.55, 0.65, 0.75]), 1)
                out.append(self._conflict(
                    p, f"Given that the full {m} model only reaches {_fmt(wrong)} {p.metric} on the {d} test set, "
                       f"which component limits its performance?",
                    "conflict_wrong_value", ours.chunk_ids,
                    f"{m} reaches {_fmt(ours.fields['value'])}, not {_fmt(wrong)}"))
            ep = next((f for f in p.facts if f.kind == "epochs" and f.chunk_ids), None)
            if ep:
                wrong_ep = ep.fields["epochs"] * 4
                out.append(self._conflict(
                    p, f"Why was {m} trained for as many as {wrong_ep} epochs?", "conflict_wrong_training",
                    ep.chunk_ids, f"{m} is trained for {ep.fields['epochs']} epochs"))
        rng.shuffle(out)
        return out

    def _conflict(self, p: SyntheticPaper, q: str, template: str, distractors: list[str], truth: str) -> Draft:
        return Draft(q, Category.SUBTLE_CONFLICT, [], ABSTENTION_ANSWER, {
            "arxiv_id": p.arxiv_id, "difficulty": "hard", "generator": OFFLINE_GENERATOR,
            "template": template, "distractor_chunk_ids": list(dict.fromkeys(distractors)),
            "contradicted_claim": truth,
        })

    def generate(self, quotas: Quotas) -> list[Draft]:
        seen: set[str] = set()
        builders: dict[Category, Callable[[], list[Draft]]] = {
            Category.IN_DOMAIN_FACTOID: self._factoid_candidates,
            Category.IN_DOMAIN_REASONING: self._reasoning_candidates,
            Category.OUT_OF_DOMAIN: lambda: _ood_drafts(self.vocab, OFFLINE_GENERATOR),
            Category.UNSUPPORTED_FACT: self._unsupported_candidates,
            Category.SUBTLE_CONFLICT: self._conflict_candidates,
        }
        drafts: list[Draft] = []
        for category, n in quotas.as_dict().items():
            if n:
                drafts.extend(_take(builders[category](), n, seen, category))
        return drafts


# ---------------------------------------------------------------------------
# Offline extractive engine over an existing (e.g. arXiv) chunk corpus
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")  # keeps "71.2" inside its sentence
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(%?)(?!\w|\.\d)")


class ExtractiveCorpusEngine:
    """Cloze-style questions from real chunks. Lower fidelity than the LLM engine:
    questions copy much of the source wording, which favours lexical retrieval."""

    def __init__(self, chunks: Sequence[Chunk], rng: random.Random) -> None:
        self.chunks = list(chunks)
        self.rng = rng
        self.vocab = corpus_vocabulary(chunks)
        self.by_paper: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            self.by_paper.setdefault(c.arxiv_id, []).append(c)

    def _numeric_sentences(self) -> list[tuple[Chunk, str, re.Match]]:
        found = []
        for c in self.chunks:
            for raw in _SENTENCE_SPLIT_RE.split(c.text):
                s = " ".join(raw.split())
                num = _NUMBER_RE.search(s)
                if 60 <= len(s) <= 260 and num and len(tokenize(s)) >= 6:
                    found.append((c, s, num))
        return found

    def _gold(self, chunk: Chunk, sentence: str) -> list[str]:
        norm = lambda t: " ".join(t.split())  # noqa: E731
        return [c.chunk_id for c in self.by_paper[chunk.arxiv_id] if sentence in norm(c.text)] or [chunk.chunk_id]

    def generate(self, quotas: Quotas) -> list[Draft]:
        rng = self.rng
        sents = self._numeric_sentences()
        rng.shuffle(sents)
        meta = lambda c, t, d, **kw: {"arxiv_id": c.arxiv_id, "difficulty": d,  # noqa: E731
                                      "generator": EXTRACTIVE_GENERATOR, "template": t, **kw}
        factoid, conflict = [], []
        for c, s, num in sents:
            masked = s[:num.start()] + "___" + s[num.end():]
            factoid.append(Draft(f"According to \"{c.title}\", what value fills the blank: \"{masked}\"?",
                                 Category.IN_DOMAIN_FACTOID, self._gold(c, s), num.group(0),
                                 meta(c, "extractive_cloze", "medium", evidence_sentences=[s])))
            wrong = f"{float(num.group(1)) * 1.7 + 3:.1f}{num.group(2)}"
            altered = s[:num.start()] + wrong + s[num.end():]
            conflict.append(Draft(f"The paper \"{c.title}\" states: \"{altered}\" What explains this result?",
                                  Category.SUBTLE_CONFLICT, [], ABSTENTION_ANSWER,
                                  meta(c, "extractive_wrong_value", "hard", distractor_chunk_ids=self._gold(c, s),
                                       contradicted_claim=s)))
        reasoning = []
        pairs: dict[str, list[tuple[Chunk, str, re.Match]]] = {}
        for item in sents:
            pairs.setdefault(item[0].arxiv_id, []).append(item)
        for items in pairs.values():
            for (c1, s1, n1), (c2, s2, n2) in zip(items[::2], items[1::2]):
                v1, v2 = float(n1.group(1)), float(n2.group(1))
                if v1 == v2:
                    continue
                m1 = s1[:n1.start()] + "___" + s1[n1.end():]
                m2 = s2[:n2.start()] + "___" + s2[n2.end():]
                hi = "the first" if v1 > v2 else "the second"
                reasoning.append(Draft(
                    f"In \"{c1.title}\", which blank holds the larger value: (1) \"{m1}\" or (2) \"{m2}\"?",
                    Category.IN_DOMAIN_REASONING, list(dict.fromkeys(self._gold(c1, s1) + self._gold(c2, s2))),
                    f"{hi} ({n1.group(0)} vs {n2.group(0)})",
                    meta(c1, "extractive_compare", "hard", evidence_sentences=[s1, s2])))
        fakes = _invented_names(rng, self.vocab, max(quotas.unsupported_fact, 1), ["Net", "Former", "RAG"])
        titles = sorted({c.title for c in self.chunks})
        unsupported = [Draft(f"What results does \"{rng.choice(titles)}\" report for {fake}?",
                             Category.UNSUPPORTED_FACT, [], ABSTENTION_ANSWER,
                             {"difficulty": "medium", "generator": EXTRACTIVE_GENERATOR,
                              "template": "extractive_hallucinated_entity", "hallucinated_entity": fake})
                       for fake in fakes]
        seen: set[str] = set()
        pools = {Category.IN_DOMAIN_FACTOID: factoid, Category.IN_DOMAIN_REASONING: reasoning,
                 Category.OUT_OF_DOMAIN: _ood_drafts(self.vocab, EXTRACTIVE_GENERATOR),
                 Category.UNSUPPORTED_FACT: unsupported, Category.SUBTLE_CONFLICT: conflict}
        drafts: list[Draft] = []
        for category, n in quotas.as_dict().items():
            if n:
                drafts.extend(_take(pools[category], n, seen, category))
        return drafts


# ---------------------------------------------------------------------------
# LLM engine
# ---------------------------------------------------------------------------


PROMPT_VERSION = "v1"
SYSTEM_PROMPT = (
    "You build evaluation data for a retrieval-augmented QA system over research papers. "
    "Follow the instructions exactly and answer with a single JSON object and nothing else."
)
FACTOID_PROMPT = """Write one factoid question that is answerable ONLY from the passage below.
Paraphrase: do not copy long phrases from the passage. Name the specific method, dataset or quantity asked about.
Return JSON: {{"question": str, "answer": str, "evidence_quote": str}}.
"evidence_quote" must be copied verbatim from the passage and must contain the answer.

Paper: {title}
Passage:
{text}"""
REASONING_PROMPT = """Write one question that requires combining BOTH passages (e.g. a comparison or a difference).
Return JSON: {{"question": str, "answer": str, "evidence_quotes": [str, str]}}.
Each evidence quote must be copied verbatim from passage 1 and passage 2 respectively.

Paper: {title}
Passage 1:
{text1}

Passage 2:
{text2}"""
UNSUPPORTED_PROMPT = """Write one plausible, specific question on the same topic as the passage that the passage
and paper CANNOT answer: ask about a detail, setting or entity that is never mentioned.
Return JSON: {{"question": str}}.

Paper: {title}
Passage:
{text}"""
CONFLICT_PROMPT = """Write one question whose premise subtly CONTRADICTS a claim in the passage
(e.g. swaps a comparison or changes a number), so that it cannot be answered as asked.
Return JSON: {{"question": str, "contradicted_claim": str}}.

Paper: {title}
Passage:
{text}"""
OOD_PROMPT = """Write one specific factual question about the topic "{topic}". It must be unrelated to machine
learning, information retrieval and computer science. Return JSON: {{"question": str}}."""
JUDGE_PROMPT = """Can the question be answered correctly using ONLY the passages below?
Answer "answerable": true only if a passage states the needed information explicitly.
A question with a false premise is NOT answerable.
Return JSON: {{"answerable": bool, "chunk_id": str or null, "rationale": str}}.

Question: {question}

{passages}"""
OOD_TOPICS = ["alpine farming", "baroque music", "volcanology", "beekeeping", "maritime law",
              "medieval architecture", "rugby rules", "wine making", "ornithology", "pottery glazes",
              "tax law in Portugal", "Himalayan geology", "origami", "chess history", "sourdough baking"]


class _QA(BaseModel):
    question: str
    answer: str
    evidence_quote: str


class _Reasoning(BaseModel):
    question: str
    answer: str
    evidence_quotes: list[str]


class _Question(BaseModel):
    question: str
    contradicted_claim: str | None = None


class _Judgement(BaseModel):
    answerable: bool
    chunk_id: str | None = None
    rationale: str = ""


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class LLMEngine:
    """LLM-generated items with verbatim-evidence filtering and an adversarial judge."""

    def __init__(
        self,
        client: LLMClient,
        chunks: Sequence[Chunk],
        rng: random.Random,
        *,
        retrieve: Callable[[str, int], list[Chunk]] | None = None,
        temperature: float = 0.7,
        max_attempts_per_item: int = 4,
        max_tokens: int = 2048,
    ) -> None:
        self.client = client
        self.chunks = [c for c in chunks if len(c.text) >= 200] or list(chunks)
        self.rng = rng
        self.retrieve = retrieve
        self.temperature = temperature
        self.max_attempts_per_item = max_attempts_per_item
        self.max_tokens = max_tokens
        self.by_paper: dict[str, list[Chunk]] = {}
        for c in self.chunks:
            self.by_paper.setdefault(c.arxiv_id, []).append(c)
        self.rejections: dict[str, int] = {}

    def _ask(self, prompt: str, model: type[BaseModel], temperature: float | None) -> BaseModel:
        raw = self.client.complete(SYSTEM_PROMPT, prompt, temperature=temperature, max_tokens=self.max_tokens)
        return model.model_validate(extract_json(raw))

    def _reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _meta(self, chunk: Chunk | None, template: str, difficulty: str, **extra: Any) -> dict[str, Any]:
        return {"arxiv_id": chunk.arxiv_id if chunk else None, "difficulty": difficulty,
                "generator": f"llm:{self.client.name}", "template": f"{template}_{PROMPT_VERSION}", **extra}

    def _factoid(self) -> Draft | None:
        c = self.rng.choice(self.chunks)
        out = self._ask(FACTOID_PROMPT.format(title=c.title, text=c.text), _QA, self.temperature)
        if _norm(out.evidence_quote) not in _norm(c.text) or not out.answer.strip():
            return self._reject("evidence_not_verbatim")
        return Draft(out.question, Category.IN_DOMAIN_FACTOID, [c.chunk_id], out.answer,
                     self._meta(c, "llm_factoid", "medium", evidence_sentences=[out.evidence_quote]))

    def _reasoning(self) -> Draft | None:
        multi = [p for p, cs in self.by_paper.items() if len(cs) >= 2]
        if not multi:
            return self._reject("no_multi_chunk_paper")
        c1, c2 = self.rng.sample(self.by_paper[self.rng.choice(sorted(multi))], 2)
        out = self._ask(REASONING_PROMPT.format(title=c1.title, text1=c1.text, text2=c2.text),
                        _Reasoning, self.temperature)
        quotes = out.evidence_quotes
        if (len(quotes) != 2 or _norm(quotes[0]) not in _norm(c1.text)
                or _norm(quotes[1]) not in _norm(c2.text) or not out.answer.strip()):
            return self._reject("evidence_not_verbatim")
        return Draft(out.question, Category.IN_DOMAIN_REASONING, [c1.chunk_id, c2.chunk_id], out.answer,
                     self._meta(c1, "llm_reasoning", "hard", evidence_sentences=quotes))

    def _unanswerable(self, category: Category) -> Draft | None:
        chunk = None
        if category is Category.OUT_OF_DOMAIN:
            out = self._ask(OOD_PROMPT.format(topic=self.rng.choice(OOD_TOPICS)), _Question, self.temperature)
            template, difficulty = "llm_ood", "easy"
        else:
            chunk = self.rng.choice(self.chunks)
            prompt = UNSUPPORTED_PROMPT if category is Category.UNSUPPORTED_FACT else CONFLICT_PROMPT
            out = self._ask(prompt.format(title=chunk.title, text=chunk.text), _Question, self.temperature)
            template = "llm_unsupported" if category is Category.UNSUPPORTED_FACT else "llm_conflict"
            difficulty = "hard"
        distractors: list[str] = []
        if self.retrieve is not None:
            top = self.retrieve(out.question, 5)
            distractors = [c.chunk_id for c in top]
            passages = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in top)
            verdict = self._ask(JUDGE_PROMPT.format(question=out.question, passages=passages), _Judgement, 0.0)
            if verdict.answerable:
                return self._reject("judge_says_answerable")
        extra: dict[str, Any] = {"distractor_chunk_ids": distractors, "verified": self.retrieve is not None}
        if out.contradicted_claim:
            extra["contradicted_claim"] = out.contradicted_claim
        return Draft(out.question, category, [], ABSTENTION_ANSWER, self._meta(chunk, template, difficulty, **extra))

    def generate(self, quotas: Quotas) -> list[Draft]:
        makers: dict[Category, Callable[[], Draft | None]] = {
            Category.IN_DOMAIN_FACTOID: self._factoid,
            Category.IN_DOMAIN_REASONING: self._reasoning,
            Category.OUT_OF_DOMAIN: lambda: self._unanswerable(Category.OUT_OF_DOMAIN),
            Category.UNSUPPORTED_FACT: lambda: self._unanswerable(Category.UNSUPPORTED_FACT),
            Category.SUBTLE_CONFLICT: lambda: self._unanswerable(Category.SUBTLE_CONFLICT),
        }
        seen: set[str] = set()
        drafts: list[Draft] = []
        for category, n in quotas.as_dict().items():
            got, attempts = 0, 0
            while got < n:
                if attempts >= n * self.max_attempts_per_item:
                    raise RuntimeError(f"could not fill {category.value}: {got}/{n} after {attempts} attempts "
                                       f"(rejections: {self.rejections})")
                attempts += 1
                try:
                    draft = makers[category]()
                except (ValueError, ValidationError, LLMRefusalError) as exc:
                    self._reject(type(exc).__name__)
                    continue
                if draft is None or draft.question.strip().lower() in seen:
                    continue
                seen.add(draft.question.strip().lower())
                drafts.append(draft)
                got += 1
            logger.info("%s: %d items in %d attempts", category.value, got, attempts)
        return drafts


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_eval_set(
    drafts: Sequence[Draft],
    *,
    chunks: Sequence[Chunk],
    corpus_path: Path | str,
    generator: str,
    seed: int | None,
    rng: random.Random,
) -> EvalSet:
    """Shuffle (so categories interleave), assign ``eval_001``..., validate everything."""
    order = list(drafts)
    rng.shuffle(order)
    items = [
        EvalItem(
            id=make_item_id(i),
            question=d.question,
            is_answerable=d.is_answerable,
            category=d.category,
            ground_truth_chunk_ids=d.ground_truth_chunk_ids,
            reference_answer=d.reference_answer,
            metadata=ItemMetadata(**d.metadata),
        )
        for i, d in enumerate(order, start=1)
    ]
    eval_set = EvalSet(
        generator=generator,
        seed=seed,
        corpus=CorpusInfo(path=Path(corpus_path).as_posix(), num_chunks=len(chunks),
                          sha256=corpus_fingerprint(chunks)),
        counts=count_categories(items),
        items=items,
    )
    eval_set.validate_against_corpus(c.chunk_id for c in chunks)
    return eval_set


def build_offline_eval_set(
    *,
    seed: int = 42,
    quotas: Quotas = Quotas(),
    corpus_path: Path | str | None = None,
    synthetic_corpus_out: Path | str = DEFAULT_SYNTHETIC_CORPUS,
    n_papers: int = 16,
) -> tuple[EvalSet, list[Chunk]]:
    """Deterministic set: synthetic corpus by default, extractive over ``corpus_path`` if given."""
    rng = random.Random(seed)
    if corpus_path is None:
        papers = SyntheticCorpusFactory(seed, n_papers).build()
        chunks = chunk_papers(papers)
        save_chunks_jsonl(chunks, synthetic_corpus_out)
        drafts = OfflineTemplateEngine(papers, chunks, rng).generate(quotas)
        path, generator = synthetic_corpus_out, OFFLINE_GENERATOR
    else:
        chunks = load_chunks_jsonl(corpus_path)
        drafts = ExtractiveCorpusEngine(chunks, rng).generate(quotas)
        path, generator = corpus_path, EXTRACTIVE_GENERATOR
    return assemble_eval_set(drafts, chunks=chunks, corpus_path=path, generator=generator,
                             seed=seed, rng=rng), chunks


def summarize(eval_set: EvalSet) -> str:
    lines = [f"{'category':<22}{'n':>5}"]
    lines += [f"{name:<22}{n:>5}" for name, n in eval_set.counts.items()]
    lines.append(f"{'total':<22}{len(eval_set.items):>5}  "
                 f"(answerable {len(eval_set.answerable)}, unanswerable {len(eval_set.unanswerable)})")
    lines.append(f"corpus {eval_set.corpus.path}: {eval_set.corpus.num_chunks} chunks, "
                 f"sha256 {eval_set.corpus.sha256[:12]}…")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.synthetic_gen", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="build the evaluation set")
    gen.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    gen.add_argument("--offline", action="store_true", help="deterministic template engine (no LLM)")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--corpus", type=Path, default=None,
                     help="existing chunks.jsonl; default (offline): build a synthetic corpus")
    gen.add_argument("--synthetic-corpus-out", type=Path, default=DEFAULT_SYNTHETIC_CORPUS)
    gen.add_argument("--provider", choices=("anthropic", "openai"), default=None)
    gen.add_argument("--model", default=None, help="default: claude-opus-5-5 (anthropic)")
    gen.add_argument("--base-url", default=None, help="OpenAI-compatible server URL")
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--embedder", choices=("bge", "hashing"), default="bge",
                     help="embedder for the judge's retrieval step (LLM mode)")
    for cat, default in Quotas().as_dict().items():
        flag = {"in_domain_factoid": "factoid", "in_domain_reasoning": "reasoning",
                "out_of_domain": "ood", "unsupported_fact": "unsupported",
                "subtle_conflict": "conflict"}[cat.value]
        gen.add_argument(f"--n-{flag}", type=int, default=default, dest=cat.value)
    return parser


def _llm_eval_set(args: argparse.Namespace, quotas: Quotas) -> tuple[EvalSet, list[Chunk]]:
    from src.retriever import HashingEmbedder, HybridRetriever, SentenceTransformerEmbedder

    corpus = args.corpus or Path("data/processed/chunks.jsonl")
    chunks = load_chunks_jsonl(corpus)
    if args.provider == "anthropic":
        client: LLMClient = AnthropicClient(args.model or "claude-opus-5-5")
    else:
        if not args.model:
            raise SystemExit("--model is required with --provider openai")
        client = OpenAICompatibleClient(args.model, base_url=args.base_url)
    embedder = HashingEmbedder() if args.embedder == "hashing" else SentenceTransformerEmbedder()
    retriever = HybridRetriever(embedder).index(chunks)
    rng = random.Random(args.seed)
    engine = LLMEngine(client, chunks, rng, temperature=args.temperature,
                       retrieve=lambda q, k: [r.chunk for r in retriever.retrieve(q, top_k=k)])
    drafts = engine.generate(quotas)
    logger.info("rejections: %s", engine.rejections)
    return assemble_eval_set(drafts, chunks=chunks, corpus_path=corpus, generator=f"llm:{client.name}",
                             seed=args.seed, rng=rng), chunks


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    quotas = Quotas(**{c.value: getattr(args, c.value) for c in Category})
    if args.offline:
        eval_set, _ = build_offline_eval_set(seed=args.seed, quotas=quotas, corpus_path=args.corpus,
                                             synthetic_corpus_out=args.synthetic_corpus_out)
    elif args.provider:
        eval_set, _ = _llm_eval_set(args, quotas)
    else:
        raise SystemExit("choose --offline or --provider {anthropic,openai}")
    eval_set.save(args.output)
    print(summarize(eval_set))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
