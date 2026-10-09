"""Strict Pydantic schemas for the evaluation ground truth (``data/eval/eval_set.json``).

The validators enforce the invariants that make the benchmark trustworthy:

* answerable items cite at least one evidence chunk and carry a real answer;
* unanswerable items cite **no** chunk and carry exactly the canonical abstention string;
* the category always agrees with ``is_answerable``;
* every cited chunk id exists in the corpus the set was built from.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.constants import ABSTENTION_ANSWER  # noqa: F401  (re-exported)
from src.data_loader import Chunk

SCHEMA_VERSION = "1.0"


class Category(str, Enum):
    IN_DOMAIN_FACTOID = "in_domain_factoid"
    IN_DOMAIN_REASONING = "in_domain_reasoning"
    OUT_OF_DOMAIN = "out_of_domain"
    UNSUPPORTED_FACT = "unsupported_fact"
    SUBTLE_CONFLICT = "subtle_conflict"


ANSWERABLE_CATEGORIES = frozenset({Category.IN_DOMAIN_FACTOID, Category.IN_DOMAIN_REASONING})
UNANSWERABLE_CATEGORIES = frozenset(set(Category) - ANSWERABLE_CATEGORIES)


class ItemMetadata(BaseModel):
    """Per-item provenance. Extra keys are allowed so generators can attach more context."""

    model_config = ConfigDict(extra="allow", frozen=True)

    arxiv_id: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    generator: str
    template: str | None = None
    # Chunks that look relevant but do NOT answer the question (e.g. the chunk a
    # false-premise question contradicts). For analysis only; never ground truth.
    distractor_chunk_ids: list[str] = Field(default_factory=list)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^eval_\d{3,}$")
    question: str
    is_answerable: bool
    category: Category
    ground_truth_chunk_ids: list[str]
    reference_answer: str
    metadata: ItemMetadata

    @field_validator("question", "reference_answer")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _check_answerability(self) -> EvalItem:
        if self.is_answerable:
            if self.category not in ANSWERABLE_CATEGORIES:
                raise ValueError(f"answerable item cannot have category {self.category.value!r}")
            if not self.ground_truth_chunk_ids:
                raise ValueError("answerable item needs at least one ground_truth_chunk_id")
            if len(set(self.ground_truth_chunk_ids)) != len(self.ground_truth_chunk_ids):
                raise ValueError("ground_truth_chunk_ids must be unique")
            if self.reference_answer.strip() == ABSTENTION_ANSWER:
                raise ValueError("answerable item cannot use the abstention answer")
        else:
            if self.category not in UNANSWERABLE_CATEGORIES:
                raise ValueError(f"unanswerable item cannot have category {self.category.value!r}")
            if self.ground_truth_chunk_ids:
                raise ValueError("unanswerable item must have no ground_truth_chunk_ids")
            if self.reference_answer != ABSTENTION_ANSWER:
                raise ValueError("unanswerable item must use the canonical abstention answer")
        return self


class CorpusInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    num_chunks: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def corpus_fingerprint(chunks: Iterable[Chunk]) -> str:
    """sha256 over the canonical JSON of every chunk, in order."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps(chunk.to_dict(), sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class EvalSet(BaseModel):
    """Top-level document of ``eval_set.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_VERSION
    generator: str
    seed: int | None = None
    corpus: CorpusInfo
    counts: dict[str, int]
    items: list[EvalItem]

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalSet:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("item ids must be unique")
        if ids != sorted(ids):
            raise ValueError("items must be sorted by id")
        if self.counts != count_categories(self.items):
            raise ValueError("counts do not match the items")
        return self

    @property
    def answerable(self) -> list[EvalItem]:
        return [i for i in self.items if i.is_answerable]

    @property
    def unanswerable(self) -> list[EvalItem]:
        return [i for i in self.items if not i.is_answerable]

    def validate_against_corpus(self, chunk_ids: Iterable[str]) -> None:
        known = set(chunk_ids)
        missing = sorted(
            {cid for item in self.items for cid in item.ground_truth_chunk_ids if cid not in known}
            | {cid for item in self.items for cid in item.metadata.distractor_chunk_ids if cid not in known}
        )
        if missing:
            raise ValueError(f"{len(missing)} cited chunk ids are not in the corpus, e.g. {missing[:3]}")

    def to_json(self) -> str:
        """Deterministic serialisation: same content gives the same bytes."""
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> EvalSet:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def count_categories(items: Iterable[EvalItem]) -> dict[str, int]:
    counts = {c.value: 0 for c in Category}
    for item in items:
        counts[item.category.value] += 1
    return counts


def make_item_id(index: int) -> str:
    return f"eval_{index:03d}"
