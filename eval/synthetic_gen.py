"""Synthetic evaluation set generation. Phase 3, not implemented yet.

Builds ``data/eval/qa.jsonl`` with answerable questions grounded in specific chunks
(gold ``chunk_id`` + answer) and deliberately unanswerable ones (out-of-corpus
topics, entity/number perturbations), so abstention has real negatives to learn from.
"""

from __future__ import annotations
