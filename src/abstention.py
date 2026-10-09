"""Calibrated abstention via split conformal prediction. Phase 4, not implemented yet.

Planned components:

* ``AbstentionFeatures``: retrieval margins (top-1 minus top-2 RRF/dense/BM25),
  reranker confidence, answer-evidence entailment and self-consistency.
* ``ConformalAbstainer``: fits a threshold on a held-out calibration split so that
  the selective risk satisfies ``P(error | answered) <= alpha`` with a
  finite-sample guarantee (conformal risk control / Learn-then-Test).
"""

from __future__ import annotations


class ConformalAbstainer:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 4: see ROADMAP.md")
