"""Cross-encoder reranking of hybrid-retrieval candidates. Phase 2, not implemented yet.

Planned interface::

    class CrossEncoderReranker:
        def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 16): ...
        def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...

Reranker logits (raw and softmax-normalised) feed the abstention layer as
query-passage relevance features.
"""

from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 2: see ROADMAP.md")
