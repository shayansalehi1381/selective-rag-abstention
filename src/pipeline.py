"""End-to-end selective RAG pipeline. Phase 3/4, not implemented yet.

Planned flow: retrieve (HybridRetriever) -> rerank -> generate -> score -> answer or abstain.
"""

from __future__ import annotations


class SelectiveRAGPipeline:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Phase 3/4: see ROADMAP.md")
