"""FastAPI service exposing the selective RAG pipeline. Phase 7 (future work), not implemented yet.

Planned endpoints: ``POST /query`` (answer or abstention + evidence), ``GET /health``.
Until then, use the CLI (``python -m src.cli ask ...``), which wraps the same
``SelectiveRAGPipeline``. Planned run command: ``uvicorn src.server:app``.
"""

from __future__ import annotations
