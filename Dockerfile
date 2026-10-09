# CPU image for the Selective RAG API.
#
#   docker build -t selective-rag .                                  # offline backends (small image)
#   docker build -t selective-rag:models --build-arg WITH_MODELS=true .   # + bge / cross-encoder / LLM SDKs
#   docker run -p 8000:8000 -e SRAG_ADMIN_TOKEN=change-me selective-rag
#
# The default command serves the committed synthetic corpus with offline stand-ins
# (--offline). For real models, build with WITH_MODELS=true and override the command, e.g.
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... selective-rag:models \
#     selective-rag serve --host 0.0.0.0 --strict-reranker --strict-generator
FROM python:3.12-slim

ARG WITH_MODELS=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (better layer caching), then the code and the committed eval data.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY eval ./eval
RUN pip install ".[serving]" \
 && if [ "$WITH_MODELS" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
     && pip install ".[models,llm]"; \
    fi
COPY data/eval/eval_set.json data/eval/synthetic_corpus.jsonl ./data/eval/

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker per container: each worker holds its own models and its own calibrated tau.
CMD ["selective-rag", "serve", "--host", "0.0.0.0", "--port", "8000", "--offline", "--no-cache"]
