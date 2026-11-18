FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd -m -u 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=25s \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "docsage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
