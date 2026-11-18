#!/usr/bin/env bash
set -euo pipefail
OLLAMA=${OLLAMA_HOST:-http://localhost:11434}
for model in llama3.1:8b-instruct-q4_K_M nomic-embed-text; do
  echo "puxando $model"
  curl -fsS "$OLLAMA/api/pull" -d "{\"name\":\"$model\"}" | tail -1
done
