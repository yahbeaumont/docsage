#!/usr/bin/env bash
set -euo pipefail

if [ -n "${OPENAI_API_KEY:-}" ] && [ -z "${LLM_PROVIDER:-}" ]; then
  export LLM_PROVIDER=openai
fi

echo "== corpus + índice =="
make corpus index

if [ -f configs/calibrated.env ]; then
  echo "== limiar calibrado =="
  set -a; . configs/calibrated.env; set +a
  echo "ABSTAIN_THRESHOLD=$ABSTAIN_THRESHOLD"
fi

mkdir -p eval-out
echo "== braço baseline =="
python -m docsage.eval.run_eval --out eval-out/baseline.json --label baseline --baseline

echo "== braço docsage =="
python -m docsage.eval.run_eval --out eval-out/current.json --label current

echo "== gate =="
python -m docsage.eval.gate --current eval-out/current.json
