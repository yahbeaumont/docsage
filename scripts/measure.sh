#!/usr/bin/env bash
# Cadeia de medição reproduzível: calibra limiar, roda os dois braços e o A/B pareado.
# Uso: ./scripts/measure.sh   (requer .env preenchido e índice construído)
set -euo pipefail

mkdir -p eval-out
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "[$(stamp)] calibração"
python -m docsage.calibrate --folds 5 --out configs/calibrated.env
set -a; . configs/calibrated.env; set +a
echo "ABSTAIN_THRESHOLD=$ABSTAIN_THRESHOLD"

echo "[$(stamp)] braço docsage (híbrido + rerank)"
python -m docsage.eval.run_eval --out eval-out/current.json --label docsage

echo "[$(stamp)] braço baseline (vetorial puro, sem rerank)"
python -m docsage.eval.run_eval --out eval-out/baseline.json --label baseline --baseline

echo "[$(stamp)] A/B pareado com bootstrap"
python -m docsage.eval.ab --out eval-out/ab.json

echo "[$(stamp)] fim"
