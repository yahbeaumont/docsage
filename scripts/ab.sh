#!/usr/bin/env bash
set -euo pipefail
set -a; [ -f configs/calibrated.env ] && . configs/calibrated.env; set +a
python -m docsage.eval.ab "$@"
