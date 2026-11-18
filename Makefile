.PHONY: install corpus index serve eval eval-baseline calibrate gate ab verdict reproduce promote test lint fmt plots clean

install:
	pip install -e ".[dev]"

corpus:
	python -m docsage.cli corpus fetch --out data/corpus
	python -m docsage.cli corpus parse --in data/corpus --out data/chunks.jsonl

index:
	python -m docsage.cli index build --chunks data/chunks.jsonl

serve:
	uvicorn docsage.api.main:app --host 0.0.0.0 --port 8000

eval:
	python -m docsage.eval.run_eval --out eval-out/current.json --label current

eval-baseline:
	python -m docsage.eval.run_eval --out eval-out/baseline.json --label baseline --baseline

calibrate:
	python -m docsage.calibrate --folds 5 --out configs/calibrated.env

gate:
	python -m docsage.eval.gate --current eval-out/current.json

ab:
	./scripts/ab.sh

verdict:
	python -m docsage.eval.gate --verdict eval-out/ab.json

reproduce:
	./scripts/reproduce.sh

promote:
	python -m docsage.eval.gate --current eval-out/current.json --update-baseline

plots:
	python scripts/plot_results.py --out docs/img

test:
	pytest

lint:
	ruff check src tests && mypy src

fmt:
	ruff format src tests && ruff check --fix src tests

clean:
	rm -rf data/chunks.jsonl .bm25 data/qdrant_local eval-out .pytest_cache
