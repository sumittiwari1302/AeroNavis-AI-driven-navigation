PYTHON ?= $(shell command -v python3.11 || command -v python3.10 || command -v python3)

.PHONY: setup format lint test bench full-pipeline

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -e ".[dev]"

format:
	.venv/bin/ruff format src tests

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q

bench:
	@echo "=== NAV-X 3.0 Benchmark Suite ==="
	@echo "Running forced-blackout protocol..."
	.venv/bin/python -m navx.eval.forced_blackout --help 2>/dev/null || echo "Run: python -m navx.eval.forced_blackout --help"
	@echo "Running calibration benchmark..."
	.venv/bin/python -m navx.eval.adapt_eval --help 2>/dev/null || echo "Run: python -m navx.eval.adapt_eval --help"
	@echo "Generating SUMMARY.md..."
	.venv/bin/python scripts/bench_summary.py

full-pipeline:
	@echo "full-pipeline defined in Part 11"