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
	@echo "bench defined in Part 10"

full-pipeline:
	@echo "full-pipeline defined in Part 11"