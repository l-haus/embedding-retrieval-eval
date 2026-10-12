.PHONY: help sync hooks fmt lint test cov check run scifact run-scifact clean

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-12s %s\n", $$1, $$2}'

sync:  ## Install the project and dev dependencies
	uv sync --all-extras

hooks:  ## Install pre-commit hooks (run once per clone)
	uv run pre-commit install

fmt:  ## Format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Check formatting and lint without modifying files
	uv run ruff format --check .
	uv run ruff check .

test:  ## Run the test suite
	uv run pytest

cov:  ## Run tests with a coverage report
	uv run pytest --cov=retrieval_eval --cov-report=term-missing

check: lint test  ## What CI runs: lint then test

run:  ## Run the harness on the example config
	uv run retrieval-eval run --config configs/example.yaml

scifact:  ## Download BEIR SciFact and convert its test split
	uv run python -m retrieval_eval.beir scifact --split test --out data/scifact

run-scifact: scifact  ## Run the harness on SciFact
	uv run retrieval-eval run --config configs/scifact.yaml

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
