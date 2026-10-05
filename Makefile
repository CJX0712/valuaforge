# ValuaForge developer entry points.
# Author: 晨星
#
# Every target fails loudly: no `|| true`, no `-k` that swallows errors.
# A lint or test step that cannot fail is a step that reports nothing
# (pitfall library H -- the MiaForge v0.1.1 `|| true` incident).

PYTHON ?= python
RUFF   ?= ruff

.DEFAULT_GOAL := help
.PHONY: help install lint format fmt-check test test-verbose coverage demo repro check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

lint: ## Lint (hard gate)
	$(RUFF) check .

format: ## Format in place
	$(RUFF) format .

fmt-check: ## Verify formatting without writing (CI gate)
	$(RUFF) format --check .

test: ## Run the test suite
	$(PYTHON) -m pytest -q

test-verbose: ## Run the test suite with per-test output
	$(PYTHON) -m pytest -v

coverage: ## Test with coverage over core/
	$(PYTHON) -m pytest -q --cov=core --cov-report=term-missing

demo: ## End-to-end smoke run
	$(PYTHON) -m cli

repro: ## Re-run the V1-V11 evidence scripts (slow, not in CI)
	$(PYTHON) repro/verify_knn_shapley.py
	$(PYTHON) repro/verify_estimators.py
	$(PYTHON) repro/verify_surrogate.py

check: lint fmt-check test ## Full local gate, same order as CI

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
