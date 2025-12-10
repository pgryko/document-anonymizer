.PHONY: help install install-dev lint format test test-unit test-integration test-coverage clean build docker-build docker-run

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
POETRY := poetry

# Project directories
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	$(POETRY) install --only main

install-dev: ## Install all dependencies including dev
	$(POETRY) install
	$(POETRY) run pre-commit install

lint: ## Run all linters
	$(POETRY) run ruff check $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run black --check $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run mypy $(SRC_DIR)
	$(POETRY) run bandit -r $(SRC_DIR) -ll -i

format: ## Format code with black and ruff
	$(POETRY) run black $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run ruff check --fix $(SRC_DIR) $(TESTS_DIR)

test: ## Run all tests
	$(POETRY) run pytest $(TESTS_DIR) -v

test-unit: ## Run unit tests only
	$(POETRY) run pytest $(TESTS_DIR)/unit -v

test-integration: ## Run integration tests only
	$(POETRY) run pytest $(TESTS_DIR)/integration -v

test-coverage: ## Run tests with coverage report
	$(POETRY) run pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term

test-security: ## Run security tests
	$(POETRY) run bandit -r $(SRC_DIR) -ll -i -f json -o bandit-report.json
	$(POETRY) run safety check --json > safety-report.json || true

benchmark: ## Run performance benchmarks
	$(POETRY) run pytest $(TESTS_DIR) -v -m benchmark

clean: ## Clean build artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage coverage.xml

build: clean ## Build distribution packages
	$(POETRY) build

docker-build: ## Build Docker image
	docker build -t document-anonymizer:latest .

docker-build-dev: ## Build development Docker image
	docker build -t document-anonymizer:dev --target development .

docker-run: ## Run Docker container
	docker run --rm -it \
		-v $(PWD)/data:/data \
		-v $(PWD)/models:/models \
		document-anonymizer:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

pre-commit: ## Run pre-commit on all files
	$(POETRY) run pre-commit run --all-files

docs-build: ## Build documentation
	cd $(DOCS_DIR) && $(POETRY) run mkdocs build

docs-serve: ## Serve documentation locally
	cd $(DOCS_DIR) && $(POETRY) run mkdocs serve

# Development workflow targets
dev-setup: install-dev pre-commit ## Complete development environment setup

dev-check: lint test-unit ## Quick check before committing

ci-local: lint test-coverage test-security ## Run CI pipeline locally

# Batch processing specific targets
batch-test: ## Test batch processing with sample data
	$(PYTHON) main.py batch-anonymize \
		--input-dir data/samples/input \
		--output-dir data/samples/output \
		--max-parallel 2 \
		--batch-size 4

# Model management targets
models-download: ## Download required models
	$(PYTHON) -m src.anonymizer.models.downloader --all

models-validate: ## Validate downloaded models
	$(PYTHON) -m src.anonymizer.models.validator --check-all
