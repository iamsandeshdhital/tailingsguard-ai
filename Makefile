.PHONY: help install test lint run docker-build docker-run clean

PYTHON := python3
PIP := pip3

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov flake8

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --cov=core --cov-report=term-missing

test-verbose: ## Run all tests with verbose output
	$(PYTHON) -m pytest tests/ -vvv --tb=long

lint: ## Run linter
	flake8 core tests --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 core tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

run: ## Run the application
	$(PYTHON) tailingsguard.py --config config.yaml

run-demo: ## Run in demo mode with failure scenario
	$(PYTHON) tailingsguard.py --config config.yaml --demo

docker-build: ## Build Docker image
	docker build -t tailingsguard-ai:latest .

docker-run: ## Run Docker container
	docker run -p 8050:8050 -v $(PWD)/data:/app/data tailingsguard-ai:latest

docker-compose-up: ## Start with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop docker-compose
	docker-compose down

clean: ## Clean up generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf core/__pycache__ tests/__pycache__
	rm -rf data/*.db
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

setup: ## Initial setup
	mkdir -p data models
	touch data/.gitkeep models/.gitkeep
