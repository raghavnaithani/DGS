# Makefile for DGS repository
# Provides common developer tasks for backend, frontend, tests, and docker-compose

# Resolve a usable Python executable from the .venv if present, otherwise fallback to system python
PYTHON := $(shell [ -f .venv/Scripts/python.exe ] && echo .venv/Scripts/python.exe || ( [ -f .venv/bin/python ] && echo .venv/bin/python || echo python ))
PIP := $(PYTHON) -m pip
POETRY := poetry

.PHONY: help init-venv install-backend install-dev install-frontend test test-ragas lint format compose-up compose-down build clean

help:
	@echo "DGS Makefile targets:"
	@echo "  init-venv           Create virtualenv at .venv and upgrade pip"
	@echo "  install-backend     Install backend package into .venv (uses poetry if available)"
	@echo "  install-dev         Install common dev dependencies (pytest, black, ruff, ragas)"
	@echo "  install-frontend    Install frontend dependencies (pnpm preferred, falls back to npm)"
	@echo "  test                Run backend test suite (pytest)"
	@echo "  test-ragas          Run RAGAS integration test only"
	@echo "  lint                Run ruff (installs it if missing)"
	@echo "  format              Run black (installs it if missing)"
	@echo "  compose-up          Start services via docker-compose"
	@echo "  compose-down        Stop services via docker-compose"
	@echo "  build               Build docker images via docker-compose"
	@echo "  clean               Remove python build artifacts and caches"

init-venv:
	@echo "Creating virtualenv at .venv if missing..."
	@test -d .venv || python -m venv .venv
	@$(PIP) install --upgrade pip setuptools wheel

install-backend: init-venv
	@echo "Installing backend package into .venv (editable)..."
	@if command -v $(POETRY) >/dev/null 2>&1; then \
		$(POETRY) install -C backend; \
	else \
		$(PIP) install -e backend; \
	fi

install-dev: init-venv
	@echo "Installing common development dependencies into .venv..."
	@$(PIP) install -U pytest black ruff ragas 'langchain-community==0.3.31'

install-frontend:
	@echo "Installing frontend dependencies (pnpm preferred)..."
	@if command -v pnpm >/dev/null 2>&1; then \
		cd frontend && pnpm install; \
	elif command -v npm >/dev/null 2>&1; then \
		cd frontend && npm install; \
	else \
		echo "pnpm/npm not found; install frontend deps manually"; \
	fi

test: init-venv
	@echo "Running backend tests..."
	@$(PYTHON) -m pytest backend -q

test-ragas: init-venv
	@echo "Running RAGAS integration test..."
	@$(PYTHON) -m pytest backend/tests/test_ragas_integration.py -q

lint:
	@echo "Running ruff on backend..."
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check backend; \
	else \
		$(PIP) install ruff && ruff check backend; \
	fi

format:
	@echo "Running black on repository..."
	@if command -v black >/dev/null 2>&1; then \
		black backend frontend shared || true; \
	else \
		$(PIP) install black && black backend frontend shared || true; \
	fi

compose-up:
	@echo "Starting services with docker-compose..."
	@docker-compose up -d

compose-down:
	@echo "Stopping services with docker-compose..."
	@docker-compose down

build:
	@echo "Building docker images (docker-compose build)..."
	@docker-compose build

clean:
	@echo "Cleaning python caches and build artifacts..."
	@find . -type f -name '*.pyc' -delete || true
	@find . -type d -name '__pycache__' -print0 | xargs -0 -r rm -rf || true
	@rm -rf build dist *.egg-info || true
