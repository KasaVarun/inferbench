.PHONY: setup test lint format test-py test-go lint-py lint-go format-py format-go

PYTHON := python3.12
VENV := .venv
VENV_BIN := $(VENV)/bin

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -e ".[dev]"
	cd apps/api-go && go mod tidy

test: test-py test-go

test-py:
	$(VENV_BIN)/pytest

test-go:
	cd apps/api-go && go test ./...

lint: lint-py lint-go

lint-py:
	$(VENV_BIN)/ruff check .
	$(VENV_BIN)/mypy

lint-go:
	cd apps/api-go && test -z "$$(gofmt -l .)"
	cd apps/api-go && go vet ./...

format: format-py format-go

format-py:
	$(VENV_BIN)/ruff format .
	$(VENV_BIN)/ruff check --fix .

format-go:
	cd apps/api-go && gofmt -w .
