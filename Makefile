.PHONY: fmt lint typecheck test test-live check

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

test-live:
	uv run pytest -m live

check: lint typecheck test
