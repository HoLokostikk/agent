.PHONY: fmt lint typecheck test test-live test-docker sandbox eval check

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

test-docker:
	uv run pytest -m docker

eval:
	uv run agent eval toolcalls -m local-small
	uv run agent eval tasks -m local-small -k 3

sandbox:
	docker build -t agent-sandbox:0.1 src/agent/sandbox

check: lint typecheck test
