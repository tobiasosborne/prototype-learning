#!/bin/sh
# All local gates. Exit non-zero on the first failure.
set -e
cd "$(dirname "$0")/.."
uv run ruff check src tests experiments
uv run ruff format --check src tests experiments
uv run mypy src
uv run pytest -m "not slow"
