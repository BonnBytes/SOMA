.PHONY: format check-format test lint check-all typing all

# Format code
format:
	uv run isort scripts/toy_random_projection src/ tests/
	uv run black scripts/toy_random_projection src/ tests/

# Check formatting (without making changes)  
check-format:
	uv run isort --check-only --diff scripts/toy_random_projection src/ tests/
	uv run black --check --diff scripts/toy_random_projection src/ tests/

lint:
	uv run flake8 scripts/toy_random_projection/ src/ tests/

typing:
	uv run mypy scripts/toy_random_projection src/ tests/

test:
	uv run pytest

# Run all pre-commit checks manually
check-all:
	uv run pre-commit run --all-files

install-precommit:
	uv run pre-commit install --hook-type pre-push

all: format typing lint test
	@echo "All checks passed!"