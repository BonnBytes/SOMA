.PHONY: format check-format test lint check-all typing all

# Format code
format:
	uv run isort scripts/image_classification/ src/ tests/
	uv run black scripts/image_classification/ src/ tests/

# Check formatting (without making changes)  
check-format:
	uv run isort --check-only --diff scripts/image_classification/ src/ tests/
	uv run black --check --diff scripts/image_classification/ src/ tests/

lint:
	uv run flake8 scripts/image_classification/ src/ tests/

typing:
	uv run mypy src/

test:
	uv run pytest

# Run all pre-commit checks manually
check-all:
	uv run pre-commit run --all-files

install-precommit:
	uv run pre-commit install --hook-type pre-push

all: format typing lint test
	@echo "All checks passed!"