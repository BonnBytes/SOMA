.PHONY: format check-format test lint check-all typing all

# Format code
format:
	uv run isort scripts/toy_random_projection tests/test_randomprojection.py
	uv run black scripts/toy_random_projection tests/test_randomprojection.py

# Check formatting (without making changes)  
check-format:
	uv run isort --check-only --diff scripts/toy_random_projection tests/test_randomprojection.py
	uv run black --check --diff scripts/toy_random_projection tests/test_randomprojection.py

lint:
	uv run flake8 scripts/toy_random_projection/ tests/test_randomprojection.py

typing:
	uv run mypy scripts/toy_random_projection tests/test_randomprojection.py

test:
	uv run pytest

# Run all pre-commit checks manually
check-all:
	uv run pre-commit run --all-files

all: format test lint typing
	@echo "All checks passed!"