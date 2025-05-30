.PHONY: format check-format

# Format code
format:
	uv run isort scripts/convergence/random_projection tests/test_randomprojection.py
	uv run black scripts/convergence/random_projection tests/test_randomprojection.py

# Check formatting (without making changes)  
check-format:
	uv run isort --check-only --diff scripts/convergence/random_projection tests/test_randomprojection.py
	uv run black --check --diff scripts/convergence/random_projection tests/test_randomprojection.py