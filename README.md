# SOMA

## Installation
We use UV for installation and packaging purpose. Run the following command to create virtual environment

```bash
uv sync --extra cu126   # for CUDA
uv sync --extra cpu     # for CPU
```
Note:

To install precommit hook checks run the following
```bash
make install-precommit
```
and for the lint, test, and typing purposes run
```bash
make format    # Format isort and black
make lint      # Flake8
make typing    # MyPy
make test      # PyTest
make check-all # pre-commit check
```

## Image classification
Upcoming
