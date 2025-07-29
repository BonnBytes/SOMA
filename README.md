# Orthogonal VeRA

## Installation
We use UV for installation and packaging purpose. Run the following command to create virtual environment

```bash
uv sync --extra cu124
```
Note: For developers

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

## Toy example
To train the model for the toy example in the paper, execute
```bash
bash scripts/toy_random_projection/runs.sh
```
