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


## Citation
```
@article{
veeramacheneni2026soma,
title={{SOMA}: Structured and Orthogonal Matrix Adaptation for Parameter Efficient Fine-Tuning of Transformers},
author={Lokesh Veeramacheneni and Zahra Ganji and Moritz Wolter and Juergen Gall},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=m9GBX3oKHH},
note={}
}
```
