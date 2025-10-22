"""Tests for random projection."""

from typing import Tuple

import pytest
import torch as th

from scripts.toy_random_projection.simple_convergence import (
    FitModel,
    OrthoFit,
)


@pytest.mark.parametrize("init_scheme", ["r", "f", "w", "h", "m"])
def test_zero_output(init_scheme: str):
    """Test for zero output, since lambdas are init to zero.

    Args:
        init_scheme (str): string containing initialisation scheme.
    """
    full_rank = 256
    vera_rank = 32
    x = th.ones((full_rank, 1))
    y = th.zeros((full_rank, 1))
    model = (
        OrthoFit(full_rank, vera_rank, "z", init_scheme)
        if init_scheme != "r"
        else FitModel(full_rank, vera_rank)
    )
    model.init_weights()
    y_pred = model(x)
    assert th.allclose(y_pred, y)


walsh_3d = th.tensor([[1, 1, 1], [1, -1, 1], [1, 1, -1]])
walsh_4d = th.tensor(
    [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
)


@pytest.mark.parametrize("degree", [(3, walsh_3d), (4, walsh_4d)])
def test_walsh_init(degree: Tuple[int, th.Tensor]):
    """Test Walsh initialisation.

    Args:
        degree (Tuple[int, th.Tensor]): Tuple containing full_rank and walsh tensor.
    """
    full_rank, result = degree
    vera_rank = 1
    model = OrthoFit(full_rank, vera_rank, "z", "h")
    assert th.allclose(model.A, result)
    assert th.allclose(model.B, result.T)


@pytest.mark.parametrize("init_scheme", ["r", "f", "w", "h", "m"])
def test_lambda_inits(init_scheme: str):
    """Test lambda intialisation.

    Args:
        init_scheme (str): String containing initialisation scheme.
    """
    full_rank = 256
    vera_rank = 32
    model = (
        OrthoFit(full_rank, vera_rank, "z", init_scheme)
        if init_scheme != "r"
        else FitModel(full_rank, vera_rank)
    )
    model.init_weights()
    assert th.allclose(model.lambda_b, th.zeros_like(model.lambda_b))
