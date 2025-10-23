"""Test VeRA Linear layer."""

import random
from typing import Final

import numpy as np
import pytest
import torch as th

from src.overa.vera_linear import VeraLinear, VeraMergedLinear

FULL_RANK: Final[int] = 784
RANK: Final[int] = 128
BASE_LINEAR: Final[th.nn.Module] = th.nn.Linear(FULL_RANK, FULL_RANK)


def _set_seed():
    """Set random seed."""
    random.seed(60)
    np.random.seed(60)
    th.manual_seed(60)
    th.use_deterministic_algorithms(True)


def test_lambdas():
    """Test Lambda intialisation."""
    _set_seed()
    vera_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": "f",
        "ut_lambda": "z",
    }
    vera_layer = VeraLinear(BASE_LINEAR, vera_config)
    gt_lambda_d = th.empty(RANK)
    th.nn.init.normal_(gt_lambda_d)
    th.allclose(th.zeros(FULL_RANK), vera_layer.vera_lambda_b.data)
    th.allclose(gt_lambda_d, vera_layer.vera_lambda_d.data)


@pytest.mark.parametrize("ut_init", ("z", "o"))
def test_untrained_lambda(ut_init: str):
    """Test untrained lambda inits.

    Args:
        ut_init (str): Untrained lambdas init scheme.
    """
    _set_seed()
    vera_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": "h",
        "ut_lambda": ut_init,
    }
    vera_layer = VeraLinear(BASE_LINEAR, vera_config)
    gt_fixed = th.zeros(FULL_RANK - RANK)
    if ut_init == "o":
        gt_fixed = th.ones(FULL_RANK - RANK)
    # Skipping random init-check as torch seed isnt working with randn.
    assert th.allclose(gt_fixed, vera_layer.fixed_lambda_d.data)


@pytest.mark.parametrize("init_scheme", ("f", "w", "h", "m"))
def test_vera_zero_output(init_scheme: str):
    """Test VeRA zero output.

    Args:
        init_scheme (str): Initialisation scheme.
    """
    _set_seed()
    x = th.randn((2, 4, FULL_RANK))
    expected_op = BASE_LINEAR(x)
    vera_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": init_scheme,
        "ut_lambda": "z",
        "wavelet": "Haar",
        "transform_level": 2,
    }
    vera_layer = VeraLinear(BASE_LINEAR, vera_config)
    vera_op = vera_layer(x)
    assert th.allclose(expected_op, vera_op)


@pytest.mark.parametrize("init_scheme", ("f", "w", "h", "m"))
def test_vera_merged_zero_output(init_scheme: str):
    """Test MergedLinear VeRA-Zero output.

    Args:
        init_scheme (str): intialisation scheme.
    """
    _set_seed()
    x = th.randn((2, 4, FULL_RANK))
    merged_base = th.nn.Linear(FULL_RANK, FULL_RANK * 3, bias=False)
    expected_op = merged_base(x)
    vera_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": init_scheme,
        "ut_lambda": "z",
        "wavelet": "Haar",
        "transform_level": 2,
    }
    vera_layer = VeraMergedLinear(merged_base, vera_config)
    vera_op = vera_layer(x)
    assert th.allclose(expected_op, vera_op)
