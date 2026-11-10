"""Test VeRA Linear layer."""

from typing import Final

import pytest
import torch as th

from src.soma.soma_linear import SomaLinear, SomaMergedLinear
from src.soma.utils import set_seed

FULL_RANK: Final[int] = 768
RANK: Final[int] = 128
BASE_LINEAR: Final[th.nn.Module] = th.nn.Linear(FULL_RANK, FULL_RANK)


def test_lambdas():
    """Test Lambda intialisation."""
    set_seed()
    soma_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": "f",
        "ut_lambda": "z",
    }
    soma_layer = SomaLinear(BASE_LINEAR, soma_config)
    gt_lambda_d = th.empty(RANK)
    th.nn.init.normal_(gt_lambda_d)
    th.allclose(th.zeros(FULL_RANK), soma_layer.vera_lambda_b.data)
    th.allclose(gt_lambda_d, soma_layer.vera_lambda_d.data)


@pytest.mark.parametrize("ut_init", ("z", "o"))
def test_untrained_lambda(ut_init: str):
    """Test untrained lambda inits.

    Args:
        ut_init (str): Untrained lambdas init scheme.
    """
    set_seed()
    soma_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": "h",
        "ut_lambda": ut_init,
    }
    soma_layer = SomaLinear(BASE_LINEAR, soma_config)
    gt_fixed = th.zeros(FULL_RANK - RANK)
    if ut_init == "o":
        gt_fixed = th.ones(FULL_RANK - RANK)
    # Skipping random init-check as torch seed isnt working with randn.
    assert th.allclose(gt_fixed, soma_layer.fixed_lambda_d.data)


@pytest.mark.parametrize("init_scheme", ("f", "w", "h"))
def test_zero_output(init_scheme: str):
    """Test SOMA zero output.

    Args:
        init_scheme (str): Initialisation scheme.
    """
    set_seed()
    x = th.randn((2, 4, FULL_RANK))
    expected_op = BASE_LINEAR(x)
    soma_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": init_scheme,
        "ut_lambda": "z",
        "wavelet": "Haar",
        "transform_level": 2,
    }
    soma_layer = SomaLinear(BASE_LINEAR, soma_config)
    vera_op = soma_layer(x)
    assert th.allclose(expected_op, vera_op)


@pytest.mark.parametrize("init_scheme", ("f", "w", "h"))
def test_merged_zero_output(init_scheme: str):
    """Test MergedLinear Zero output.

    Args:
        init_scheme (str): intialisation scheme.
    """
    set_seed()
    x = th.randn((2, 4, FULL_RANK))
    merged_base = th.nn.Linear(FULL_RANK, FULL_RANK * 3, bias=False)
    expected_op = merged_base(x)
    soma_config = {
        "rank": RANK,
        "alpha": 1.0,
        "init_scheme": init_scheme,
        "ut_lambda": "z",
        "wavelet": "Haar",
        "transform_level": 2,
    }
    soma_layer = SomaMergedLinear(merged_base, soma_config)
    vera_op = soma_layer(x)
    assert th.allclose(expected_op, vera_op)


@pytest.mark.parametrize(
    "init_scheme, out_dim, rank",
    (
        ("h", 8, 2),
        ("h", 16, 2),
        ("h", 32, 2),
    ),
)
def test_ab_matrix_orthogonality_hadamard(
    init_scheme: str, out_dim: int, rank: int
):
    """Test orthogonality/inverse relation for A and B matrices.

    Args:
        init_scheme (str): Orthogonal initialisation scheme.
        out_dim (int): Base layer in/out dimension.
        rank (int): VeRA rank.
    """
    base_linear = th.nn.Linear(out_dim, out_dim, bias=False)
    soma_config = {
        "rank": rank,
        "alpha": 1.0,
        "init_scheme": init_scheme,
        "ut_lambda": "z",
        "wavelet": "Haar",
        "transform_level": 2,
    }

    soma_layer = SomaLinear(base_linear, soma_config)
    eye = th.eye(out_dim, dtype=th.float64)
    a_mat = soma_layer.A_vera(eye)
    b_mat = soma_layer.B_vera(eye)
    expected_scale = 1.0
    scaled_eye = expected_scale * eye

    assert th.allclose(a_mat @ b_mat, scaled_eye, atol=1e-5, rtol=1e-5)
    assert th.allclose(b_mat @ a_mat, scaled_eye, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "init_scheme, out_dim, rank",
    (
        ("w", 768, 256),
        ("w", 1024, 256),
        ("w", 448, 102),
    ),
)
def test_ab_matrix_orthogonality_wavelet(
    init_scheme: str, out_dim: int, rank: int
):
    """Test orthogonality/inverse relation for A and B matrices.

    Args:
        init_scheme (str): Orthogonal initialisation scheme.
        out_dim (int): Base layer in/out dimension.
        rank (int): VeRA rank.
    """
    base_linear = th.nn.Linear(out_dim, out_dim, bias=False)
    soma_config = {
        "rank": rank,
        "alpha": 1.0,
        "init_scheme": init_scheme,
        "ut_lambda": "z",
        "wavelet": "Haar",
        "transform_level": 5,
    }

    soma_layer = SomaLinear(base_linear, soma_config)
    eye = th.eye(out_dim, dtype=th.float64)
    a_mat, sizes = soma_layer.A_vera(eye)
    coeff_eye = th.eye(a_mat.shape[-1], dtype=th.float64)
    b_mat = soma_layer.B_vera(coeff_eye, sizes=sizes)
    expected_scale = 1.0
    scaled_eye = expected_scale * eye

    assert th.allclose(a_mat @ a_mat.T, scaled_eye, atol=1e-5, rtol=1e-5)
    assert th.allclose(b_mat @ b_mat.T, scaled_eye, atol=1e-5, rtol=1e-5)
    assert th.allclose(a_mat @ b_mat, eye, atol=1e-5, rtol=1e-5)
