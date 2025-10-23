"""Tests for Utils package."""

from math import sqrt
from typing import Tuple

import pytest
import torch as th

from src.overa.utils import create_hippo, next_pow, split_qkv_weight


def test_qkv_split():
    """Test QKV split function."""
    dummy_input = th.randn((2, 256, 784))
    linear_layer = th.nn.Linear(
        in_features=784, out_features=784 * 3, bias=False
    )
    gt = linear_layer(dummy_input)
    gt = gt.reshape((2, 256, 3, 784))
    gt_q, gt_k, gt_v = gt.unbind(2)
    w_q, w_k, w_v = split_qkv_weight(linear_layer.weight.data)
    q = dummy_input @ w_q.T
    k = dummy_input @ w_k.T
    v = dummy_input @ w_v.T
    assert th.allclose(gt_q, q)
    assert th.allclose(gt_k, k)
    assert th.allclose(gt_v, v)


@pytest.mark.parametrize(
    "io_maps", ((3, 4), (768, 1024), (1975, 2048), (3452, 4096))
)
def test_nextpower(io_maps: Tuple[int, int]):
    """Test finding next power of 2, given a number.

    Args:
        io_maps (Tuple[int, int]): Tuple containing current number and next power of 2.
    """
    input_int, near_power = io_maps
    power = next_pow(input_int)
    assert power == near_power


@pytest.mark.parametrize(
    "io_maps",
    (
        (2, th.tensor([[1, 0], [sqrt(3), 2]])),
        (
            3,
            th.tensor(
                [[1, 0, 0], [sqrt(3), 2, 0], [sqrt(5), sqrt(5) * sqrt(3), 3]]
            ),
        ),
    ),
)
def test_hippo_matrix(io_maps: Tuple[int, th.Tensor]):
    """Test hippo initialisation.

    Args:
        io_maps (Tuple[int, th.Tensor]): Tuple containing hippo matrix and shape.
    """
    n, gt = io_maps
    matrix = create_hippo(n)
    assert th.allclose(matrix, gt)
