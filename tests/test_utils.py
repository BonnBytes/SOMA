"""Tests for Utils package."""

from typing import Tuple

import numpy as np
import pytest
import pywt
import torch as th
from scipy.linalg import hadamard

from src.soma.transforms import fast_wavelet_transform, fwht_optimized
from src.soma.utils import next_pow, split_qkv_weight


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


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Reset torch.compile caches so per-shape recompiles don't hit the limit.

    Yields:
        None: Test.
    """
    th._dynamo.reset()
    yield


def test_hadamard_transform():
    """Test Hadamard forward and inverse against the explicit matrix."""
    n = 256
    x = th.randn((4, n), dtype=th.float64)
    h_mat = th.from_numpy(hadamard(n)).to(th.float64)
    y = fwht_optimized(x)
    assert th.allclose(y, x @ h_mat)
    assert th.allclose(fwht_optimized(y, inverse=True), x)


def test_wavelet_transform():
    """Test wavelet forward against PyWavelets and inverse reconstruction."""
    wavelet = pywt.Wavelet("haar")
    x = th.randn((4, 768), dtype=th.float64)
    coeffs, sizes = fast_wavelet_transform(
        x, wavelet=wavelet, transform_level=5
    )
    gt = pywt.wavedec(x.numpy(), wavelet, mode="zero", level=5, axis=-1)
    assert sizes == [c.shape[-1] for c in gt]
    assert th.allclose(coeffs, th.from_numpy(np.concatenate(gt, axis=-1)))
    rec = fast_wavelet_transform(
        coeffs, inverse=True, sizes=sizes, wavelet=wavelet
    )
    assert th.allclose(rec, x)
