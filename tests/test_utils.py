"""Tests for Utils package."""

import torch as th

from src.overa.utils import split_qkv_weight


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
