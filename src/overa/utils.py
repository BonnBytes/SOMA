"""Utils file."""

import random
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch as th
from ptwt.matmul_transform import construct_boundary_a, construct_boundary_s
from ptwt.sparse_math import cat_sparse_identity_matrix


def construct_wavelet_matrices(
    transform_level: int, matrix_type: str, out_feats: int, wavelet: str
) -> th.Tensor:
    """Construct multi-level wavelet transformation matrix.

    Args:
        transform_level (int): Transformation level.
        matrix_type (str): Wavelet matrix type. Analysis or Synthesis.
        out_feats (int): Base layer output size.
        wavelet (str): Wavelet to use.

    Returns:
        th.Tensor: Multi-level wavelet basis matrix.
    """
    matrices = []
    length = deepcopy(out_feats)
    wavelet_fn = (
        construct_boundary_a if matrix_type == "a" else construct_boundary_s
    )
    for i in range(transform_level):
        mat = wavelet_fn(wavelet="Haar", length=length)
        length = length // 2
        if i != 0:
            mat = cat_sparse_identity_matrix(mat, out_feats)
        matrices.append(mat)
    return __construct_final_matrix(matrix_type, tuple(matrices), out_feats)


def __construct_final_matrix(
    matrix_type: str, matrices: Tuple[th.Tensor, ...], out_feats: int
) -> th.Tensor:
    """Compute final wavelet matrix.

    Args:
        matrix_type (str): Wavelet matrix type. Analysis or Synthesis.
        matrices (Tuple[th.Tensor, ...]): Tuple of multi-level wavelet matrices.
        out_feats (int): Base layer output size.

    Returns:
        th.Tensor: Multi-level wavelet basis matrix.
    """
    final_matrix = th.eye(out_feats, dtype=matrices[0].dtype)
    matrices = matrices if matrix_type == "s" else matrices[::-1]
    for mat in matrices:
        final_matrix = final_matrix @ mat
    return final_matrix


def next_pow(n: int) -> int:
    """Compute next power of two.

    Args:
        n (int): Input number.

    Returns:
        int: next power of two.
    """
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    return 1 << n.bit_length()


def create_hippo(n: int) -> th.Tensor:
    """Create HiPPO matrices.

    Args:
        n (int): matrix size.

    Returns:
        th.Tensor: HiPPO matrix.
    """
    matrix = th.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i < k:
                matrix[i, k] = 0
            elif i == k:
                matrix[i, k] = i + 1
            elif i > k:
                matrix[i, k] = ((2 * i + 1) ** 0.5) * ((2 * k + 1) ** 0.5)
    return matrix


def split_qkv_weight(weight: th.Tensor) -> Tuple[th.Tensor, ...]:
    """Split QKV tensor.

    Args:
        weight (th.Tensor): Pretraiend weight tensor.

    Raises:
        ValueError: Error if wrong linear weight is provided.

    Returns:
        Tuple[th.Tensor, ...]: Unbinded tuple containing Q, K, and V matrices.
    """
    out_, in_ = weight.shape
    if out_ != 3 * in_:
        raise ValueError("Improper weight tensor.")
    new_weight = weight.reshape((3, in_, in_))
    q, k, v = th.unbind(new_weight, dim=0)
    return q, k, v


def set_seed(seed: Optional[int] = 60):
    """Set global seeds.

    Args:
        seed (int, optional): seed value. Defaults to 60.
    """
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True


def welsh_hadamard_transform(x: th.Tensor, inverse: bool = False) -> th.Tensor:
    """Compute Welsh Hadamard Transform

    Args:
        x (th.Tensor): Input tensor
        inverse (bool, optional): Whether to compute the inverse transform. Defaults to False.

    Returns:
        th.Tensor: Transformed tensor
    """
    if x.dim() == 0:
        raise TypeError("Expected at least 1D tensor.")
    a = x.float().clone()
    n_orig = a.shape[-1]

    # Pad last dim to next power of 2
    n = next_pow(n_orig)
    if n > n_orig:
        pad = th.zeros(*a.shape[:-1], n-n_orig, dtype=a.dtype, device=a.device)
        a = th.cat([a, pad], dim=-1)
    
    h = 2
    while h <= n:
        hf = h // 2
        shape = a.shape[:-1] + (n // h, 2, hf)
        a = a.reshape(shape)
        u, v = a[..., 0, :].clone(), a[..., 1, :].clone()
        a[..., 0, :] = u + v
        a[..., 1, :] = u - v
        a = a.reshape(*a.shape[:-3] + (n,))
        h *= 2
    
    if inverse:
        a = a / n
    
    if n > n_orig:
        a = a[..., :n_orig]
    return a