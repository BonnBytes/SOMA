"""Utils file."""

from copy import deepcopy
from typing import Tuple

import torch as th
from ptwt.matmul_transform import construct_boundary_a, construct_boundary_s
from ptwt.sparse_math import cat_sparse_identity_matrix


def construct_wavelet_matrices(
    transform_level: int, matrix_type: str, out_feats: int
) -> th.Tensor:
    """Construct multi-level wavelet transformation matrix.

    Args:
        transform_level (int): Transformation level.
        matrix_type (str): Wavelet matrix type. Analysis or Synthesis.
        out_feats (int): Base layer output size.

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
