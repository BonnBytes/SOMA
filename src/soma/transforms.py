"""Naive hadmard transform."""

from typing import List, Optional, Tuple, Union

import ptwt
import torch as th

from src.soma.utils import next_pow


@th.compile(fullgraph=True)
def fwht_optimized(x: th.Tensor, inverse: Optional[bool] = False) -> th.Tensor:
    """Hadamard Transform.

    Args:
        x (th.Tensor): Input Tensor.
        inverse (bool, optional): Boolean specifying inverse or not. Defaults to False.

    Raises:
        TypeError: Atleast 1D tensor is expected.

    Returns:
        th.Tensor: Transformed output.
    """
    if x.dim() == 0:
        raise TypeError("Expected at least 1D tensor.")

    n_orig = x.shape[-1]
    n = next_pow(n_orig)

    a = th.nn.functional.pad(x, (0, n - n_orig)) if n > n_orig else x.clone()
    batch_shape = a.shape[:-1]

    h = 2
    while h <= n:
        hf = h // 2
        a = a.view(*batch_shape, n // h, 2, hf)
        u = a[..., 0, :] + a[..., 1, :]
        v = a[..., 0, :] - a[..., 1, :]
        a = th.stack([u, v], dim=-2).view(*batch_shape, n)
        h *= 2

    if inverse:
        a /= n
    if n > n_orig:
        a = a[..., :n_orig]

    return a


def fast_wavelet_transform(
    x: th.Tensor,
    inverse: Optional[bool] = False,
    sizes: Optional[Tuple[int, ...]] = None,
    **kwargs,
) -> Union[th.Tensor, Tuple[th.Tensor, List[int]]]:
    """Compute fast wavelet transform and inverse.

    Args:
        x (th.Tensor): Input tensor.
        inverse (Optional[bool], optional): Boolean to specify if inverse or not. Defaults to False.
        sizes (Optional[Tuple[int, ...]], optional): Packet sizes for inverse transform. Defaults to None.

    Raises:
        ValueError: ValueError if sizes are not provided with inverse.
        TypeError: Atleast 1D tensor is expected.

    Returns:
        Union[th.Tensor, Tuple[th.Tensor, List[int]]]: If ``inverse`` is False,
            a tuple of the concatenated coefficients and the size of each
            coefficient band. Otherwise, the reconstructed tensor.
    """
    if inverse and sizes is None:
        raise ValueError("Sizes must be provided for inverse transform.")
    if x.dim() == 0:
        raise TypeError("Expected at least 1D tensor.")
    wavelet = kwargs.get("wavelet")
    transform_level = kwargs.get("transform_level")
    if not inverse:
        coeffs = ptwt.wavedec(
            x, wavelet=wavelet, level=transform_level, mode="zero"
        )
        sizes_out = [coeff.shape[-1] for coeff in coeffs]
        return th.cat(coeffs, dim=-1), sizes_out
    else:
        parts = list(x.split(sizes, dim=-1))
        return ptwt.waverec(parts, wavelet=wavelet)
