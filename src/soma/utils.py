"""Utils file."""

import random
from typing import Optional, Tuple

import numpy as np
import torch as th


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
