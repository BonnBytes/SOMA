"""Vera Linear Layer."""

from functools import partial
from typing import Any, Callable, Dict, Tuple, Union

import torch as th
from scipy.linalg import hadamard

from .utils import (
    construct_wavelet_matrices,
    create_hippo,
    next_pow,
    split_qkv_weight,
)
from .vera_base import VeraBase


class VeraLinear(VeraBase):
    """VeRA Linear Layer."""

    def __init__(
        self, base_layer: th.nn.Module, config: Dict[str, Any]
    ) -> None:
        """Initialise VeRA Linear.

        Args:
            base_layer (th.nn.Module): Base linear layer.
            config (Dict[str, Any]): VeRA Config.

        Raises:
            ValueError: Error if ortho init is outside of selected ones.
            ValueError: Error if untrained lambdas as not properly intialised.
        """
        super(VeraLinear, self).__init__(base_layer)
        self.rank = config["rank"]
        self.alpha = config["alpha"]
        self.__init_scheme = config["init_scheme"]
        if self.__init_scheme not in ["f", "w", "h", "m"]:
            raise ValueError("Improper init scheme")
        self.__ut_lambda_init = config["ut_lambda"]
        if self.__ut_lambda_init not in ["z", "o", "r"]:
            raise ValueError("Improper untrained lambda init scheme")
        if self.__init_scheme == "w":
            self.__wavelet = config["wavelet"]
            self.__transform_level = config["transform_level"]
        self.update_layer()

    def compute_bases(self) -> Tuple[Union[Callable, th.Tensor], ...]:
        """Compute relevant bases matrices.

        Returns:
            Tuple[Union[Callable, th.Tensor], ...]: Tuple containing A and B matrices.
        """
        if self.__init_scheme == "w":
            return self.wavelet_bases()
        elif self.__init_scheme == "h":
            return self.welsh_bases()
        elif self.__init_scheme == "m":
            return self.hippo_bases()
        else:
            return self.fourier_bases()

    def wavelet_bases(self) -> Tuple[th.Tensor, ...]:
        """Compute Wavelet bases.

        Returns:
            Tuple[th.Tensor, ...]: Tuple containing wavelet bases matrices.
        """
        a_mat = construct_wavelet_matrices(
            self.__transform_level, "a", self.out_feat, self.__wavelet
        ).to_dense()
        b_mat = construct_wavelet_matrices(
            self.__transform_level, "s", self.out_feat, self.__wavelet
        ).to_dense()
        return a_mat, b_mat

    def fourier_bases(self) -> Tuple[Callable, ...]:
        """Compute rFFT bases.

        Returns:
            Tuple[Callable, ...]: Tuple containing fourier matrices.
        """
        a_mat = partial(th.fft.rfft, norm="ortho", n=self.out_feat)
        b_mat = partial(th.fft.irfft, norm="ortho", n=self.out_feat)
        return a_mat, b_mat

    def welsh_bases(self) -> Tuple[th.Tensor, ...]:
        """Compute Wavelet transform bases.

        Returns:
            Tuple[th.Tensor, ...]: Tuple containing wavelet matrices.
        """
        power = next_pow(self.out_feat)
        A = th.from_numpy(hadamard(power))
        B = th.from_numpy(hadamard(power))
        A = A[: self.out_feat, : self.out_feat]
        B = A.T
        return A, B

    def hippo_bases(self) -> Tuple[th.Tensor, ...]:
        """Compute Hippo bases.

        Returns:
            Tuple[th.Tensor, ...]: Tuple containing Hippo matrices.
        """
        A = create_hippo(self.out_feat)
        B = th.linalg.inv(A)
        return A, B

    def update_layer(self) -> None:
        """Initialise VeRA parameters.

        Raises:
            ValueError: Error if rank is greater than Fourier embedding dimension.
        """
        self.vera_lambda_b = th.nn.Parameter(
            th.zeros(self.out_feat), requires_grad=True
        )
        self.vera_lambda_d = th.nn.Parameter(
            th.empty(self.rank), requires_grad=True
        )
        th.nn.init.normal_(self.vera_lambda_d)
        init_fn = th.zeros
        if self.__ut_lambda_init == "o":
            init_fn = th.ones
        elif self.__ut_lambda_init == "r":
            init_fn = th.randn
        fixed_shape = self.out_feat - self.rank
        if self.__init_scheme == "f":
            if self.rank > self.out_feat // 2 + 1:
                raise ValueError(
                    "Rank cannot be greater than Fourier embedding dimension."
                )
            fixed_shape = (self.out_feat // 2 + 1) - self.rank
        self.fixed_lambda_d = th.nn.Parameter(
            init_fn(fixed_shape), requires_grad=False
        )
        a_mat, b_mat = self.compute_bases()
        if isinstance(a_mat, th.Tensor) and isinstance(b_mat, th.Tensor):
            self.A_vera = th.nn.Parameter(a_mat, requires_grad=False)
            self.B_vera = th.nn.Parameter(b_mat, requires_grad=False)
        else:
            self.A_vera, self.B_vera = a_mat, b_mat
        self.move_to_device()

    def __fft_forward(self, x: th.Tensor) -> th.Tensor:
        """FFT-VeRA forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: VeRA-FFT Output.
        """
        dt_cf = th.cfloat
        temp = self.A_vera(x)
        l_d = th.diag(
            th.cat(
                [
                    self.vera_lambda_d.to(dtype=dt_cf),
                    self.fixed_lambda_d.to(dtype=dt_cf),
                ]
            )
        ).to(dtype=dt_cf)
        temp = temp.to(dtype=dt_cf) @ l_d
        temp = self.B_vera(temp)
        l_b = th.diag(self.vera_lambda_b)
        y_out = temp @ l_b.to(dtype=temp.dtype)
        return y_out

    def __ortho_forward(self, x: th.Tensor) -> th.Tensor:
        """Orthogonal forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: VeRA-Ortho output.
        """
        dt = x.dtype
        l_d = th.diag(
            th.cat(
                [
                    self.vera_lambda_d.to(dtype=dt),
                    self.fixed_lambda_d.to(dtype=dt),
                ]
            )
        ).to(dtype=dt)
        l_b = th.diag(self.vera_lambda_b.to(dtype=dt)).to(dtype=dt)
        temp_x = x @ self.A_vera.to(dtype=dt).T
        temp_x = temp_x @ l_d
        temp_x = temp_x @ self.B_vera.to(dtype=dt).T
        y_out = temp_x @ l_b
        return y_out

    def forward(self, x: th.Tensor, *args, **kwargs) -> th.Tensor:
        """Linear forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Linear output.
        """
        result = self.base_layer(x, *args, **kwargs)
        if self.__init_scheme == "f":
            vera_result = self.__fft_forward(x)
        else:
            vera_result = self.__ortho_forward(x)
        return result + self.alpha * vera_result


class VeraMergedLinear(th.nn.Module):
    """Vera Merged Linear Layer."""

    def __init__(
        self, base_layer: th.nn.Module, config: Dict[str, Any]
    ) -> None:
        """Process and intialise OVeRA layers.

        Args:
            base_layer (th.nn.Module): Base linear layer.
            config (Dict[str, Any]): Ortho-VeRA configutation.
        """
        super(VeraMergedLinear, self).__init__()
        _, embed_dim = th.tensor(base_layer.weight.data).shape
        self.w_q = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=False
        )
        self.w_k = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=False
        )
        self.w_v = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=False
        )
        self.__process_qkv_split(base_layer)
        self.vera_q = VeraLinear(base_layer=self.w_q, config=config)
        self.vera_k = VeraLinear(base_layer=self.w_k, config=config)
        self.vera_v = VeraLinear(base_layer=self.w_v, config=config)

    def __process_qkv_split(self, layer: th.nn.Module) -> None:
        """Split and intialise QKV layers into individual layers.

        Args:
            layer (th.nn.Module): Base layer.
        """
        base_weight = th.tensor(
            layer.weight.data
        )  # convert to tensor to make mypy happy.
        # base_weight = layer.weight.detach().clone()
        w_q, w_k, w_v = split_qkv_weight(base_weight)
        self.w_q.weight.data = w_q
        self.w_k.weight.data = w_k
        self.w_v.weight.data = w_v

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Vera merged output.
        """
        q = self.vera_q(x)
        k = self.vera_k(x)
        v = self.vera_v(x)
        qkv = th.cat([q, k, v], dim=-1)
        return qkv
