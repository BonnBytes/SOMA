"""Vera Linear Layer."""

from functools import partial
from typing import Any, Callable, Dict, Tuple, Union

import torch as th
from scipy import hadamard

from .utils import construct_wavelet_matrices, create_hippo, next_pow
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
        """
        super(VeraLinear, self).__init__(base_layer)
        self.rank = config["rank"]
        self.alpha = config["alpha"]
        self.__init_scheme = config["init_scheme"]
        if self.__init_scheme not in ["f", "w", "h", "m"]:
            raise ValueError("Improper init scheme")
        self.__ut_lambda_init = config["ut_lambda"]
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
        a_mat = partial(th.fft.rfft, norm="ortho")
        b_mat = partial(th.fft.irfft, norm="ortho")
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
        """Initialise VeRA parameters."""
        self.vera_lambda_b = th.nn.Parameter(
            th.zeros(self.out_feat), requires_grad=True
        )
        self.vera_lambda_d = th.nn.Parameter(
            th.empty(self.rank), requires_grad=True
        )
        th.nn.init.normal_(self.vera_lambda_d)
        self.fixed_lambda_d = th.nn.Parameter(
            th.zeros(self.out_feat - self.rank), requires_grad=False
        )
        self.A_vera, self.B_vera = self.compute_bases()
        self.__move_to_device()

    def __fft_forward(self, x: th.Tensor) -> th.Tensor:
        """FFT-VeRA forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: VeRA-FFT Output.
        """
        dt_cf = th.cfloat
        temp = self.A_vera(x.T)
        _, u = temp.shape
        l_d = th.diag(
            th.cat(
                [
                    self.vera_lambda_d.to(dtype=dt_cf),
                    self.fixed_lambda_d.to(dtype=dt_cf),
                ]
            )
        ).to(dtype=dt_cf)
        temp = temp @ l_d
        temp = self.B_vera(temp)
        l_b = th.diag(self.vera_lambda_b)
        y_out = temp @ l_b.to(dtype=x.dtype)
        return y_out.T

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
        H = l_b @ self.B_vera.to(dtype=dt) @ l_d @ self.A_vera.to(dtype=dt)
        y_out = H @ x.to(dtype=dt)
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
