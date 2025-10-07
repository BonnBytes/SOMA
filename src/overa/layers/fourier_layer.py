"""Fourier VeRA Linear Layer."""

from functools import partial
from typing import Any, Dict, Tuple

import torch as th

from ..vera_base import VeraBase


class FourierLinear(VeraBase):
    """Fourier based VeRA Layer."""

    def __init__(
        self, base_layer: th.nn.Module, config: Dict[str, Any]
    ) -> None:
        """Initialise Fourier layer."""
        super(FourierLinear, self).__init__(base_layer)
        self.rank = config["rank"]
        self.alpha = config["alpha"]
        self.update_layer()

    def compute_bases(self) -> Tuple[Any, Any]:
        """Compute rFFT bases.

        Returns:
            Tuple[Any, Any]: Tuple containing fourier matrices.
        """
        a_mat = partial(th.fft.rfft, norm="ortho")
        b_mat = partial(th.fft.irfft, norm="ortho")
        return a_mat, b_mat

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
        """FFT VeRA forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: VeRA-FFT Output.
        """
        dt_cf = th.cfloat
        dt_32 = th.float32
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
        y_out = temp @ l_b.to(dtype=dt_32)
        return y_out.T

    def forward(self, x: th.Tensor, *args, **kwargs) -> th.Tensor:
        """Linear forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Linear output.
        """
        result = self.base_layer(x, *args, **kwargs)
        vera_result = self.__fft_forward(x)
        return result + self.alpha * vera_result
