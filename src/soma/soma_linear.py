"""Vera Linear Layer."""

from functools import partial
from typing import Any, Callable, Dict, Tuple, Union, cast

import pywt
import torch as th

from .transforms import fast_wavelet_transform, fwht_optimized
from .utils import split_qkv_weight
from .vera_base import VeraBase


class SomaLinear(VeraBase):
    """SOMA Linear Layer."""

    def __init__(
        self, base_layer: th.nn.Module, config: Dict[str, Any]
    ) -> None:
        """Initialise SOMA Linear.

        Args:
            base_layer (th.nn.Module): Base linear layer.
            config (Dict[str, Any]): SOMA Config.

        Raises:
            ValueError: Error if ortho init is outside of selected ones.
            ValueError: Error if untrained lambdas as not properly intialised.
        """
        super(SomaLinear, self).__init__(base_layer)
        self.rank = config["rank"]
        self.alpha = config["alpha"]
        self.__init_scheme = config["init_scheme"]
        if self.__init_scheme not in ["f", "w", "h", "m", "vo"]:
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
        else:
            return self.fourier_bases()

    def wavelet_bases(self) -> Tuple[Callable, ...]:
        """Compute Wavelet bases.

        Returns:
            Tuple[th.Tensor, ...]: Tuple containing wavelet bases matrices.
        """
        wlet = pywt.Wavelet(self.__wavelet)
        a_mat = partial(
            fast_wavelet_transform,
            inverse=False,
            sizes=None,
            wavelet=wlet,
            transform_level=self.__transform_level,
        )
        b_mat = partial(
            fast_wavelet_transform,
            inverse=True,
            wavelet=wlet,
            transform_level=self.__transform_level,
        )
        return a_mat, b_mat

    def fourier_bases(self) -> Tuple[Callable, ...]:
        """Compute rFFT bases.

        Returns:
            Tuple[Callable, ...]: Tuple containing fourier matrices.
        """
        a_mat = partial(th.fft.rfft, norm="ortho", n=self.in_feat)
        b_mat = partial(th.fft.irfft, norm="ortho", n=self.out_feat)
        return a_mat, b_mat

    def welsh_bases(self) -> Tuple[Callable, ...]:
        """Compute Wavelet transform bases.

        Returns:
            Tuple[th.Tensor, ...]: Tuple containing wavelet matrices.
        """
        a_mat = partial(fwht_optimized, inverse=False)
        b_mat = partial(fwht_optimized, inverse=True)
        return a_mat, b_mat

    def update_layer(self) -> None:
        """Initialise SOMA parameters.

        Raises:
            ValueError: Error if rank is greater than min(in_feat, out_feat).
            ValueError: Error if rank is greater than Fourier embedding dimension.
        """
        self.vera_lambda_b = th.nn.Parameter(
            th.zeros(self.out_feat), requires_grad=True
        )
        self.vera_lambda_d = th.nn.Parameter(
            th.empty(self.rank), requires_grad=True
        )
        th.nn.init.constant_(self.vera_lambda_d, 1e-7)
        init_fn = th.zeros
        if self.__ut_lambda_init == "o":
            init_fn = th.ones
        elif self.__ut_lambda_init == "r":
            init_fn = th.randn
        fixed_shape = self.out_feat - self.rank
        if self.__init_scheme == "f":
            if self.rank > self.in_feat // 2 + 1:
                raise ValueError(
                    "Rank cannot be greater than Fourier embedding dimension."
                )
            fixed_shape = (self.in_feat // 2 + 1) - self.rank
        self.fixed_lambda_d = th.nn.Parameter(
            init_fn(fixed_shape), requires_grad=False
        )
        a_mat, b_mat = self.compute_bases()
        if isinstance(a_mat, th.Tensor) and isinstance(b_mat, th.Tensor):
            self.A_vera = th.nn.Parameter(a_mat, requires_grad=False)
            self.B_vera = th.nn.Parameter(b_mat, requires_grad=False)
        else:
            self.A_vera, self.B_vera = a_mat, b_mat

    def __fft_forward(self, x: th.Tensor) -> th.Tensor:
        """FFT-SOMA forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: SOMA-FFT Output.
        """
        orig_dtype = x.dtype
        x_fft = x.to(th.float32) if orig_dtype == th.float16 else x
        temp = self.A_vera(x_fft)
        lambda_d = th.cat([self.vera_lambda_d, self.fixed_lambda_d]).to(
            dtype=temp.dtype
        )
        freq_dim = self.in_feat // 2 + 1
        if lambda_d.numel() < freq_dim:
            lambda_d = th.cat(
                [lambda_d, temp.new_zeros(freq_dim - lambda_d.numel())]
            )
        else:
            lambda_d = lambda_d[:freq_dim]
        temp = temp * lambda_d
        temp = self.B_vera(temp)
        y_out = temp * self.vera_lambda_b.to(dtype=temp.dtype)
        if orig_dtype == th.float16:
            y_out = y_out.to(orig_dtype)
        return y_out

    def __hadamard_forward(self, x: th.Tensor) -> th.Tensor:
        """Hadamard-SOMA forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: SOMA-Hadamard Output.
        """
        temp = self.A_vera(x)
        l_d_vec = th.cat([self.vera_lambda_d, self.fixed_lambda_d]).to(
            dtype=temp.dtype
        )
        temp = temp * l_d_vec
        temp = self.B_vera(temp)
        l_b_vec = self.vera_lambda_b.to(dtype=temp.dtype)
        y_out = temp * l_b_vec
        return y_out

    def __wavelet_forward(self, x: th.Tensor) -> th.Tensor:
        temp, sizes = self.A_vera(x)
        l_d_vec = th.cat([self.vera_lambda_d, self.fixed_lambda_d]).to(
            dtype=temp.dtype
        )
        temp = temp * l_d_vec
        temp = self.B_vera(temp, sizes=sizes)
        l_b_vec = self.vera_lambda_b.to(dtype=temp.dtype)
        y_out = temp * l_b_vec
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
            vera_result = self.__fft_forward(x)  #
        elif self.__init_scheme == "h":
            vera_result = self.__hadamard_forward(x)
        elif self.__init_scheme == "w":
            vera_result = self.__wavelet_forward(x)
        return result + self.alpha * vera_result


class SomaMergedLinear(th.nn.Module):
    """Soma Merged Linear Layer."""

    def __init__(
        self, base_layer: th.nn.Module, config: Dict[str, Any]
    ) -> None:
        """Process and intialise SOMA layers.

        Args:
            base_layer (th.nn.Module): Base linear layer.
            config (Dict[str, Any]): SOMA configutation.
        """
        super(SomaMergedLinear, self).__init__()
        qkv_layer = cast(th.nn.Linear, base_layer)
        embed_dim = qkv_layer.in_features
        has_bias = qkv_layer.bias is not None
        self.w_q = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=has_bias
        )
        self.w_k = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=has_bias
        )
        self.w_v = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=has_bias
        )
        self.__process_qkv_split(qkv_layer)
        self.vera_q = SomaLinear(base_layer=self.w_q, config=config)
        self.vera_k = SomaLinear(base_layer=self.w_k, config=config)
        self.vera_v = SomaLinear(base_layer=self.w_v, config=config)

    def __process_qkv_split(self, layer: th.nn.Module) -> None:
        """Split and intialise QKV layers into individual layers.

        Args:
            layer (th.nn.Module): Base layer.

        Raises:
            ValueError: Error if qkv bias shape is improper.
        """
        qkv_layer = cast(th.nn.Linear, layer)
        base_weight = qkv_layer.weight.detach().clone()
        w_q, w_k, w_v = split_qkv_weight(base_weight)
        self.w_q.weight.data = w_q
        self.w_k.weight.data = w_k
        self.w_v.weight.data = w_v

        if qkv_layer.bias is not None:
            base_bias = qkv_layer.bias.detach().clone()
            if base_bias.numel() != 3 * self.w_q.out_features:
                raise ValueError("Improper qkv bias tensor.")
            b_q, b_k, b_v = base_bias.reshape(3, self.w_q.out_features).unbind(
                0
            )
            self.w_q.bias.data = b_q
            self.w_k.bias.data = b_k
            self.w_v.bias.data = b_v

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


class SomaMergedLinearQV(th.nn.Module):
    """Soma merged linear layer with SOMA on Q and V only."""

    def __init__(
        self, base_layer: th.nn.Module, config: Dict[str, Any]
    ) -> None:
        """Process and initialise QV-only SOMA layers.

        Args:
            base_layer (th.nn.Module): Base merged qkv layer.
            config (Dict[str, Any]): SOMA configuration.
        """
        super(SomaMergedLinearQV, self).__init__()
        qkv_layer = cast(th.nn.Linear, base_layer)
        embed_dim = qkv_layer.in_features
        has_bias = qkv_layer.bias is not None
        self.w_q = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=has_bias
        )
        self.w_k = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=has_bias
        )
        self.w_v = th.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, bias=has_bias
        )
        self.__process_qkv_split(qkv_layer)
        self.vera_q = SomaLinear(base_layer=self.w_q, config=config)
        self.vera_v = SomaLinear(base_layer=self.w_v, config=config)

    def __process_qkv_split(self, layer: th.nn.Module) -> None:
        """Split and initialise QKV layers into individual layers.

        Args:
            layer (th.nn.Module): Base layer.

        Raises:
            ValueError: Error if qkv bias shape is improper.
        """
        qkv_layer = cast(th.nn.Linear, layer)
        base_weight = qkv_layer.weight.detach().clone()
        w_q, w_k, w_v = split_qkv_weight(base_weight)
        self.w_q.weight.data = w_q
        self.w_k.weight.data = w_k
        self.w_v.weight.data = w_v

        if qkv_layer.bias is not None:
            base_bias = qkv_layer.bias.detach().clone()
            if base_bias.numel() != 3 * self.w_q.out_features:
                raise ValueError("Improper qkv bias tensor.")
            b_q, b_k, b_v = base_bias.reshape(3, self.w_q.out_features).unbind(
                0
            )
            self.w_q.bias.data = b_q
            self.w_k.bias.data = b_k
            self.w_v.bias.data = b_v

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Vera merged output.
        """
        q = self.vera_q(x)
        k = self.w_k(x)
        v = self.vera_v(x)
        return th.cat([q, k, v], dim=-1)
