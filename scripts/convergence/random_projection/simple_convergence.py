"""MSE objective optimisation for various inits."""

import pickle
import random
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from ptwt.matmul_transform import construct_boundary_a, construct_boundary_s
from ptwt.sparse_math import cat_sparse_identity_matrix
from scipy.linalg import hadamard
from tqdm import tqdm


def setseed(seed: int) -> None:
    """Set global seeds.

    Args:
        seed (int): Seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True


class FitModel(nn.Module):
    """Random VeRA."""

    def __init__(self, full_rank: int, rank: int) -> None:
        """Model intialisation.

        Args:
            full_rank (int): Full rank.
            rank (int): VeRA rank.
        """
        super(FitModel, self).__init__()
        self.A = th.empty((rank, full_rank))
        self.B = th.empty((full_rank, rank))
        self.__kaiming_normal_init()
        self.lambda_d = nn.Parameter(th.empty(rank), requires_grad=True)
        self.lambda_b = nn.Parameter(th.empty(full_rank), requires_grad=True)

    def init_weights(self) -> None:
        """Initialise lambdas."""
        nn.init.normal_(self.lambda_d)
        nn.init.zeros_(self.lambda_b)

    def __kaiming_normal_init(self) -> None:
        """Kaiming init A and B."""
        nn.init.kaiming_normal_(self.A)
        nn.init.kaiming_normal_(self.B)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: VeRA output.
        """
        device = x.device
        x = self.A.to(device) @ x
        x = th.diag(self.lambda_d) @ x
        x = self.B.to(device) @ x
        y_out = th.diag(self.lambda_b) @ x
        return y_out


class OrthoFit(nn.Module):
    """Orthogonal VeRA."""

    def __init__(
        self,
        full_rank: int,
        rank: int,
        untrain_lambda_init: Optional[str] = "z",
        init_scheme: Optional[str] = "f",
    ):
        """Intiialise model.

        Args:
            full_rank (int): Full Rank.
            rank (int): VeRA rank.
            untrain_lambda_init (str, optional): Initialise untrained lambdas. Defaults to zero.
            init_scheme (str, optional): Model intialisation scheme. Defaults to fourier.

        Raises:
            ValueError: Raise error in wrong initialisation scheme.
        """
        super().__init__()
        self.full_rank = full_rank
        self.rank = rank
        if init_scheme not in ["f", "w", "h", "m"]:
            raise ValueError("Improper init scheme")
        init_fn = self.fourier_inits
        if init_scheme == "w":
            init_fn = self.wavelet_inits
        elif init_scheme == "h":
            init_fn = self.welsh_inits
        elif init_scheme == "m":
            init_fn = self.hippo_inits
        self.A, self.B = init_fn()
        self.lambda_d = nn.Parameter(th.empty(rank), requires_grad=True)
        self.ut_lambdas = th.zeros(full_rank - rank)
        if untrain_lambda_init == "o":
            self.ut_lambdas = th.ones(full_rank - rank)
        elif untrain_lambda_init == "r":
            self.ut_lambdas = th.randn(full_rank - rank)
        self.lambda_b = nn.Parameter(th.empty(full_rank), requires_grad=True)
        self.init_scheme = init_scheme

    def fourier_inits(self) -> Tuple[th.Tensor, th.Tensor]:
        """Fourier initialsiation.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Fourier intialised A and B matrices.
        """
        print("Fourier init")

        def fourier_matrix(x):
            indices = th.arange(x)
            real_part = th.cos(-2 * th.pi * th.outer(indices, indices) / x)
            imag_part = th.sin(-2 * th.pi * th.outer(indices, indices) / x)
            return th.complex(real_part, imag_part)

        def inv_fourier(x):
            Q = fourier_matrix(x)
            return th.conj(Q.T) / x

        A = fourier_matrix(self.full_rank)
        B = inv_fourier(self.full_rank)
        return A, B

    def __construct_wavelet_matrices(
        self, tranform_level: int, mat_type: str
    ) -> th.Tensor:
        """Construct multi-level wavelet basis.

        Args:
            tranform_level (int): Wavelet transformation level.
            mat_type (str): Type of wavelet matrix, analysis or synthesis matrix.

        Returns:
            th.Tensor: Multi-level wavelet basis matrix.
        """
        matrices = []
        length = deepcopy(self.full_rank)
        wavelet_fn = (
            construct_boundary_a if mat_type == "a" else construct_boundary_s
        )
        for i in range(tranform_level):
            mat = wavelet_fn(wavelet="Haar", length=length)
            length = length // 2
            if i != 0:
                mat = cat_sparse_identity_matrix(mat, self.full_rank)
            matrices.append(mat)
        return self.__construct_final_matrix(mat_type, tuple(matrices))

    def __construct_final_matrix(
        self, mat_type: str, matrices: Tuple[th.Tensor, ...]
    ) -> th.Tensor:
        """Multiply multiple levels of wavelet matrices.

        Args:
            mat_type (str): Type of wavelet matrix, analysis or synthesis matrix.
            matrices (Tuple[th.Tensor, ...]): Tuple of multi-level wavelet matrices.

        Returns:
            th.Tensor: Multi-level wavelet basis matrix.
        """
        final_matrix = th.eye(self.full_rank, dtype=matrices[0].dtype)
        matrices = matrices if mat_type == "s" else matrices[::-1]
        for mat in matrices:
            final_matrix = final_matrix @ mat
        return final_matrix

    def wavelet_inits(self) -> Tuple[th.Tensor, th.Tensor]:
        """Wavelet initialisation.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Wavelet intialised A and B matrices.
        """
        print("Wavelet init")
        transform_level = 3
        A = self.__construct_wavelet_matrices(transform_level, "a").to_dense()
        B = self.__construct_wavelet_matrices(transform_level, "s").to_dense()
        return A, B

    def welsh_inits(self) -> Tuple[th.Tensor, th.Tensor]:
        """Welsh-Hadamard intialisation.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Welsh intialised A and B matrices.
        """
        print("Welsh-hadamard init")
        power = self.next_pow(self.full_rank)
        A = th.from_numpy(hadamard(power))
        B = th.from_numpy(hadamard(power))
        A = A[: self.full_rank, : self.full_rank]
        B = A.T
        return A, B

    def next_pow(self, n: int) -> int:
        """Find next power of 2.

        Args:
            n (int): Number of interest.

        Returns:
            int: Next nearest number with power of 2.
        """
        if n <= 0:
            return 1
        if n & (n - 1) == 0:
            return n
        return 1 << n.bit_length()

    def hippo_inits(self) -> Tuple[th.Tensor, th.Tensor]:
        """Hippo Intialisation.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Hippo intialised A and B matrices.
        """
        print("Hippo init")

        def create_hippo(num: int) -> th.Tensor:
            """Create Hippo matrix.

            Args:
                num (int): matrix size.

            Returns:
                th.Tensor: Hippo matrix.
            """
            matrix = th.zeros((num, num))
            for n in range(num):
                for k in range(num):
                    if n < k:
                        matrix[n, k] = 0
                    elif n == k:
                        matrix[n, k] = n + 1
                    elif n > k:
                        matrix[n, k] = ((2 * n + 1) ** 0.5) * (
                            (2 * k + 1) ** 0.5
                        )
            return matrix

        A = create_hippo(self.full_rank)
        B = th.linalg.inv(A)
        return A, B

    def init_weights(self):
        """Initialise lamdbas."""
        nn.init.normal_(self.lambda_d)
        nn.init.zeros_(self.lambda_b)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: Ortho-VeRA output.
        """
        device = x.device
        if self.init_scheme != "f":
            dt = th.float32
        elif self.init_scheme == "f":
            dt = th.cfloat
        l_d = th.diag(
            th.cat(
                [
                    self.lambda_d.to(dtype=dt),
                    self.ut_lambdas.to(dtype=dt).to(device),
                ]
            )
        ).to(dtype=dt)
        l_b = th.diag(self.lambda_b.to(dtype=dt)).to(dtype=dt)
        H = (
            l_b
            @ self.B.to(device).to(dtype=dt)
            @ l_d
            @ self.A.to(device).to(dtype=dt)
        )
        y_out = H @ x.to(dtype=dt)
        if self.init_scheme == "f":
            return y_out.real.to(th.float32)
        return y_out.to(th.float32)


def main(args: Namespace):
    """MSE objective optimisation.

    Args:
        args (Namespace): CLI arguments.
    """
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
    print(f"Using backend: {device}")
    seed_mses = []
    for seed in range(0, 10):
        print(f"Seed: {seed}")
        setseed(seed)
        mses = []
        y = th.randn((args.full_rank, 1))
        x = th.ones((args.full_rank, 1))
        model = (
            FitModel(args.full_rank, args.rank)
            if args.init == "r"
            else OrthoFit(
                args.full_rank, args.rank, args.untrain_lambda_init, args.init
            )
        )
        model.init_weights()
        model = model.to(device)
        y, x = y.to(device), x.to(device)
        loss = nn.MSELoss()
        opt = th.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in (pbar := tqdm(range(0, args.iterations))):
            opt.zero_grad()
            out = model(x)
            loss_val = loss(out, y)
            loss_val.backward()
            opt.step()
            mses.append(loss_val.item())
            pbar.set_description(f"Epoch: {epoch} -- Loss: {loss_val.item()}")
        seed_mses.append(mses)
    with open(
        f"{args.init}_mse_{args.rank}_ut_init_{args.untrain_lambda_init}.pkl",
        "wb",
    ) as fp:
        pickle.dump(seed_mses, fp)


if __name__ == "__main__":
    parser = ArgumentParser(description="MSE Objektive")
    parser.add_argument(
        "--init",
        type=str,
        choices=["f", "w", "h", "m", "r"],
        required=True,
        help="Intialisation method to choose",
    )
    parser.add_argument(
        "--iterations", type=int, default=2500, help="Number of iterations"
    )
    parser.add_argument(
        "--full_rank", type=int, default=768, help="Full rank of the matrices"
    )
    parser.add_argument(
        "--rank", type=int, default=256, help="VeRA rank of the matrices"
    )
    parser.add_argument(
        "--untrain_lambda_init",
        type=str,
        choices=["z", "o", "r"],
        default="z",
        help="Intialise argument for untrained lambdas.",
    )
    args = parser.parse_args()
    main(args)
