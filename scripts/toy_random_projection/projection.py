"""Random Projection Problem."""

import os
import pickle
import time
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Union

import numpy as np
import torch as th
from tqdm.auto import tqdm

from src.overa.utils import set_seed
from src.overa.vera_linear import VeraLinear

th.set_default_dtype(th.float32)


class VeraModel(th.nn.Module):
    """Vera Model."""

    def __init__(self, embed_dim: int, rank: int) -> None:
        """Intialise VeRA model.

        Args:
            embed_dim (int): Model embedding dimension.
            rank (int): Model VeRA rank.
        """
        super().__init__()
        self.base_layer = th.nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_matrix = th.nn.Parameter(
            th.empty((embed_dim, rank)), requires_grad=False
        )
        self.b_matrix = th.nn.Parameter(
            th.empty((rank, embed_dim)), requires_grad=False
        )
        self.vera_lambda_d = th.nn.Parameter(
            th.empty(rank), requires_grad=True
        )
        self.vera_lambda_b = th.nn.Parameter(
            th.empty(embed_dim), requires_grad=True
        )
        self.alpha = 1.0
        self.__init_weights()

    def __init_weights(self) -> None:
        """Initialise model."""
        th.nn.init.zeros_(self.base_layer.weight)
        th.nn.init.kaiming_uniform_(self.a_matrix)
        th.nn.init.kaiming_uniform_(self.b_matrix)
        th.nn.init.normal_(self.vera_lambda_d)
        th.nn.init.zeros_(self.vera_lambda_b)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: VeRA output.
        """
        res = self.base_layer(x)
        h = (
            self.a_matrix
            @ th.diag(self.vera_lambda_d)
            @ self.b_matrix
            @ th.diag(self.vera_lambda_b)
        )
        vera_res = x @ h
        return res + self.alpha * vera_res


class OveraModel(th.nn.Module):
    """Ortho VeRA model."""

    def __init__(self, embed_dim: int, overa_config: Dict[str, Any]) -> None:
        """Initialise model.

        Args:
            embed_dim (int): Model embedding dimension.
            overa_config (Dict[str, Any]): Ortho VeRA config.
        """
        super().__init__()
        self.__base_layer = th.nn.Linear(embed_dim, embed_dim, bias=False)
        th.nn.init.zeros_(self.__base_layer.weight)
        self.vera_layer = VeraLinear(
            base_layer=self.__base_layer, config=overa_config
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            th.Tensor: OVeRA output
        """
        return self.vera_layer(x)


def train(args: Namespace):
    """MSE objective optimisation.

    Args:
        args (Namespace): CLI arguments.
    """
    os.makedirs("./toy_results/", exist_ok=True)
    device = th.device("cpu")
    if th.cuda.is_available():
        device = th.device("cuda")
    # elif th.backends.mps.is_available():
    #     device = th.device("mps")
    print(f"Using backend: {device}")
    seed_times = []
    seed_mses = []
    for seed in range(0, 10):
        th.cuda.synchronize()
        start_time = time.perf_counter()
        print(f"Seed: {seed}")
        set_seed(seed)
        mses = []
        y = th.randn((1, 1, args.embed_dim))
        y = y.repeat((100, 1, 1))
        x = th.ones((100, 1, args.embed_dim))
        model: Union[VeraModel, OveraModel] = VeraModel(
            embed_dim=args.embed_dim, rank=args.rank
        )
        if args.init != "r":
            overa_config = {
                "rank": args.rank,
                "alpha": 1.0,
                "init_scheme": args.init,
                "ut_lambda": args.ut_lambda,
                "wavelet": "Haar",
                "transform_level": 7,
            }
            model = OveraModel(
                embed_dim=args.embed_dim, overa_config=overa_config
            )
        model = model.to(device)
        y, x = y.to(device), x.to(device)
        loss = th.nn.MSELoss()
        for name, param in model.named_parameters():
            if "vera_lambda_b" in name or "vera_lambda_d" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        opt = th.optim.Adam(model.parameters(), lr=1e-3)
        num_params = sum(
            param.numel()
            for param in model.parameters()
            if param.requires_grad
        )
        print(f"# Trainable Params: {num_params}")
        for epoch in (pbar := tqdm(range(0, args.iterations))):
            opt.zero_grad()
            out = model(x)
            loss_val = loss(out, y)
            loss_val.backward()
            opt.step()
            mses.append(loss_val.item())
            pbar.set_description(f"Epoch: {epoch} -- Loss: {loss_val.item()}")
        seed_mses.append(mses)
        th.cuda.synchronize()
        end_time = time.perf_counter()
        seed_times.append(end_time - start_time)
    mean_time = np.mean(seed_times)
    std_time = np.std(seed_times)
    print(
        f"Initialisation: {args.init}, Time mean: {mean_time}, std: {std_time}"
    )
    with open(
        f"./toy_results/{args.init}_mse_{args.rank}_ut_init_{args.ut_lambda}.pkl",
        "wb",
    ) as fp:
        pickle.dump(seed_mses, fp)


if __name__ == "__main__":
    parser = ArgumentParser(description="MSE Objective")
    parser.add_argument(
        "--init",
        type=str,
        choices=["f", "w", "h", "m", "r"],
        required=True,
        help="Intialisation method to choose.",
    )
    parser.add_argument(
        "--iterations", type=int, default=2500, help="Number of iterations."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=768, help="Full rank of the matrices."
    )
    parser.add_argument(
        "--rank", type=int, default=256, help="VeRA rank of the matrices."
    )
    parser.add_argument(
        "--ut_lambda",
        type=str,
        choices=["z", "o", "r"],
        default="z",
        help="Intialise argument for untrained lambdas.",
    )
    args = parser.parse_args()
    train(args)
