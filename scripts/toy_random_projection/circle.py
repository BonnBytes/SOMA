"""Unit cirle example."""

import os
import random
from glob import glob
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from PIL import Image

DIM: int = 2


class Model(th.nn.Module):
    """Model."""

    def _init_(
        self, dim: Optional[int] = DIM, init: Optional[str] = "r"
    ) -> None:
        """Initialise.

        Args:
            dim (int, optional): model dimension. Defaults to DIM.
            init (str, optional): Initialisation. Defaults to "r".

        Raises:
            ValueError: Raises Value error if init donot exist.
        """
        super().__init__()
        self.dim = dim
        self.A = th.empty((self.dim, self.dim))
        self.B = th.empty((self.dim, self.dim))
        self.lambdas = th.nn.Parameter(th.ones(self.dim))
        self.b_lambdas = th.nn.Parameter(th.ones(self.dim))
        if init == "r":
            th.nn.init.kaiming_uniform_(self.A)
            th.nn.init.kaiming_uniform_(self.B)
        elif init == "o":
            self.A = th.tensor([[1.0, 1.0], [1.0, -1.0]])
            self.B = self.A.T
            self.A = self.A / (self.dim**0.5)
            self.B = self.B / (self.dim**0.5)
            print(
                "Hadamard orthogonal check:",
                th.allclose(self.A @ self.A.T, th.eye(self.dim), atol=1e-7),
                th.allclose(self.B @ self.B.T, th.eye(self.dim), atol=1e-7),
            )
        elif init == "ilr":
            th.nn.init.kaiming_uniform_(self.A)
            th.nn.init.kaiming_uniform_(self.B)
            # make column linear dependence
            self.A[0, :] = -0.5 * self.A[1, :] + 0.01
        else:
            raise ValueError(f"Unknown init: {init}")
        self.bias = th.nn.Parameter(th.zeros(self.dim))

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward.

        Args:
            x (th.Tensor): Input tensor.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Output map.
        """
        interm = self.A.to(x.device) @ x
        y = th.diag(self.lambdas) @ interm
        y = self.B.to(x.device) @ y
        y = th.diag(self.b_lambdas) @ y
        return y + self.bias.reshape(-1, 1), interm


def gen_circle_points(n: Optional[int] = 50000) -> th.Tensor:
    """Generate circle points.

    Args:
        n (int, optional): num of points. Defaults to 50000.

    Returns:
        th.Tensor: Circle points.
    """
    rad = 1.0
    theta = th.linspace(0, 2 * th.pi, n)
    circle = th.stack((rad * th.cos(theta), rad * th.sin(theta)), dim=0)
    return circle


def mx_proj(circle_x: th.Tensor) -> th.Tensor:
    """Project circle.

    Args:
        circle_x (th.Tensor): Input circle.

    Returns:
        th.Tensor: Mapped circle.
    """
    s_x, s_y = 3.0, 0.5
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R = th.tensor([[c, -s], [s, c]], dtype=th.float32)
    S = th.tensor([[s_x, 0], [0, s_y]], dtype=th.float32)
    M = R @ S
    return M @ circle_x + 2.5


def plot_projection(
    circle_x: np.ndarray,
    circle_y: np.ndarray,
    pred: np.ndarray,
    epoch: int,
    init: str,
) -> None:
    """Plot circles.

    Args:
        circle_x (np.ndarray): Input circle.
        circle_y (np.ndarray): Ground truth.
        pred (np.ndarray): Prediction.
        epoch (int): Epoch.
        init (str): Initialision.
    """
    init_dict = {
        "r": "Kaiming Uniform Init",
        "o": "Orthogonal Init",
        "ilr": "Ill conditioned Random Init",
    }

    plt.plot(
        circle_x[0], circle_x[1], label="Input", zorder=2, color="tab:blue"
    )
    plt.plot(
        circle_y[0], circle_y[1], label="GT", zorder=2, color="tab:orange"
    )
    plt.plot(pred[0], pred[1], label="Pred", zorder=2, color="tab:green")
    plt.scatter(
        pred[0, 0],
        pred[1, 0],
        marker="D",
        s=50,
        color="tab:green",
        edgecolors="black",
        zorder=5,
    )
    plt.axis("equal")
    plt.grid(True, alpha=0.5, zorder=-2)
    plt.title(f"{init_dict[init]}: Iteration {epoch + 1}")
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False
    )
    os.makedirs(f"./toy_tri/{init}_images", exist_ok=True)
    plt.savefig(
        f"./toy_tri/{init}images/proj{epoch + 1}.png", bbox_inches="tight"
    )
    plt.close()


def save_gif_from_pngs(
    images_dir: str,
    output_name: str,
    slow_duration: Optional[int] = 800,
    fast_duration: Optional[int] = 100,
) -> None:
    """Save png to GIFs.

    Args:
        images_dir (str): Input image directory.
        output_name (str): Output name.
        slow_duration (Optional[int], optional): Duration. Defaults to 800.
        fast_duration (Optional[int], optional): Duratrion. Defaults to 100.
    """
    frame_paths = sorted(
        glob(os.path.join(images_dir, "proj_*.png")),
        key=lambda p: int(
            os.path.splitext(os.path.basename(p))[0].split("_")[-1]
        ),
    )

    if not frame_paths:
        print("No PNG frames found; skipping GIF export.")
        return

    frames = [Image.open(path).convert("RGB") for path in frame_paths]

    durations = []
    for i in range(len(frames)):
        if i < 20:
            durations.append(slow_duration)
        else:
            durations.append(fast_duration)

    output_path = os.path.join("./toy_tri/", output_name)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )

    for frame in frames:
        frame.close()

    print(f"Saved GIF: {output_path}")


def main():
    """Train.

    Raises:
        ValueError: Error if unknown init is given.
    """
    th.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    circle_x = gen_circle_points()
    circle_y = mx_proj(circle_x)
    init = "o"
    if init == "r":
        itrs = 3000
    elif init == "o":
        itrs = 1000
        # itrs = 3000
    elif init == "ilr":
        itrs = 2000
    else:
        raise ValueError(f"Unknown init: {init}")
    model = Model(init=init)
    model = model.to(device)
    circle_x = circle_x.to(device)
    circle_y = circle_y.to(device)
    lambda_b = []
    lambda_d = []
    optimizer = th.optim.SGD(model.parameters(), lr=1e-2)
    for epoch in range(itrs):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(circle_x)
        if epoch % 10 == 0 or epoch < 10:
            plot_projection(
                circle_x.detach().cpu().numpy(),
                circle_y.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                epoch,
                init,
            )
        lambda_b.append(model.b_lambdas.detach().cpu().numpy())
        lambda_d.append(model.lambdas.detach().cpu().numpy())
        loss = th.nn.functional.mse_loss(pred.T, circle_y.T)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    save_gif_from_pngs(f"./toy_tri/{init}_images", f"{init}_projection.gif")
    lambda_b = np.array(lambda_b)
    lambda_d = np.array(lambda_d)
    init_dict = {
        "r": "Kaiming Uniform Init",
        "o": "Orthogonal Init",
        "ilr": "Ill conditioned Random Init",
    }
    plt.figure(figsize=(10, 4))
    plt.plot(lambda_d[:, 0], label="Lambda 1")
    plt.plot(lambda_d[:, 1], label="Lambda 2")
    plt.title(f"{init_dict.get(init, init)} - Lambda_d over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Lambda Value")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(f"./toy_tri/{init}_lambda_d.png", bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.plot(lambda_b[:, 0], label="Lambda 1")
    plt.plot(lambda_b[:, 1], label="Lambda 2")
    plt.title(f"{init_dict.get(init, init)} - Lambda_b over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Lambda Value")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(f"./toy_tri/{init}_lambda_b.png", bbox_inches="tight")
    plt.close()


if __name__ == "_main_":
    main()
