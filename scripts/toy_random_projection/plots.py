"""Code for convergence plots."""

import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_file(fname: str) -> np.ndarray:
    """Load the convergence pickle data.

    Args:
        fname (str): File name.

    Raises:
        ValueError: Value error is convergence data not found.

    Returns:
        np.ndarray: Array of convergence data.
    """
    lists = []
    with open(fname, "rb") as fp:
        lists = pickle.load(fp)
    if len(lists) > 0:
        return np.array(lists)
    else:
        raise ValueError("List is empty")


def main():
    """Plot code."""
    r = 32
    random_mse = load_file(f"./r_mse_{r}_ut_init_z.pkl")
    wavelet_mse = load_file(f"./w_mse_{r}_ut_init_z.pkl")
    fourier_mse = load_file(f"./f_mse_{r}_ut_init_z.pkl")
    welsh_mse = load_file(f"./h_mse_{r}_ut_init_z.pkl")
    hippo_mse = load_file(f"./m_mse_{r}_ut_init_z.pkl")

    print(
        random_mse.shape,
        wavelet_mse.shape,
        fourier_mse.shape,
        welsh_mse.shape,
        hippo_mse.shape,
    )

    f_mean, f_std = np.mean(fourier_mse, axis=0), np.std(fourier_mse, axis=0)
    r_mean, r_std = np.mean(random_mse, axis=0), np.std(random_mse, axis=0)
    w_mean, w_std = np.mean(wavelet_mse, axis=0), np.std(wavelet_mse, axis=0)
    we_mean, we_std = np.mean(welsh_mse, axis=0), np.std(welsh_mse, axis=0)
    m_mean, m_std = np.mean(hippo_mse, axis=0), np.std(hippo_mse, axis=0)

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(0, len(r_mean)), r_mean, c="b", label="Random matrix")
    plt.plot(np.arange(0, len(w_mean)), w_mean, c="g", label="Wavelet basis")
    plt.plot(np.arange(0, len(f_mean)), f_mean, c="k", label="Fourier basis")
    plt.plot(
        np.arange(0, len(we_mean)), we_mean, c="r", label="Hadamard basis"
    )
    plt.plot(np.arange(0, len(m_mean)), m_mean, c="c", label="Hippo basis")
    plt.fill_between(
        np.arange(0, len(r_mean)),
        r_mean - r_std,
        r_mean + r_std,
        color="b",
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(0, len(w_mean)),
        w_mean - w_std,
        w_mean + w_std,
        color="g",
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(0, len(f_mean)),
        f_mean - f_std,
        f_mean + f_std,
        color="y",
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(0, len(we_mean)),
        we_mean - we_std,
        we_mean + we_std,
        color="r",
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(0, len(m_mean)),
        m_mean - m_std,
        m_mean + m_std,
        color="c",
        alpha=0.3,
    )
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.savefig(f"all_initialisations_{r}.png", dpi=600, bbox_inches="tight")
    plt.close()


def rank_plots():
    """Plot ranks."""
    init_scheme = "w"
    lambda_init = "z"
    ranks = [32, 64, 128, 256, 512, 1024]
    files = []
    for r in ranks:
        file_ = load_file(f"./{init_scheme}_mse_{r}_ut_init_{lambda_init}.pkl")
        files.append(file_)
    mean_vals = [np.mean(vals, axis=0) for vals in files]
    # std_vals = [np.std(vals, axis=0) for vals in files]
    colors = ["b", "g", "r", "k", "m", "y"]

    plt.figure()
    for idx in range(len(files)):
        x = np.arange(0, len(mean_vals[idx]))
        plt.plot(x, mean_vals[idx], c=colors[idx], label=f"Rank-{ranks[idx]}")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.savefig(
        f"{init_scheme}_mse_{lambda_init}_ranks.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


def init_plots():
    """Init ranks."""
    init_scheme = "m"
    r = 32
    lambda_init = ["z", "o", "r"]
    inits = ["Zeros", "Ones", "Random"]
    files = []
    colors = ["r", "g", "b"]
    for it in lambda_init:
        file_ = load_file(f"./{init_scheme}_mse_{r}_ut_init_{it}.pkl")
        files.append(file_)
    mean_vals = [np.mean(vals, axis=0) for vals in files]
    plt.figure()
    for idx in range(len(files)):
        x = np.arange(0, len(mean_vals[idx]))
        plt.plot(x, mean_vals[idx], c=colors[idx], label=f"-{inits[idx]}")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.savefig(
        f"{init_scheme}_mse_r_32.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
    # rank_plots()
    # init_plots()
