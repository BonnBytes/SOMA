"""Code for convergence plots."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch as th


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
    random_mse = load_file("./r_mse.pkl")
    wavelet_mse = load_file("./w_mse_zero.pkl")
    fourier_mse = load_file("./f_mse_zero.pkl")
    welsh_mse = load_file("./h_mse_zero.pkl")
    hippo_mse = load_file("./m_mse_zero.pkl")

    f_mean, f_std = np.mean(fourier_mse, axis=0), np.std(fourier_mse, axis=0)
    r_mean, r_std = np.mean(random_mse, axis=0), np.std(random_mse, axis=0)
    w_mean, w_std = np.mean(wavelet_mse, axis=0), np.std(wavelet_mse, axis=0)
    we_mean, we_std = np.mean(welsh_mse, axis=0), np.std(welsh_mse, axis=0)
    m_mean, m_std = np.mean(hippo_mse, axis=0), np.std(hippo_mse, axis=0)

    plt.figure()
    plt.plot(np.arange(0, len(r_mean)), r_mean, c="b", label="Random basis")
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
    plt.savefig("all_initialisations.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
