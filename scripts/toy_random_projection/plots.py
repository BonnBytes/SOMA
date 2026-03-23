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
    base_folder = "./toy_results/"
    r = 32
    random_mse = load_file(f"{base_folder}r_mse_{r}_ut_init_z.pkl")
    wavelet_mse = load_file(f"{base_folder}w_mse_{r}_ut_init_z.pkl")
    fourier_mse = load_file(f"{base_folder}f_mse_{r}_ut_init_z.pkl")
    welsh_mse = load_file(f"{base_folder}h_mse_{r}_ut_init_z.pkl")
    # hippo_mse = load_file(f"{base_folder}m_mse_{r}_ut_init_z.pkl")

    print(
        random_mse.shape,
        wavelet_mse.shape,
        fourier_mse.shape,
        welsh_mse.shape,
        # hippo_mse.shape,
    )

    f_mean, f_std = np.mean(fourier_mse, axis=0), np.std(fourier_mse, axis=0)
    r_mean, r_std = np.mean(random_mse, axis=0), np.std(random_mse, axis=0)
    w_mean, w_std = np.mean(wavelet_mse, axis=0), np.std(wavelet_mse, axis=0)
    we_mean, we_std = np.mean(welsh_mse, axis=0), np.std(welsh_mse, axis=0)
    # m_mean, m_std = np.mean(hippo_mse, axis=0), np.std(hippo_mse, axis=0)

    _, ax = plt.subplots()
    ax.plot(
        np.arange(0, len(r_mean)), r_mean, c="tab:blue", label="Random matrix"
    )
    ax.plot(
        np.arange(0, len(f_mean)), f_mean, c="tab:brown", label="Fourier basis"
    )
    ax.plot(
        np.arange(0, len(w_mean)),
        w_mean,
        "--",
        c="tab:cyan",
        label="Wavelet basis",
    )
    ax.plot(
        np.arange(0, len(we_mean)),
        we_mean,
        c="tab:red",
        label="Hadamard basis",
    )
    # ax.plot(np.arange(0, len(m_mean)), m_mean, c="c", label="Hippo basis")
    ax.fill_between(
        np.arange(0, len(r_mean)),
        np.maximum(0, r_mean - r_std),
        r_mean + r_std,
        color="tab:blue",
        alpha=0.3,
    )
    ax.fill_between(
        np.arange(0, len(w_mean)),
        np.maximum(0, w_mean - w_std),
        w_mean + w_std,
        color="tab:green",
        alpha=0.3,
    )
    ax.fill_between(
        np.arange(0, len(f_mean)),
        np.maximum(0, f_mean - f_std),
        f_mean + f_std,
        color="tab:brown",
        alpha=0.3,
    )
    ax.fill_between(
        np.arange(0, len(we_mean)),
        np.maximum(0, we_mean - we_std),
        we_mean + we_std,
        color="tab:red",
        alpha=0.3,
    )
    # ax.fill_between(
    #     np.arange(0, len(m_mean)),
    #     m_mean - m_std,
    #     m_mean + m_std,
    #     color="c",
    #     alpha=0.3,
    # )
    ax.legend()
    time_eval = 2000
    axin = ax.inset_axes([0.48, 0.25, 0.5, 0.4])
    axin.semilogy(
        np.arange(0, len(r_mean))[time_eval:], r_mean[time_eval:], c="tab:blue"
    )
    axin.semilogy(
        np.arange(0, len(f_mean))[time_eval:],
        f_mean[time_eval:],
        c="tab:brown",
    )
    axin.semilogy(
        np.arange(0, len(w_mean))[time_eval:],
        w_mean[time_eval:],
        "--",
        c="tab:cyan",
    )
    axin.semilogy(
        np.arange(0, len(we_mean))[time_eval:],
        we_mean[time_eval:],
        c="tab:red",
    )
    # axin.semilogy(np.arange(0, len(m_mean))[time_eval:], m_mean[time_eval:], c="c")
    axin.tick_params(axis="both", labelsize=7)
    # axin.grid()
    ax.indicate_inset_zoom(axin, alpha=1.0, linewidth=2.0, edgecolor="k")

    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.grid()
    plt.savefig(
        f"{base_folder}all_initialisations_{r}.png",
        dpi=600,
        bbox_inches="tight",
    )
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
