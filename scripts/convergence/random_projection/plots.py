import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch as th

def load_file(fname):
    lists = []
    with open(fname, "rb") as fp:
        lists = pickle.load(fp)
    if len(lists) > 0:
        return np.array(lists)
    else:
        raise ValueError("List is empty")


# def main():
#     random_mse = load_file("./random_mse.pkl")
#     wavelet_mse_zeros = load_file("./wavelet_mse_zeros.pkl")
#     wavelet_mse_ones = load_file("./wavelet_mse_ones.pkl")
#     wavelet_mse_randn = load_file("./wavelet_mse_randn.pkl")
#     fourier_mse_zeros = load_file("./fourier_mse_zeros.pkl")
#     fourier_mse_ones = load_file("./fourier_mse_ones.pkl")
#     fourier_mse_randn = load_file("./fourier_mse_randn.pkl")

#     # Fourier plots
#     mean_f_ones, std_f_ones = np.mean(fourier_mse_ones, axis=0), np.std(fourier_mse_ones, axis=0)
#     mean_f_zeros, std_f_zeros = np.mean(fourier_mse_zeros, axis=0), np.std(fourier_mse_zeros, axis=0)
#     mean_f_rn, std_f_rn = np.mean(fourier_mse_randn, axis=0), np.std(fourier_mse_randn, axis=0)
#     x = np.arange(0, len(mean_f_rn))
#     plt.figure()
#     plt.plot(x, mean_f_ones, c="b", label="ones init")
#     plt.plot(x, mean_f_zeros, c="g", label="zeros init")
#     plt.plot(x, mean_f_rn, '-', c="k", label="random init")
#     plt.fill_between(x, mean_f_ones - std_f_ones, mean_f_ones + std_f_ones, color="b", alpha=0.3)
#     plt.fill_between(x, mean_f_zeros - std_f_zeros, mean_f_zeros + std_f_zeros, color="g", alpha=0.3)
#     plt.fill_between(x, mean_f_rn - std_f_rn, mean_f_rn + std_f_rn, color="y", alpha=0.3)
#     plt.legend()
#     plt.xlabel("Iterations")
#     plt.ylabel("MSE")
#     plt.savefig("Fourier_results.png", dpi=600, bbox_inches="tight")
#     plt.show()

#     # Wavelet plots
#     mean_w_ones, std_w_ones = np.mean(wavelet_mse_ones, axis=0), np.std(wavelet_mse_ones, axis=0)
#     mean_w_zeros, std_w_zeros = np.mean(wavelet_mse_zeros, axis=0), np.std(wavelet_mse_zeros, axis=0)
#     mean_w_rn, std_w_rn = np.mean(wavelet_mse_randn, axis=0), np.std(wavelet_mse_randn, axis=0)
#     plt.figure()
#     plt.plot(x, mean_w_ones, c="b", label="ones init")
#     plt.plot(x, mean_w_zeros, c="g", label="zeros init")
#     plt.plot(x, mean_w_rn, '-', c="k", label="random init")
#     plt.fill_between(x, mean_w_ones - std_w_ones, mean_w_ones + std_w_ones, color="b", alpha=0.3)
#     plt.fill_between(x, mean_w_zeros - std_w_zeros, mean_w_zeros + std_w_zeros, color="g", alpha=0.3)
#     plt.fill_between(x, mean_w_rn - std_w_rn, mean_w_rn + std_w_rn, color="y", alpha=0.3)
#     plt.legend()    
#     plt.xlabel("Iterations")
#     plt.ylabel("MSE")
#     plt.savefig("wavelet_results.png", dpi=600, bbox_inches="tight")
#     plt.show()

#     # All plots
#     mean_r, std_r = np.mean(random_mse, axis=0), np.std(random_mse, axis=0)
#     plt.figure()
#     plt.plot(x, mean_r, c="b", label="Random basis")
#     plt.plot(x, mean_w_ones, c="g", label="Wavelet basis")
#     plt.plot(x, mean_f_zeros, c="k", label="Fourier basis")
#     plt.fill_between(x, mean_r - std_r, mean_r + std_r, color="b", alpha=0.3)
#     plt.fill_between(x, mean_w_ones - std_w_ones, mean_w_ones + std_w_ones, color="g", alpha=0.3)
#     plt.fill_between(x, mean_f_rn - std_f_rn, mean_f_rn + std_f_rn, color="y", alpha=0.3)
#     plt.legend()
#     plt.xlabel("Iterations")
#     plt.ylabel("MSE")
#     plt.savefig("all_initialisations.png", dpi=600, bbox_inches="tight")
#     plt.show()

def main():
    random_mse = load_file("./random_mse.pkl")
    wavelet_mse = load_file("./wavelet_mse_zero.pkl")
    fourier_mse = load_file("./fourier_mse_zero.pkl")
    welsh_mse = load_file("./welsh_mse_zero.pkl")
    hippo_mse = load_file("./hippo_mse_zero.pkl")

    f_mean, f_std = np.mean(fourier_mse, axis=0), np.std(fourier_mse, axis=0)
    r_mean, r_std = np.mean(random_mse, axis=0), np.std(random_mse, axis=0)
    w_mean, w_std = np.mean(wavelet_mse, axis=0), np.std(wavelet_mse, axis=0)
    we_mean, we_std = np.mean(welsh_mse, axis=0), np.std(welsh_mse, axis=0)
    # m_mean, m_std = np.mean(hippo_mse, axis=0), np.std(hippo_mse, axis=0)

    x = np.arange(0, len(f_mean))
    plt.figure()
    plt.plot(x, r_mean, c="b", label="Random basis")
    plt.plot(x, w_mean, c="g", label="Wavelet basis")
    plt.plot(x, f_mean, c="k", label="Fourier basis")
    plt.plot(np.arange(0, len(we_mean)), we_mean, c="r", label="Hadamard basis")
    # plt.plot(np.arange(0, len(m_mean)), m_mean, c="c", label="Hippo basis")
    plt.fill_between(x, r_mean - r_std, r_mean + r_std, color="b", alpha=0.3)
    plt.fill_between(x, w_mean - w_std, w_mean + w_std, color="g", alpha=0.3)
    plt.fill_between(x, f_mean - f_std, f_mean + f_std, color="y", alpha=0.3)
    plt.fill_between(np.arange(0, len(we_mean)), we_mean - we_std, we_mean + we_std, color="r", alpha=0.3)
    # plt.fill_between(np.arange(0, len(m_mean)), m_mean - m_std, m_mean + m_std, color="c", alpha=0.3)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.savefig("all_initialisations.png", dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()