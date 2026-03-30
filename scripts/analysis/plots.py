"""Generate plots for analysis."""

import os
from argparse import ArgumentParser

import torch as th
import numpy as np
import matplotlib.pyplot as plt


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--kaiming",
        default="/home/loki/PhD/OVeRA_results/vera_weights_k.pt",
        help = "Path to pretrained weights for Kaiming"
    )
    parser.add_argument(
        "--fourier",
        default="/home/loki/PhD/OVeRA_results/vera_weights_f.pt",
        help = "Path to pretrained weights for Fourier"
    )
    parser.add_argument(
        "--wavelet",
        default="/home/loki/PhD/OVeRA_results/vera_weights_w.pt",
        help = "Path to pretrained weights for Wavelet"
    )
    parser.add_argument(
        "--hadamard",
        default="/home/loki/PhD/OVeRA_results/vera_weights_h.pt",
        help = "Path to pretrained weights for Hadamard"
    )
    parser.add_argument(
        "--plot_type",
        default="singular_values",
        choices=["singular_values", "matrix_rank", "condition_number"],
        help = "Type of plot to generate"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help = "Directory to save the generated plots"
    )    
    return parser.parse_args()


def plot_matrix_ranks(weights, colors, output_dir):
    for layer in ["q", "v"]:
        plt.figure()
        for init_scheme, weight_dict in weights.items():
            max_iter = max(weight_dict.keys())
            iter_weights = weight_dict[max_iter]
            print(f"Init scheme: {init_scheme}, max iteration: {max_iter}")
            idx = 0
            x_layers = []
            ranks = []
            for _, weights_qv in iter_weights.items():
                w = weights_qv[layer]
                svs = th.linalg.svdvals(w)
                # Normalize using squared singular values (energy)
                p = svs
                p = p / p.sum()
                # Compute entropy safely
                entropy = -(p * (p.clamp(min=1e-12).log())).sum()
                rank = np.exp(entropy)
                idx += 1
                x_layers.append(idx)
                ranks.append(rank)
            plt.plot(x_layers, ranks, label=f"{init_scheme}", color=colors[init_scheme], linewidth=2)
        plt.xlabel("Layer")
        plt.ylabel("Matrix Rank")
        plt.title(f"Matrix Rank of Delta W for {layer.upper()} Layers")
        plt.legend()
        plt.grid()
        file_name = os.path.join(output_dir, f"matrix_rank_{layer}.png")
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {file_name}")
    

def plot_cond_number(weights, colors, output_dir):
    for layer in ["q", "v"]:
        plt.figure()
        for init_scheme, weight_dict in weights.items():
            max_iter = max(weight_dict.keys())
            iter_weights = weight_dict[max_iter]
            print(f"Init scheme: {init_scheme}, max iteration: {max_iter}")
            idx = 0
            x_layers = []
            conds = []
            for _, weights_qv in iter_weights.items():
                w = weights_qv[layer]
                svs = th.linalg.svdvals(w)
                svs, _ = th.sort(svs, descending=True)
                k = 256  # or 50, or 128
                kappa = svs[0] / (svs[k] + 1e-12)
                idx += 1
                x_layers.append(idx)
                conds.append(kappa.item())
            plt.plot(x_layers, conds, label=f"{init_scheme}", color=colors[init_scheme], linewidth=2)
        plt.xlabel("Layer")
        plt.ylabel("Condition Number")
        plt.title(f"Condition Number of Delta W for {layer.upper()} Layers")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        file_name = os.path.join(output_dir, f"condition_number_{layer}.png")
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {file_name}")


def plot_singular_values(weights, output_dir):
    for layer in ["q", "v"]:
        for block in range(12):
            plt.figure(figsize=(20, 10))
            s_ix = 1
            for init_scheme, weight_dict in weights.items():
                x_iters = []
                s_vals = []
                for iter, weights_blocks in weight_dict.items():
                    if iter > 144:
                        continue
                    key = f"blocks.{block}"
                    if key not in weights_blocks:
                        key = f"block.{block}"
                    w = weights_blocks[key]
                    w = w[layer]
                    svs = th.linalg.svdvals(w)[:25].cpu().numpy()
                    x_iters.append(iter)
                    s_vals.append(svs)
                s_vals = np.asarray(s_vals)
                plt.subplot(2, 2, s_ix)
                s_ix += 1
                sv_colors = plt.cm.viridis(np.linspace(0.0, 1.0, s_vals.shape[1]))
                for i in range(s_vals.shape[1]):
                    plt.plot(x_iters, s_vals[:, i], label=f"Singular Value {i+1}", color=sv_colors[i])
                plt.xlabel("Iteration")
                plt.ylabel("Singular Value Magnitude")
                plt.title(f"{init_scheme}")
                plt.grid()
            plt.suptitle(f"Block:{block+1}: {layer.upper()}-Layer top-25 Singular Values")
            file_name = os.path.join(output_dir, f"{layer}/singular_values_block_{block}_{layer}.png")
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            plt.savefig(file_name, dpi=600, bbox_inches="tight")
            plt.close()
            print(f"Saved plot to {file_name}")


def main():
    args = _parse_args()
    weights = {
        "Kaiming": th.load(args.kaiming),
        "Fourier": th.load(args.fourier),
        "Wavelet": th.load(args.wavelet),
        "Hadamard": th.load(args.hadamard)
    }
    colors = {
        "Kaiming": "tab:cyan",
        "Fourier": "tab:blue",
        "Wavelet": "tab:green",
        "Hadamard": "tab:red"
    }
    args.output_dir = os.path.join(args.output_dir, args.plot_type)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot_type == "matrix_rank":
        plot_matrix_ranks(weights, colors, args.output_dir)
    elif args.plot_type == "condition_number":
        plot_cond_number(weights, colors, args.output_dir)
    elif args.plot_type == "singular_values":
        plot_singular_values(weights, args.output_dir)


if __name__ == "__main__":
    main()