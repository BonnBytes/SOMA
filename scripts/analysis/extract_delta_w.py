"""Extract Delta W from a trained model."""

import os
from glob import glob
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

import torch as th
from timm.models import create_model

from scripts.image_classification.data_config import CONFIG
from scripts.image_classification.utils import replace_qkv
from src.overa.utils import welsh_hadamard_transform

def _parse_args():
    """Parse CLI arguments."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default=None,
        required=True,
        help=(
            "Directory containing checkpoints named like vit_iter_{iter}.pt; "
            "output will be {iter: {layer: {q_delta_w/...}}}."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="cifar",
        type=str,
        choices=["cifar", "flowers", "resisc", "food"],
        help="Dataset used when training the checkpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "large"],
        help="Backbone model variant.",
    )
    parser.add_argument("--rank", default=256, type=int, help="OVeRA rank")
    parser.add_argument(
        "--init",
        type=str,
        choices=["f", "w", "h", "k"],
        default="f",
        help="VeRA initialisation scheme used for training.",
    )
    parser.add_argument(
        "--ut_lambda",
        type=str,
        choices=["z", "o", "r"],
        default="z",
        help="Untrained lambda initialisation scheme.",
    )
    parser.add_argument(
        "--wavelet", type=str, default="Haar", help="Wavelet family"
    )
    parser.add_argument(
        "--wavelet_level",
        type=int,
        default=8,
        help="Wavelet transform level.",
    )
    parser.add_argument(
        "--target_proj",
        type=str,
        choices=["qv", "qkv"],
        default="qv",
        help="Attention projections adapted with VeRA in training.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output path for singular-value dictionary (.pt).",
    )
    return parser.parse_args()


def _build_model(args) -> th.nn.Module:
    """Build a ViT model with VeRA layers configured to match training."""
    if args.model == "large":
        model = create_model("vit_large_patch16_224_in21k", pretrained=True)
    else:
        model = create_model("vit_base_patch16_224.augreg_in21k", pretrained=True)

    overa_config = {
        "rank": args.rank,
        "alpha": CONFIG[args.dataset]["alpha"],
        "init_scheme": args.init,
        "ut_lambda": args.ut_lambda,
        "wavelet": args.wavelet,
        "transform_level": args.wavelet_level,
    }
    model = replace_qkv(model, overa_config, target_proj=args.target_proj)
    model.reset_classifier(CONFIG[args.dataset]["n_c"])
    return model


def compute_delta_w(vera_proj: th.nn.Module, init_scheme: str) -> th.Tensor:
    lambda_d = vera_proj.vera_lambda_d.detach()
    lambda_b = vera_proj.vera_lambda_b.detach()
    fixed_lambda_d = vera_proj.fixed_lambda_d.detach()
    lambda_d_full = th.cat([lambda_d, fixed_lambda_d])
    model_dim = lambda_b.shape[0]
    vera_A = None
    vera_B = None
    dt = th.float32

    if init_scheme in ["f", "h"]:
        eye = th.eye(model_dim)
        if init_scheme == "f":
            vera_A = th.fft.rfft(eye, n=model_dim, norm="ortho")
            l_d = th.diag(lambda_d_full.to(dtype=vera_A.dtype))
            temp_d = vera_A @ l_d
            vera_B = th.fft.irfft(temp_d, n=model_dim, norm="ortho")
            H = vera_B @ th.diag(lambda_b.to(dtype=vera_B.dtype))
            return H
        elif init_scheme == "h":
            vera_A = welsh_hadamard_transform(eye, inverse=False)
            l_d = th.diag(lambda_d_full.to(dtype=vera_A.dtype))
            temp_d = vera_A @ l_d
            vera_B = welsh_hadamard_transform(eye, inverse=True)
            H = vera_B @ th.diag(lambda_b.to(dtype=vera_B.dtype))
            return H
    elif init_scheme == "w":
        vera_A = vera_proj.A_vera.detach().to(dt)
        vera_B = vera_proj.B_vera.detach().to(dt)
        # H = vera_A.T @ th.diag(lambda_d_full).to(dt) @ vera_B.T @ th.diag(lambda_b).to(dt)
        H = th.diag(lambda_b).to(dt) @ vera_B.to(dt) @ th.diag(lambda_d_full).to(dt) @ vera_A.to(dt)
        return H
    else:
        raise ValueError(f"Unsupported init scheme: {init_scheme}")
    

def get_delta_w(model: th.nn.Module, init_scheme: str) -> th.Tensor:
    results = {}
    for idx, block in enumerate(model.blocks):
        qkv = block.attn.qkv
        if not hasattr(qkv, "vera_q") and not hasattr(qkv, "vera_v"):
            continue

        layer_base_key = f"block.{idx}"
        layer_dict = {}
        if hasattr(qkv, "vera_q"):
            q_delta_w = compute_delta_w(qkv.vera_q, init_scheme).detach().cpu()
            layer_dict["q"] = q_delta_w
        if hasattr(qkv, "vera_v"):
            v_delta_w = compute_delta_w(qkv.vera_v, init_scheme).detach().cpu()
            layer_dict["v"] = v_delta_w
        if hasattr(qkv, "vera_k"):
            k_delta_w = compute_delta_w(qkv.vera_k, init_scheme).detach().cpu()
            layer_dict["k"] = k_delta_w
        if layer_dict:
            results[layer_base_key] = layer_dict
    return results


def get_delta_w_kaiming(state_dict):
    results = {}
    for idx in range(12):
        layer_base_key = f"block.{idx}"
        layer_dict = {}
        for proj in ["q_proj", "v_proj"]:
            lambda_b_key = f"blocks.{idx}.attn.qkv.{proj}.vera_lambda_b"
            lambda_d_key = f"blocks.{idx}.attn.qkv.{proj}.vera_lambda_d"
            vera_b_key = f"blocks.{idx}.attn.qkv.{proj}.vera_B"
            vera_a_key = f"blocks.{idx}.attn.qkv.{proj}.vera_A"
            if (
                lambda_b_key not in state_dict
                or lambda_d_key not in state_dict
                or vera_b_key not in state_dict
                or vera_a_key not in state_dict
            ):
                raise ValueError(
                    f"Missing keys for block {idx} proj {proj} in state dict."
                )
            lambda_b = state_dict[lambda_b_key].detach()
            lambda_d = state_dict[lambda_d_key].detach()
            vera_B = state_dict[vera_b_key].detach()
            vera_A = state_dict[vera_a_key].detach()
            # H = vera_A.T @ th.diag(lambda_d).to(vera_A.dtype) @ vera_B.T @ th.diag(lambda_b).to(vera_B.dtype)
            H = th.diag(lambda_b).to(vera_B.dtype) @ vera_B @ th.diag(lambda_d).to(vera_A.dtype) @ vera_A
            layer_dict[proj[0]] = H.detach().cpu()
        if layer_dict:
            results[layer_base_key] = layer_dict
    return results


def main():
    """Main function."""
    args = _parse_args()
    if args.init != "k":
        model = _build_model(args)
        model.eval()

    sort_key =lambda p: int(Path(p).stem.split('_')[-1])

    checkpoints = sorted(glob(os.path.join(args.model_checkpoint_dir, "vit_iter_*.pt")), key=sort_key)
    if len(checkpoints) == 0:
        checkpoints = sorted(glob(os.path.join(args.model_checkpoint_dir, "vit_*.pt")), key=sort_key)
    if len(checkpoints) == 0:
        raise ValueError(f"No checkpoints found in {args.model_checkpoint_dir}.")
    
    delta_w_dict = {}
    for ckpt_path in tqdm(checkpoints):
        iteration = int(Path(ckpt_path).stem.split('_')[-1])
        state_dict = th.load(ckpt_path, map_location="cpu")
        if args.init != "k":
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                raise ValueError(
                    f"Checkpoint {ckpt_path} has missing keys: {missing} and unexpected keys: {unexpected}."
                )
            delta_w_dict[iteration] = get_delta_w(model, args.init)
        elif args.init == "k":
            delta_w_dict[iteration] = get_delta_w_kaiming(state_dict)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    file_name = os.path.join(args.output_path, f"vera_weights_{args.init}.pt")
    th.save(delta_w_dict, file_name)


if __name__ == "__main__":
    main()