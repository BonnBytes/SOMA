"""Vision Transformer Image Classification."""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

import torch as th

from timm.models import create_model
from tqdm import tqdm
import timm

from src.overa.utils import set_seed
from src.overa.vera_linear import VeraMergedLinear
from dataloader import get_dl
from data_config import CONFIG

def _parse_args() -> Namespace:
    """Argument parser.

    Returns:
        Namespace: Arguments namespace.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--rank",
        default=256,
        type=int,
        help="OVeRA Rank"
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--dataset",
        default="cifar",
        type=str,
        choices=["choices", "flowers", "resisc", "food"],
        help="Dataset to train"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_large_patch16_224_in21k",
        choices=["vit_base_patch16_224_in21k", "vit_large_patch16_224_in21k"],
        help="Choice of model"
    )
    parser.add_argument(
        "--data_seed",
        default=6,
        type=int,
        help="Dataset seed"
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["f", "w", "h", "m"],
        default="f",
        help="Choice of initialisation method"
    )
    parser.add_argument(
        "--ut_lambda",
        type=str,
        choices=["z", "o", "r"],
        default="z",
        help="Choice for untrained lambda intialisation"
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="Haar",
        help="Choice of wavelet"
    )
    parser.add_argument(
        "--wavelet_level",
        type=int,
        default=7,
        help="Wavelet transformation level"
    )
    parser.add_argument(
        "--evaluate",
        default=None,
        type=str
    )
    return parser.parse_args()


def main():
    """Main function."""
    set_seed()
    args = _parse_args()
    data_config = CONFIG[args.dataset]
    alpha = data_config["alpha"]
    _ = data_config["logger"]
    num_classes = data_config["n_c"]
    
    train_dl, test_dl = get_dl(args.dataset, args.data_seed)
    if "large" in args.model:
        vit = create_model(args.model, pretrained=True, drop_path_rate=0.1)
    elif "base" in args.model:
        vit = create_model(args.model, checkpoint_path="./ViT-B_16.npz", drop_path_rate=0.1)

    overa_config = {
        "rank": args.rank,
        "alpha": alpha,
        "init_scheme": args.init,
        "ut_lambda": args.ut_lambda,
        "wavelet": args.wavelet,
        "transform_level": args.wavelet_level,
    }
    # Replace ViT QKV with VeRAMergedLinear
    vit.reset_classifier(num_classes)
    trainable = []
    total_param = 0
    for name, param in vit.named_parameters():
        if "vera_lambda_d" in name or "vera_lambda_b" in name:
            param.requires_grad = True
            if "head" not in name:
                trainable.append(param)
                total_param += param.numel()
        else:
            param.requires_grad = False
    print(f"Total parameters: {total_param}")
    th.nn.init.zeros_(vit.head.weight)
    opt = th.optim.AdamW(
        [
            {"params": vit.head.parameters(), "lr": 1e-3},
            {"params": trainable}
        ],
        lr=4e-2, weight_decay=0.01
    )
    scheduler = th.optim.lr_scheduler.LinearLR(opt, total_iters=100)
    # Train
    # Test

if __name__ == "__main__":
    main()
