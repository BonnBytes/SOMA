"""Vision Transformer Image Classification."""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

import torch as th
from data_config import CONFIG
from dataloader import get_dl
from timm.models import create_model
from utils import replace_qkv, test, train

from src.overa.utils import set_seed


def _parse_args() -> Namespace:
    """Argument parser.

    Returns:
        Namespace: Arguments namespace.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rank", default=256, type=int, help="OVeRA Rank")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--dataset",
        default="cifar",
        type=str,
        choices=["choices", "flowers", "resisc", "food"],
        help="Dataset to train",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "large"],
        help="Choice of model",
    )
    parser.add_argument(
        "--data_seed", default=6, type=int, help="Dataset seed"
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=["f", "w", "h", "m"],
        default="f",
        help="Choice of initialisation method",
    )
    parser.add_argument(
        "--ut_lambda",
        type=str,
        choices=["z", "o", "r"],
        default="z",
        help="Choice for untrained lambda intialisation",
    )
    parser.add_argument(
        "--wavelet", type=str, default="Haar", help="Choice of wavelet"
    )
    parser.add_argument(
        "--wavelet_level",
        type=int,
        default=7,
        help="Wavelet transformation level",
    )
    parser.add_argument("--evaluate", default=None, type=str)
    return parser.parse_args()


def main():
    """Vit training script."""
    set_seed()
    args = _parse_args()
    data_config = CONFIG[args.dataset]
    alpha = data_config["alpha"]
    _ = data_config["logger"]
    num_classes = data_config["n_c"]
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

    train_dl, test_dl = get_dl(args.dataset, args.data_seed)
    if "large" == args.model:
        vit = create_model(
            "vit_large_patch16_224_in21k", pretrained=True, drop_path_rate=0.1
        )
    elif "base" == args.model:
        vit = create_model(
            "vit_base_patch16_224_in21k",
            checkpoint_path="./ViT-B_16.npz",
            drop_path_rate=0.1,
        )

    overa_config = {
        "rank": args.rank,
        "alpha": alpha,
        "init_scheme": args.init,
        "ut_lambda": args.ut_lambda,
        "wavelet": args.wavelet,
        "transform_level": args.wavelet_level,
    }
    vit = replace_qkv(vit, overa_config)
    vit.reset_classifier(num_classes)
    trainable = []
    total_param = 0
    for name, param in vit.named_parameters():
        if (
            "vera_lambda_d" in name
            or "vera_lambda_b" in name
            or "head" in name
        ):
            param.requires_grad = True
            if "head" not in name:
                trainable.append(param)
                total_param += param.numel()
        else:
            param.requires_grad = False
    print(f"Total parameters: {total_param}")
    th.nn.init.zeros_(vit.head.weight)
    vit = vit.to(device)
    opt = th.optim.AdamW(
        [{"params": vit.head.parameters(), "lr": 1e-3}, {"params": trainable}],
        lr=4e-2,
        weight_decay=0.01,
    )
    scheduler = th.optim.lr_scheduler.LinearLR(opt, total_iters=100)
    vit, last_save_name, best_acc = train(
        args=args,
        model=vit,
        dl=train_dl,
        tdl=test_dl,
        opt=opt,
        sched=scheduler,
        epochs=100,
        device=device,
    )
    acc = test(vit, test_dl)
    if acc > best_acc:
        os.remove(last_save_name)
        last_save_name = f"./vit_{args.dataset}_{round(acc, 5)}_seed_{args.seed}_ranks_{args.rank}.pt"
        th.save(vit.cpu().state_dict(), last_save_name)


if __name__ == "__main__":
    main()
