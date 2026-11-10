"""Vision Transformer Image Classification."""

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime

import torch as th
from data_config import CONFIG
from dataloader import get_dl
from timm.models import create_model
from utils import replace_qkv, test, train

from src.soma.utils import set_seed


def _parse_args() -> Namespace:
    """Argument parser.

    Returns:
        Namespace: Arguments namespace.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rank", default=256, type=int, help="SOMA Rank")
    parser.add_argument(
        "--head_lr", default=4e-3, type=float, help="Head Learning rate"
    )
    parser.add_argument(
        "--base_lr", default=2e-2, type=float, help="Base Learning rate"
    )
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="Weight decay"
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size"
    )
    parser.add_argument(
        "--dataset",
        default="cifar",
        type=str,
        choices=["cifar", "flowers", "resisc", "food"],
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
        choices=["f", "w", "h", "m", "vo"],
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
        default=8,
        help="Wavelet transformation level",
    )
    parser.add_argument(
        "--target_proj",
        type=str,
        choices=["qv", "qkv", "qkvo", "qkvof", "qvof"],
        default="qv",
        help="Attention projections to adapt with VeRA.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Network seed")
    parser.add_argument("--evaluate", default=None, type=str)
    return parser.parse_args()


def main():
    """Vit training script."""
    args = _parse_args()
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    save_folder = f"./vit-{args.model}-ckpts/{args.dataset}_{args.init}_{args.target_proj}/{current_datetime}"
    os.makedirs(save_folder, exist_ok=True)
    args.save_folder = save_folder
    set_seed(args.seed)
    data_config = CONFIG[args.dataset]
    alpha = data_config["alpha"]
    _ = data_config["logger"]
    num_classes = data_config["n_c"]
    device = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

    train_dl, test_dl = get_dl(args.dataset, args.data_seed, args.batch_size)
    if "large" == args.model:
        vit = create_model(
            "vit_large_patch16_224_in21k",
            pretrained=True,  # , drop_path_rate=0.1
        )
    elif "base" == args.model:
        vit = create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True
        )

    soma_config = {
        "rank": args.rank,
        "alpha": alpha,
        "init_scheme": args.init,
        "ut_lambda": args.ut_lambda,
        "wavelet": args.wavelet,
        "transform_level": args.wavelet_level,
    }
    vit = replace_qkv(vit, soma_config, target_proj=args.target_proj)
    vit.reset_classifier(num_classes)

    trainable = []
    total_param = 0
    for param in vit.parameters():
        param.requires_grad = False

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
                # print(name)
    print(f"Total PEFT parameters: {total_param}")

    th.nn.init.zeros_(vit.head.weight)
    vit = vit.to(device)
    opt = th.optim.AdamW(
        [
            {"params": vit.head.parameters(), "lr": args.head_lr},
            {"params": trainable},
        ],
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = th.optim.lr_scheduler.LinearLR(
        opt, total_iters=10 * len(train_dl)
    )
    vit, last_save_name, best_acc, _, _ = train(
        args=args,
        model=vit,
        dl=train_dl,
        tdl=test_dl,
        opt=opt,
        sched=scheduler,
        epochs=10,
        device=device,
    )
    acc = test(vit, test_dl, device)
    print(f"Final accuracy: {acc}")
    if acc > best_acc:
        best_acc = acc
        if os.path.exists(last_save_name):
            os.remove(last_save_name)
        last_save_name = (
            f"{args.save_folder}/vit_{args.dataset}_{args.init}_{args.target_proj}_{round(acc, 5)}"
            f"_seed_{args.seed}_ranks_{args.rank}.pt"
        )
        th.save(vit.state_dict(), last_save_name)
    print(f"Best Accuracy: {best_acc}")


if __name__ == "__main__":
    main()
