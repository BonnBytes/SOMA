"""Vision Transformer Image Classification."""

import os
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

import torch as th
from data_config import CONFIG
from dataloader import get_dl
from timm.models import create_model
from datetime import datetime
from utils import replace_qkv, test, train

from src.overa.utils import set_seed


def _parse_args() -> Namespace:
    """Argument parser.

    Returns:
        Namespace: Arguments namespace.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rank", default=256, type=int, help="OVeRA Rank")
    parser.add_argument("--head_lr", default=4e-3, type=float, help="Head Learning rate")
    parser.add_argument("--base_lr", default=2e-2, type=float, help="Base Learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
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
        choices=["f", "w", "h"],
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
        choices=["qv", "qkv"],
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
    save_folder = f"./vit-{args.model}-ckpts/{args.dataset}_{args.init}/{current_datetime}"
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
            "vit_large_patch16_224_in21k", pretrained=True#, drop_path_rate=0.1
        )
    elif "base" == args.model:
        vit = create_model(
            "vit_base_patch16_224.augreg_in21k",
            pretrained=True
        )

    #"vit_base_patch16_224_in21k",
    #checkpoint_path="./ViT-B_16.npz",
            
    overa_config = {
        "rank": args.rank,
        "alpha": alpha,
        "init_scheme": args.init,
        "ut_lambda": args.ut_lambda,
        "wavelet": args.wavelet,
        "transform_level": args.wavelet_level,
    }
    vit = replace_qkv(vit, overa_config, target_proj=args.target_proj)
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
            or name.startswith("norm.")
        ):
            param.requires_grad = True
            if "head" not in name:
                trainable.append(param)
                total_param += param.numel()
    print(f"Total PEFT parameters: {total_param}")
    th.nn.init.zeros_(vit.head.weight)
    vit = vit.to(device)
    opt = th.optim.AdamW(
        [{"params": vit.head.parameters(), "lr": args.head_lr}, {"params": trainable}],
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = th.optim.lr_scheduler.LinearLR(opt, total_iters=10*len(train_dl))
    vit, last_save_name, best_acc, train_acc, val_acc = train(
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
        os.remove(last_save_name)
        last_save_name = (
            f"{args.save_folder}/vit_{args.dataset}_{round(acc, 5)}"
            f"_seed_{args.seed}_ranks_{args.rank}.pt"
        )
        th.save(vit.cpu().state_dict(), last_save_name)

    metrics_json_path = os.path.splitext(last_save_name)[0] + "_metrics.json"
    run_summary = {
        "args": vars(args),
        "train_acc": train_acc,
        "val_acc": val_acc,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(f"Saved run metadata to {metrics_json_path}")


if __name__ == "__main__":
    main()
