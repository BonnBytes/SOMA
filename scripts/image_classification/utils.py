"""ViT training utils file."""

import os
from argparse import Namespace
from typing import Any, Dict, Tuple

import torch as th
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.overa.vera_linear import VeraMergedLinear


def replace_qkv(
    model: th.nn.Module, overa_config: Dict[str, Any]
) -> th.nn.Module:
    """Replace QKV linear with VeRA merged linear layer.

    Args:
        model (th.nn.Module): ViT model.
        overa_config (Dict[str, Any]): OVeRA configuration.

    Returns:
        th.nn.Module: VeRA replaced model.
    """
    for idx in range(0, len(model.blocks)):  # type: ignore
        base_layer = model.blocks[idx].attn.qkv  # type: ignore
        overa_layer = VeraMergedLinear(
            base_layer=base_layer, config=overa_config
        )
        model.blocks[idx].attn.qkv = overa_layer  # type: ignore
    return model


def train(
    args: Namespace,
    model: th.nn.Module,
    dl: DataLoader,
    tdl: DataLoader,
    opt: Optimizer,
    sched: LRScheduler,
    epochs: int,
    device: th.device,
) -> Tuple[th.nn.Module, str, float]:
    """Training script.

    Args:
        args (Namespace): Argument parser.
        model (th.nn.Module): ViT model.
        dl (DataLoader): Training dataloader.
        tdl (DataLoader): Test dataloader.
        opt (Optimizer): Optimiser.
        sched (LRScheduler): Learning rate scheduler.
        epochs (int): Training epochs.
        device (th.device): CUDA/CPU/MPS device.

    Returns:
        Tuple[th.nn.Module, str, float]: Tuple containing trained model and last save name.
    """
    model.train()
    acc = 0.0
    best_acc = 0.0
    last_save_name = ""
    for epoch in (pbar := tqdm(range(epochs))):
        for batch in dl:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            loss = th.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(
                f"e: {epoch}, l: {round(loss.item(), 7)}, a:{acc}"
            )
            if sched is not None:
                sched.step(epoch)
        if epoch % 5 == 0 and epoch != 0:
            if epoch >= 50:
                sched = None
            acc = test(model, tdl, device)
            if acc > best_acc:
                best_acc = acc
                if last_save_name != "":
                    os.remove(last_save_name)
                last_save_name = f"./vit_{args.dataset}_{round(acc, 5)}_seed_{args.seed}_ranks_{args.rank}.pt"
                th.save(model.cpu().state_dict(), last_save_name)
    return model, last_save_name, best_acc


@th.no_grad()
def test(model: th.nn.Module, dl: DataLoader, device: th.device) -> float:
    """Testing script.

    Args:
        model (th.nn.Module): ViT model.
        dl (DataLoader): Test dataloader.
        device (th.device): CUDA/CPU/MPS device.

    Returns:
        float: Accuracy.
    """
    model.eval()
    acc = []
    ex = []
    for batch in dl:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x).data
        out = th.argmax(out, dim=1)
        correct = (out == y).float()
        acc.extend(correct)
        ex.append(len(y))
    ac = sum(acc) / sum(ex)
    return round(ac.cpu().item(), 4)
