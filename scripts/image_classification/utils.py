"""ViT training utils file."""

import os
from argparse import Namespace
from typing import Any, Dict, Tuple

import torch as th
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from soma.soma_linear import (
    SomaLinear,
    SomaMergedLinear,
    SomaMergedLinearQV,
)


def replace_qkv(
    model: th.nn.Module,
    overa_config: Dict[str, Any],
    target_proj: str = "qv",
) -> th.nn.Module:
    """Replace QKV linear with VeRA merged linear layer.

    Args:
        model (th.nn.Module): ViT model.
        overa_config (Dict[str, Any]): OVeRA configuration.
        target_proj (str): Projection target, one of qv or qkv.

    Raises:
        ValueError: Error if target projection value is invalid.

    Returns:
        th.nn.Module: VeRA replaced model.
    """
    if "qv" in target_proj or "qkv" in target_proj:
        print(f"Replacing {target_proj.upper()}")
        for idx in range(0, len(model.blocks)):  # type: ignore
            base_layer = model.blocks[idx].attn.qkv  # type: ignore
            overa_layer: th.nn.Module
            if "qv" in target_proj:
                overa_layer = SomaMergedLinearQV(
                    base_layer=base_layer, config=overa_config
                )
            elif "qkv" in target_proj:
                overa_layer = SomaMergedLinear(
                    base_layer=base_layer, config=overa_config
                )
            else:
                raise ValueError("target_proj should be one of ['qv', 'qkv']")
            model.blocks[idx].attn.qkv = overa_layer  # type: ignore
    if "o" in target_proj:
        print("Replacing O")
        for idx in range(0, len(model.blocks)):
            base_layer = model.blocks[idx].attn.proj  # type: ignore
            overa_layer = SomaLinear(
                base_layer=base_layer, config=overa_config
            )
            model.blocks[idx].attn.proj = overa_layer  # type: ignore
    if "f" in target_proj:
        print("Replacing FFN")
        for idx in range(0, len(model.blocks)):
            base_layer1 = model.blocks[idx].mlp.fc1  # type: ignore
            overa_layer1 = SomaLinear(
                base_layer=base_layer1, config=overa_config
            )
            model.blocks[idx].mlp.fc1 = overa_layer1  # type: ignore
            base_layer2 = model.blocks[idx].mlp.fc2  # type: ignore
            overa_layer2 = SomaLinear(
                base_layer=base_layer2, config=overa_config
            )
            model.blocks[idx].mlp.fc2 = overa_layer2  # type: ignore
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
    val_acc = [0.0]
    train_acc = [0.0]
    best_acc = 0.0
    last_save_name = ""
    iter = 0
    for epoch in (pbar := tqdm(range(epochs))):
        for batch in dl:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            loss = th.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            iter += 1
            opt.step()
            pbar.set_description(
                f"e: {epoch}, l: {round(loss.item(), 7)}, a:{acc}, ta:{train_acc[-1]}"
            )
            if sched is not None:
                sched.step(epoch)
        train_acc.append(test(model, dl, device))
        acc = test(model, tdl, device)
        val_acc.append(acc)
        if acc > best_acc:
            best_acc = acc
            if os.path.exists(last_save_name):
                os.remove(last_save_name)
            last_save_name = (
                f"{args.save_folder}/vit_{args.dataset}_{args.init}_{args.target_proj}_{round(acc, 5)}"
                f"_seed_{args.seed}_ranks_{args.rank}.pt"
            )
            th.save(model.state_dict(), last_save_name)
    return model, last_save_name, best_acc, train_acc, val_acc


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
    for batch in tqdm(dl):
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x).data
        out = th.argmax(out, dim=1)
        correct = (out == y).float()
        acc.extend(correct)
        ex.append(len(y))
    ac = sum(acc) / sum(ex)
    return round(ac.cpu().item(), 4)
