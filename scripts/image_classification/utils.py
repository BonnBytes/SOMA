
import torch as th
from tqdm.auto import tqdm
import os

from src.overa.vera_linear import VeraMergedLinear

def replace_qkv(model, overa_config):
    for idx in range(0, len(model.blocks)):
        base_layer = model.blocks[idx].attn.qkv
        overa_layer = VeraMergedLinear(
            base_layer=base_layer,
            config=overa_config
        )
        model.blocks[idx].attn.qkv = overa_layer
    return model

def train(args, model, dl, tdl, opt, sched, epochs, device):
    model.train()
    acc = 0.
    best_acc = 0.0
    last_save_name = ""
    for epoch in (pbar:=tqdm(range(epochs))):    
        for batch in dl:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            loss = th.nn.functional.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_description(f"e: {epoch}, l: {round(loss.item(), 7)}, a:{acc}")
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
def test(model, dl, device):
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