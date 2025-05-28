import torch as th
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")

class AutoEncoder(th.nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = th.nn.Sequential(
        #     th.nn.Linear(28*28, 128),
        #     th.nn.ReLU(),
        #     th.nn.Linear(128, 64),
        #     th.nn.ReLU(),
        #     th.nn.Linear(64, 32),
        # )
        # self.decoder = th.nn.Sequential(
        #     th.nn.Linear(32, 64),
        #     th.nn.ReLU(),
        #     th.nn.Linear(64, 128),
        #     th.nn.ReLU(),
        #     th.nn.Linear(128, 28*28),
        #     th.nn.Sigmoid()
        # )
        self.encoder = th.nn.Sequential(
            th.nn.Linear(28 * 28, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 32),
            th.nn.ReLU(),
            th.nn.Linear(32, 16),
        )
        self.decoder = th.nn.Sequential(
            th.nn.Linear(16, 32),
            th.nn.ReLU(),
            th.nn.Linear(32, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 28 * 28),
            th.nn.Sigmoid()
        )

    
    def forward(self, x):
        lt_embed = self.encoder(x)
        y_recons = self.decoder(lt_embed)
        return y_recons, lt_embed


def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True


    tensor_transform = transforms.ToTensor()
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=tensor_transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=tensor_transform)
    trainloader = th.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
    testloader = th.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=True)
    
    print(f"Using device: {DEVICE}")
    model = AutoEncoder()
    model.to(DEVICE)
    loss_fn = th.nn.MSELoss()
    opt = th.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    epochs = 20
    iter_loss = []
    print("Starting training")
    model.train()
    i = 0
    for epoch in (pbar:=tqdm(range(epochs))):
        for imgs, _ in trainloader:
            imgs = imgs.view(-1, 28*28).to(DEVICE)

            recons, _ = model(imgs)
            loss_val = loss_fn(recons, imgs)

            opt.zero_grad()
            loss_val.backward()
            opt.step()

            iter_loss.append(loss_val.item())
            pbar.set_description(f"Epoch: {epoch}, Loss: {round(loss_val.item(), 5)}")
        # print(f"Epoch: {epoch}")
        
    plt.plot(iter_loss)
    plt.savefig("loss.png", dpi=600, bbox_inches="tight")

    print("Starting inference")
    model.eval()
    imgs = None
    for i, _ in testloader:
        imgs = i
        break
    imgs = imgs.view(-1, 28*28).to(DEVICE)
    recons, _ = model(imgs)

    print(imgs.shape, recons.shape)
    fig, axs = plt.subplots(nrows=2, ncols=12, figsize=(10, 3))
    for i in range(12):
        axs[0, i].imshow(imgs[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(recons[i].cpu().detach().numpy().reshape(28, 28), cmap="gray")
        axs[1, i].axis("off")
    plt.savefig("MNIST_recons.png", dpi=600, bbox_inches="tight")
    plt.close()

    print("Saviing model")
    model = model.to("cpu")
    th.save(model.state_dict(), "./mnist_pretrained.pt")


if __name__ == "__main__":
    main()