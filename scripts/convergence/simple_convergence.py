import torch as th
import numpy as np
import random
import torch.nn as nn
from tqdm import tqdm
import pickle
import ptwt

th.set_default_dtype(th.float32)

DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
FULL_R = 768
RANK = 32

def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = True


class FitModel(nn.Module):
    def __init__(self):
        super(FitModel, self).__init__()
        self.A = th.empty((RANK, FULL_R))
        self.B = th.empty((FULL_R, RANK))
        self.normal_init()
        self.lambda_d = nn.Parameter(th.empty(RANK), requires_grad=True)
        self.lambda_b = nn.Parameter(th.empty(FULL_R), requires_grad=True)

    def init_weights(self):
        nn.init.normal_(self.lambda_d)
        nn.init.zeros_(self.lambda_b)

    def normal_init(self):
        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)
    
    def forward(self, x):
        x = self.A.to(DEVICE) @ x
        x = th.diag(self.lambda_d) @ x
        x = self.B.to(DEVICE) @ x
        y_out = th.diag(self.lambda_b) @ x
        return y_out 


class FreqModel(nn.Module):
    def __init__(self):
        super(FreqModel, self).__init__()
        self.A = ptwt.matmul_transform.construct_boundary_a("Haar", length=FULL_R, device=DEVICE).to_dense().to(th.float32)
        self.B = ptwt.matmul_transform.construct_boundary_s("Haar", length=FULL_R, device=DEVICE).to_dense().to(th.float32)
        # self.wavelet_matrices()
        self.lambda_d = nn.Parameter(th.empty(RANK), requires_grad=True)
        self.zero_d = th.zeros(FULL_R-RANK)
        self.lambda_b = nn.Parameter(th.empty(FULL_R), requires_grad=True)
    
    def init_weights(self):
        nn.init.normal_(self.lambda_d)
        nn.init.zeros_(self.lambda_b)
    
    def forward(self, x):
        x = self.A.to(DEVICE) @ x
        x = th.diag(th.cat([self.lambda_d, self.zero_d.to(DEVICE)])) @ x
        x = self.B.to(DEVICE) @ x
        y_out = th.diag(self.lambda_b) @ x
        return y_out 

class FourierModel(nn.Module):
    def __init__(self):
        super(FourierModel, self).__init__()
        self.A = self.fourier_matrix()
        self.B = th.conj(self.A.T) / FULL_R
        self.lambda_d = nn.Parameter(th.empty(RANK), requires_grad=True)
        self.zero_d = th.randn(FULL_R-RANK)
        self.lambda_b = nn.Parameter(th.empty(FULL_R), requires_grad=True)
    
    def init_weights(self):
        nn.init.normal_(self.lambda_d)
        nn.init.zeros_(self.lambda_b)

    def fourier_matrix(self):
        indices = th.arange(FULL_R)
        real_part = th.cos(-2 * th.pi * th.outer(indices, indices) / FULL_R)
        imag_part = th.sin(-2 * th.pi * th.outer(indices, indices) / FULL_R)
        return th.complex(real_part, imag_part)
    
    def forward(self, x):
        A, B = self.A.to(DEVICE).to(dtype=th.cfloat), self.B.to(DEVICE).to(dtype=th.cfloat)
        lambda_b, lambda_d = self.lambda_b.to(DEVICE).to(dtype=th.cfloat), th.cat([self.lambda_d.to(DEVICE), self.zero_d.to(DEVICE)], dim=0).to(dtype=th.cfloat)
        y_out = (th.diag(lambda_b) @ (B @ (th.diag(lambda_d) @ (A @ x.to(dtype=th.cfloat)))))
        return y_out.real.to(th.float32)


def main():
    epochs = 2500
    seed_mses = []
    for seed in range(0, 50):
        print(f"Seed: {seed}")
        mses = []
        y = th.randn((FULL_R, 1))
        x = th.ones((FULL_R, 1))
        model = FourierModel()
        model.init_weights()
        model = model.to(DEVICE)
        y, x = y.to(DEVICE), x.to(DEVICE)
        loss = nn.MSELoss()
        opt = th.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in (pbar:=tqdm(range(0, epochs))):
            opt.zero_grad()
            out = model(x)
            loss_val = loss(out, y)
            loss_val.backward()
            opt.step()
            mses.append(loss_val.item())
            pbar.set_description(f"Epoch: {epoch} -- Loss: {loss_val.item()}")
        seed_mses.append(mses)
    with open("fourier_mse_randn.pkl", "wb") as fp:
        pickle.dump(seed_mses, fp)


if __name__ == "__main__":
    main()