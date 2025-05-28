import torch as th
import numpy as np
import random
import torch.nn as nn
from tqdm import tqdm
import pickle
import ptwt
from scipy.linalg import hadamard

th.set_default_dtype(th.float32)

DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")
FULL_R = 784
RANK = 256

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


# class FreqModel(nn.Module):
#     def __init__(self):
#         super(FreqModel, self).__init__()
#         self.A = ptwt.matmul_transform.construct_boundary_a("Haar", length=FULL_R, device=DEVICE).to_dense().to(th.float32)
#         self.B = ptwt.matmul_transform.construct_boundary_s("Haar", length=FULL_R, device=DEVICE).to_dense().to(th.float32)
#         # self.wavelet_matrices()
#         self.lambda_d = nn.Parameter(th.empty(RANK), requires_grad=True)
#         self.zero_d = th.zeros(FULL_R-RANK)
#         self.lambda_b = nn.Parameter(th.empty(FULL_R), requires_grad=True)
    
#     def init_weights(self):
#         nn.init.normal_(self.lambda_d)
#         nn.init.zeros_(self.lambda_b)
    
#     def forward(self, x):
#         x = self.A.to(DEVICE) @ x
#         x = th.diag(th.cat([self.lambda_d, self.zero_d.to(DEVICE)])) @ x
#         x = self.B.to(DEVICE) @ x
#         y_out = th.diag(self.lambda_b) @ x
#         return y_out 

# class FourierModel(nn.Module):
#     def __init__(self):
#         super(FourierModel, self).__init__()
#         # self.A = self.fourier_matrix()
#         # self.B = th.conj(self.A.T) / FULL_R
#         # self.lambda_d = nn.Parameter(th.empty(RANK), requires_grad=True)
#         # self.zero_d = th.randn(FULL_R-RANK)
#         # self.lambda_b = nn.Parameter(th.empty(FULL_R), requires_grad=True)
#         print("New fourier formulation")
#         self.A = self.fourier_matrix()
#         self.A = self.A[:RANK, :]
#         self.B = th.conj(self.A.T) / FULL_R
#         self.lambda_b = nn.Parameter(th.empty(FULL_R), requires_grad=True)
#         self.lambda_d =  nn.Parameter(th.empty(RANK), requires_grad=True)
    
#     def init_weights(self):
#         nn.init.normal_(self.lambda_d)
#         nn.init.zeros_(self.lambda_b)

#     def fourier_matrix(self):
#         indices = th.arange(FULL_R)
#         real_part = th.cos(-2 * th.pi * th.outer(indices, indices) / FULL_R)
#         imag_part = th.sin(-2 * th.pi * th.outer(indices, indices) / FULL_R)
#         return th.complex(real_part, imag_part)
    
#     def forward(self, x):
#         # A, B = self.A.to(DEVICE).to(dtype=th.cfloat), self.B.to(DEVICE).to(dtype=th.cfloat)
#         # lambda_b, lambda_d = self.lambda_b.to(DEVICE).to(dtype=th.cfloat), th.cat([self.lambda_d.to(DEVICE), self.zero_d.to(DEVICE)], dim=0).to(dtype=th.cfloat)
#         # y_out = (th.diag(lambda_b) @ (B @ (th.diag(lambda_d) @ (A @ x.to(dtype=th.cfloat)))))
#         # return y_out.real.to(th.float32)
#         A, B = self.A.to(DEVICE), self.B.to(DEVICE)
#         lambda_b, lambda_d = self.lambda_b.to(DEVICE).to(dtype=th.cfloat), self.lambda_d.to(DEVICE).to(dtype=th.cfloat)
#         yout = (th.diag(lambda_b) @ (B @ (th.diag(lambda_d) @ (A @ x.to(dtype=th.cfloat)))))
#         return yout.real.to(th.float32)


class OrthoFit(nn.Module):
    def __init__(self, init_scheme="f"):
        super().__init__()
        if init_scheme not in ["f", "w", "h", "m"]:
            raise ValueError("Improper init scheme")
        init_fn = self.fourier_inits
        if init_scheme == "w":
            init_fn = self.wavelet_inits
        elif init_scheme == "h":
            init_fn = self.welsh_inits
        elif init_scheme == "m":
            init_fn = self.hippo_inits
        self.A, self.B = init_fn()
        self.lambda_d = nn.Parameter(th.empty(RANK), requires_grad=True)
        self.zeros = th.zeros(FULL_R-RANK)
        self.lambda_b = nn.Parameter(th.empty(FULL_R), requires_grad=True)
        self.init_scheme = init_scheme
    
    def fourier_inits(self):
        print("Fourier init")
        def fourier_matrix(x):
            indices = th.arange(x)
            real_part = th.cos(-2 * th.pi * th.outer(indices, indices) / x)
            imag_part = th.sin(-2 * th.pi * th.outer(indices, indices) / x)
            return th.complex(real_part, imag_part)
        def inv_fourier(x):
            Q = fourier_matrix(x)
            return th.conj(Q.T) / x
        A = fourier_matrix(FULL_R)
        B = inv_fourier(FULL_R)
        return A, B

    def wavelet_inits(self):
        print("Wavelet init")
        A = ptwt.matmul_transform.construct_boundary_a("Haar", length=FULL_R).to_dense()
        B = ptwt.matmul_transform.construct_boundary_s("Haar", length=FULL_R).to_dense()
        return A, B
    
    def welsh_inits(self):
        print("Welsh-hadamard init")
        A = th.from_numpy(hadamard(1024))
        B = th.from_numpy(hadamard(1024))
        A = A[:FULL_R, :FULL_R]
        B = B[:FULL_R, :FULL_R]
        return A, B

    def hippo_inits(self):
        print("Hippo init")
        def create_hippo(N):
            matrix = th.zeros((N, N))
            for n in range(N):
                for k in range(N):
                    if n < k:
                        matrix[n, k] = 0
                    elif n == k:
                        matrix[n, k] = n+1
                    elif n > k:
                        matrix[n, k] = ((2*n+1)**0.5)*((2*k+1)**0.5)
            return matrix
        A = create_hippo(FULL_R)
        B = th.linalg.inv(A)
        return A, B
    
    def init_weights(self):
        nn.init.normal_(self.lambda_d)
        nn.init.zeros_(self.lambda_b)

    def forward(self, x):
        if self.init_scheme != "f":
            dt = th.float32
        elif self.init_scheme == "f":
            dt = th.cfloat
        l_d = th.diag(th.cat([self.lambda_d.to(dtype=dt), self.zeros.to(dtype=dt).to(DEVICE)])).to(dtype=dt)
        l_b = th.diag(self.lambda_b.to(dtype=dt)).to(dtype=dt)
        H = l_b @ self.B.to(DEVICE).to(dtype=dt) @ l_d @ self.A.to(DEVICE).to(dtype=dt)
        y_out = H @ x.to(dtype=dt)
        if self.init_scheme == "f":
            return y_out.real.to(th.float32)
        return y_out.to(th.float32)


def main():
    epochs = 1000
    seed_mses = []
    for seed in range(0, 10):
        print(f"Seed: {seed}")
        mses = []
        y = th.randn((FULL_R, 1))
        x = th.ones((FULL_R, 1))
        model = OrthoFit(init_scheme="m")
        # model = FitModel()
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
    with open("hippo_mse_zero.pkl", "wb") as fp:
        pickle.dump(seed_mses, fp)


if __name__ == "__main__":
    main()