import numpy as np
import pywt
import torch
import torch.nn as nn
import warnings
from typing import List, Optional
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from buffer_dict import BufferDict

def wavelet_decompose_matrix(matrix, levels, wavelet='haar'):
    """Decompose a matrix using 2D wavelet transform"""
    # wavelet decomposition
    coeffs = pywt.wavedec2(matrix, wavelet, level=levels)
    # extract Low-frequency (approximation) components
    ll = coeffs[0]

    return ll, coeffs

def wavelet_reconstruct_matrix(coeffs, origional_shape, wavelet = 'haar'):
    """Reconstruct a matrix from 2D wavelet coefficients."""
    reconstructed = pywt.waverec2(coeffs, wavelet)
    rows , cols = origional_shape
    # Remove any padding and return
    return reconstructed[:rows, :cols]

def random_permutation_matrix(size):
    """Generate a random permutation matrix of given size."""
    perm = torch.randperm(size)
    I = torch.eye(size)
    P = I[perm]
    return P


class VeraLayer(nn.Module):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("vera_lambda_b", "vera_lambda_d")
    other_param_names = ("vera_A", "vera_B")

    def __init__(self, base_layer: nn.Module, wavelet='haar', **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.wavelet = wavelet
        # Dictionary storing rank values for each adapter.
        self.r = {}
        self.vera_dropout = nn.ModuleDict({})

     
        self.vera_lambda_b = None
        self.vera_lambda_d = None
        #self.vera_lambda_d = nn.ParameterDict({})


        # Stores a reference to the vera_A/B BufferDict.
        self.vera_A: Optional[BufferDict] = None
        self.vera_B: Optional[BufferDict] = None
        
        # Mark the weight as unmerged
        self._disable_adapters = False
        # Keeps track of the adapters that have been merged with the base weights.
        self.merged_adapters = []
        # Set up layer dimensions based on the base layer
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.shape[::-1]  # Reverse if Conv1D weight shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def get_base_layer(self):
        return self.base_layer

    # Returns True if any adapters have been merged, based on the merged_adapters list.
    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)
    
    def update_layer(
        self,
        r,
        vera_dropout,
        init_weights,
        d_initial: float = 0.1,
    ):
        
        # print("self.out_features")
        # print(self.out_features)
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r = r
        if vera_dropout > 0.0:
            vera_dropout_layer = nn.Dropout(p=vera_dropout)
        else:
            vera_dropout_layer = nn.Identity()

        self.vera_dropout.update(nn.ModuleDict({"vera": vera_dropout_layer}))

        # determining the number of levels for wavelet decomposition and handle padding(bc matrix should be a power of 2)
        matrix = self.base_layer.weight.detach().cpu().numpy()
        rows, cols = matrix.shape
        max_levels = int(min(np.log2(rows), np.log2(cols)))
        level = max(1, max_levels)

        # Pad matrix to nearest power of 2 if necessary
        padded_rows = 2 ** int(np.ceil(np.log2(rows)))
        padded_cols = 2 ** int(np.ceil(np.log2(cols)))
        padded_matrix = np.zeros((padded_rows, padded_cols))
        padded_matrix[:rows, :cols] = matrix

        # Initialize learnable diagonal matrices D and G as identity
        self.lambda_b = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.lambda_d = nn.Parameter(torch.ones(r), requires_grad=True)

        # Generate fixed random permutation matrix Pi
        self.pi = random_permutation_matrix(r)

        # weights = self.base_layer.weight.detach().cpu().numpy()
        ll, coeffs, padded_rows, padded_cols = wavelet_decompose_matrix(padded_matrix, levels=level, wavelet=self.wavelet)
        
        print(f"ll shape: {torch.tensor(ll, dtype= torch.float32).shape}")
        
        self.vera_A = nn.Parameter(self.pi @ torch.tensor(ll, dtype= torch.float32), requires_grad = False)
        self.coeffs_shape = (padded_rows, padded_cols)
        self.coeffs = coeffs  # Store full coefficients for potential reconstruction

        # Compute W-inverse coefficients
        W_reconstruct = wavelet_reconstruct_matrix(self.coeffs, matrix.shape, wavelet=self.wavelet)
        W_inverse = np.linalg.pinv(W_reconstruct)
        ll_inv, coeffs_inv,_,_ = wavelet_decompose_matrix(W_inverse, wavelet=self.wavelet)
        self.vera_B = nn.Parameter(torch.tensor(ll_inv, dtype= torch.float32), requires_grad = False)
       
        if len(self.vera_A) < 1:
            raise ValueError("The `vera_A` and `vera_B` buffers are empty. This should not happen.")

        # if init_weights:
        #     self.reset_vera_parameters(d_initial)

        self._move_adapter_to_device_of_base_layer()
       

    def reset_vera_parameters(self, d_initial: float = 0.1):
        with torch.no_grad():
            #nn.init.zeros_(self.vera_lambda_d).fill_(d_initial)
            nn.init.zeros_(self.vera_lambda_b)

    def _move_adapter_to_device_of_base_layer(self):
        # Moves the adapter's parameters (vera_lambda_b, vera_lambda_d) to the same device as the base layer.
        device = self.base_layer.weight.device
        self.vera_lambda_b = self.vera_lambda_b.to(device)
        self.vera_lambda_d = self.vera_lambda_d.to(device)
        self.vera_B = self.vera_B.to(device)
        self.vera_A = self.vera_A.to(device)

   


class Linear(nn.Linear, VeraLayer):
    def __init__(
        self,
        base_layer,
        weight,    
        r: int = 0,
        vera_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = False,
        d_initial: float = 0.1,
        **kwargs,
    ):
        
        VeraLayer.__init__(self, base_layer, **kwargs)
        # adapter name: The name of the adapter being used (could be useful for multi-task learning).
       
        self.fan_in_fan_out = fan_in_fan_out
        
        # Calls update_layer to initialize or update the adapter-related parameters (vera_A, vera_B, etc.) and set up the layer for the adapter specified by adapter_name.
        self.update_layer(r, vera_dropout, init_weights, d_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.scale = 0.7

    # x: torch.Tensor: This is the input tensor that will be passed through the model. It typically represents a batch of input data.
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # First, it performs a standard forward pass through the base_layer (which is the underlying nn.Linear or similar layer).
        # print(f"xlinear{x.shape}")
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        device = self.base_layer.weight.device

        #  Each active_adapter in self.active_adapters will be processed in the upcoming steps to modify the result.
        A = self.vera_A.to(device)
        B = self.vera_B.to(device)
        

        lambda_d = self.vera_lambda_d.to(device)
        lambda_b = self.vera_lambda_b.to(device)
      
        dropout = self.vera_dropout["vera"]
        # x = x.to(lambda_d.dtype)
        fin_res = torch.diag(lambda_b) @ B @ torch.diag(lambda_d) @ A  
        adapted_result = dropout(x) @ fin_res
        result = result +  adapted_result
        # result = result.to(previous_dtype)
        print(f"x:{x.shape}")
        print(f"B:{B.shape}")
        print(f"B:{A.shape}")
        print(f"d:{lambda_d.shape}")
        print(f"b:{lambda_b.shape}")
        # print(f"res:{res.shape}")
        print(f"fin_res:{fin_res.shape}")

     
        return result
# operation effectively merges the contributions from the adapter into the final output, modifying it based on the learned parameters.

class MergedLinear(nn.Linear):
    def __init__(
        self,
        base_layer,
        weight,     
        r: int = 0,
        vera_dropout: float = 0.0,
        scale: float = 0,
        num_heads: int = 12,
        embed_dim: int = 768,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        **kwargs,
    ):
     
        # adapter name: The name of the adapter being used (could be useful for multi-task learning).
        super().__init__(base_layer.in_features, base_layer.out_features)
        self.base_layer = base_layer
        # nn.Linear.__init__(self, base_layer.in_features, base_layer.out_features)
        # self.base_layer = base_layer
        self.fan_in_fan_out = fan_in_fan_out
        self.dim = self.base_layer.out_features // 3
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        # Calls update_layer to initialize or update the adapter-related parameters (vera_A, vera_B, etc.) and set up the layer for the adapter specified by adapter_name.
        # self.update_layer(vera_A, vera_B, r, vera_dropout, init_weights, d_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        # Split thpe weight tensor into Q, K, V components
        weight_q, weight_k, weight_v = torch.split(weight, self.dim, dim=0)

        # Initialize projection layers
        q = nn.Linear(self.base_layer.in_features, self.dim)
        v = nn.Linear(self.base_layer.in_features, self.dim)

        # Assign weights and biases
        with torch.no_grad():
            q.weight.copy_(weight_q)
            v.weight.copy_(weight_v)

        # Separate linear layers for q and v
        self.q_proj = Linear(
                base_layer= q,
                weight= weight,
                r=r,
                vera_dropout=vera_dropout
            )
        self.v_proj = Linear(
                base_layer= v,
                weight=weight,
                r=r,
                vera_dropout=vera_dropout
            )
        self.scale = scale

    # x: torch.Tensor: This is the input tensor that will be passed through the model. It typically represents a batch of input data.
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # First, it performs a standard forward pass through the base_layer (which is the underlying nn.Linear or similar layer).
        result = super().forward(x)
        Q = self.q_proj(x)
        #K = self.k_proj(x)
        V = self.v_proj(x)
        K = torch.zeros_like(V)
        out = torch.cat([Q, K, V], dim=-1)
        # print(f"out{out.shape}")
        result = result + self.scale * out
        return result
# operation effectively merges the contributions from the adapter into the final output, modifying it based on the learned parameters.
