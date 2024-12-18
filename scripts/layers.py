import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from buffer_dict import BufferDict


class VeraLayer(nn.Module):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("vera_lambda_b", "vera_lambda_d")
    other_param_names = ("vera_A", "vera_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        # Dictionary storing rank values for each adapter.
        self.r = {}
        self.vera_dropout = nn.ModuleDict({})

        # For storing vector scale
        # nn.ParameterDict({}) for When you want to maintain a collection of parameters (e.g., weights or biases) that you need to optimize during training.
        # self.vera_lambda_b = nn.ParameterDict({})
        # self.vera_lambda_d = nn.ParameterDict({})
        # self.block_name = str(block_name)
         # Store unique lambda_b and lambda_d for each block
         # Initialize lambda_b and lambda_d as nn.ParameterDict
        self.vera_lambda_b = nn.ParameterDict({})
        self.vera_lambda_d = nn.ParameterDict({})


        # Stores a reference to the vera_A/B BufferDict.
        self.vera_A: Optional[BufferDict] = None
        self.vera_B: Optional[BufferDict] = None
        # print(self.block_name)
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
    # This method updates the parameters associated with a specified adapter for the Vera layer.
    def update_layer(
        self,
        vera_A,
        vera_B,
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

        # print(f"in features in layers.py:{self.in_features}, out:{self.out_features}")
        self.vera_lambda_b = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)
        # print(self.vera_lambda_b.shape)
        self.vera_A = vera_A
        self.vera_B = vera_B

        if len(self.vera_A) < 1:
            raise ValueError("The `vera_A` and `vera_B` buffers are empty. This should not happen.")

        # Checks if the input dimension of vera_A is less than the expected input features (self.in_features). If so, raises a ValueError.
        if self.vera_A.shape[1] < self.in_features:
            raise ValueError(f"vera_A has insufficient input size {self.vera_A.shape[1]}, needs {self.in_features}")
        if self.vera_B.shape[0] < self.out_features:
            raise ValueError(f"vera_B has insufficient output size {self.vera_B.shape[0]}, needs {self.out_features}")

        if init_weights:
            self.reset_vera_parameters(d_initial)

        self._move_adapter_to_device_of_base_layer()
        print(f"self.vera_lambda_b:{self.vera_lambda_b.shape}")
        print(f"self.vera_lambda_d:{self.vera_lambda_d.shape}")
       

    def reset_vera_parameters(self, d_initial: float = 0.1):
        with torch.no_grad():
            nn.init.zeros_(self.vera_lambda_d).fill_(d_initial)
            nn.init.zeros_(self.vera_lambda_b)

    def _move_adapter_to_device_of_base_layer(self):
        # Moves the adapter's parameters (vera_lambda_b, vera_lambda_d) to the same device as the base layer.
        device = self.base_layer.weight.device
        self.vera_lambda_b = self.vera_lambda_b.to(device)
        self.vera_lambda_d = self.vera_lambda_d.to(device)

   


class Linear(nn.Linear, VeraLayer):
    def __init__(
        self,
        base_layer,
        vera_A,
        vera_B,     
        r: int = 0,
        vera_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        **kwargs,
    ):
        
        VeraLayer.__init__(self, base_layer, **kwargs)
        # adapter name: The name of the adapter being used (could be useful for multi-task learning).
        # super().__init__(base_layer.in_features, base_layer.out_features)
        # nn.Linear.__init__(self, base_layer.in_features, base_layer.out_features)
       
        self.fan_in_fan_out = fan_in_fan_out
        
        # Calls update_layer to initialize or update the adapter-related parameters (vera_A, vera_B, etc.) and set up the layer for the adapter specified by adapter_name.
        self.update_layer(vera_A, vera_B, r, vera_dropout, init_weights, d_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.scale = 0.7

    # safe_merge: If True, checks for numerical stability before merging.It ensures that potential issues with NaN values are caught before altering the base layer’s weights.
    # adapter_names: An optional parameter that can be a list of strings, specifying which adapters to merge. If it’s not provided, the function will merge all adapters.
    # orig_weights  This creates a copy of the original weights of the base layer before applying any changes. it is used for validation purposes, allowing you to check if the resulting weights after the merge are valid.
    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        adapter_names = adapter_names or list(self.vera_lambda_d.keys())
        # Iterating Over Active Adapters
        for active_adapter in adapter_names:
            if active_adapter in self.vera_lambda_d.keys():
                base_layer = self.get_base_layer()
                # base_layer will be used to update its weights with the computed changes from the adapter.
                delta_weight = self.get_delta_weight(active_adapter)
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    # This prevents further merging operations that could compromise the integrity of the model.
                    if not torch.isfinite(orig_weights + delta_weight).all():
                        raise ValueError(f"NaNs detected. The adapter {active_adapter} seems broken.")
                    base_layer.weight.data += delta_weight
                else:
                    base_layer.weight.data += delta_weight
                self.merged_adapters.append(active_adapter)

    # delta_weight will contain the adjustments needed to merge the adapter's learned information into the base layer.
    def get_delta_weight(self, adapter) -> torch.Tensor:
        vera_A = self.vera_A
        vera_B = self.vera_B
        lambda_d = self.vera_lambda_d
        lambda_b = self.vera_lambda_b

        sliced_A = vera_A[:, : self.in_features]
        sliced_B = vera_B[: self.out_features, :]
        return torch.matmul(lambda_b.unsqueeze(-1) * sliced_B, lambda_d.unsqueeze(-1) * sliced_A)

    # x: torch.Tensor: This is the input tensor that will be passed through the model. It typically represents a batch of input data.
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # First, it performs a standard forward pass through the base_layer (which is the underlying nn.Linear or similar layer).
        # print(f"xlinear{x.shape}")
        result = self.base_layer(x, *args, **kwargs)
        device = self.base_layer.weight.device
        previous_dtype = x.dtype
        #  Each active_adapter in self.active_adapters will be processed in the upcoming steps to modify the result.
        sliced_A = self.vera_A[:, : self.in_features].to(device)
        # print(sliced_A.shape)
        # print(self.in_features)
        sliced_B = self.vera_B[: self.out_features, :].to(device)
        # print(sliced_B.shape)
        # print(self.out_features)
        

        lambda_d = self.vera_lambda_d.to(device)
        lambda_b = self.vera_lambda_b.to(device)

        # re = torch.diag(lambda_b) @ sliced_B @ torch.diag(lambda_d) @ sliced_A
        # adapted_result = x @ re
        # # Add the original base layer output with the adapted result.
        # result = result + self.scale * adapted_result
        dropout = self.vera_dropout["vera"]
        result = result + lambda_b * F.linear(lambda_d * F.linear(dropout(x), sliced_A), sliced_B)
        result = result.to(previous_dtype)
                    
        return result
# operation effectively merges the contributions from the adapter into the final output, modifying it based on the learned parameters.

class MergedLinear(nn.Linear):
    def __init__(
        self,
        base_layer,
        vera_A,
        vera_B,     
        r: int = 0,
        vera_dropout: float = 0.0,
        num_heads: int = 12,
        embed_dim: int = 768,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        **kwargs,
    ):
       
        # super().__init__()
        # Linear.__init__(self, base_layer, **kwargs)
        # VeraLayer.__init__(self, base_layer, **kwargs)
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
          # Separate linear layers for q and v
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.q_proj = Linear(
                base_layer=nn.Linear(self.base_layer.in_features, self.dim),
                vera_A=vera_A,
                vera_B=vera_B,
                r=r,
                vera_dropout=vera_dropout
            )
        self.k_proj = nn.Linear(self.base_layer.in_features, self.dim)  # Keep original key projection
        self.v_proj = Linear(
                base_layer=nn.Linear(self.base_layer.in_features, self.dim),
                vera_A=vera_A,
                vera_B=vera_B,
                r=r,
                vera_dropout=vera_dropout
            )
        self.scale = 0.5

    # x: torch.Tensor: This is the input tensor that will be passed through the model. It typically represents a batch of input data.
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # First, it performs a standard forward pass through the base_layer (which is the underlying nn.Linear or similar layer).
        result = super().forward(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)
        K = torch.zeros_like(V)
        # K = self.k_proj(x)
        
        out = torch.cat([Q, K, V], dim=-1)
        # print(f"out{out.shape}")
        result = result + self.scale * out
        return result
# operation effectively merges the contributions from the adapter into the final output, modifying it based on the learned parameters.