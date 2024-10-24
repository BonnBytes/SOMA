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
        self.vera_lambda_b = nn.ParameterDict({})
        self.vera_lambda_d = nn.ParameterDict({})

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
    # This method updates the parameters associated with a specified adapter for the Vera layer.
    def update_layer(
        self,
        adapter_name,
        vera_A: BufferDict,
        vera_B: BufferDict,
        r,
        vera_dropout,
        init_weights,
        d_initial: float = 0.1,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.vera_dropout[adapter_name] = nn.Dropout(p=vera_dropout) if vera_dropout > 0.0 else nn.Identity()

        self.vera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.vera_lambda_d[adapter_name] = nn.Parameter(torch.randn(r), requires_grad=True)

        self.vera_A = vera_A
        self.vera_B = vera_B

        if adapter_name not in vera_A:
            if len(self.vera_A) < 1:
                raise ValueError("The `vera_A` and `vera_B` buffers are empty. This should not happen.")
            #  If the adapter name is not in vera_A, the function retrieves the first parameter from the vera_A and vera_B buffers. 
            vera_A_param = list(self.vera_A.values())[0]
            vera_B_param = list(self.vera_B.values())[0]
            # Checks if the input dimension of vera_A_param is less than the expected input features (self.in_features). If so, raises a ValueError.
            if vera_A_param.shape[1] < self.in_features:
                raise ValueError(f"vera_A has insufficient input size {vera_A_param.shape[1]}, needs {self.in_features}")
            if vera_B_param.shape[0] < self.out_features:
                raise ValueError(f"vera_B has insufficient output size {vera_B_param.shape[0]}, needs {self.out_features}")

            self.vera_A[adapter_name] = vera_A_param
            self.vera_B[adapter_name] = vera_B_param

        if init_weights:
            self.reset_vera_parameters(adapter_name, d_initial)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_vera_parameters(self, adapter_name, d_initial: float = 0.1):
        # Resets the vera_lambda_d and vera_lambda_b parameters for the given adapter by filling them with zeros and initializing vera_lambda_d with the provided d_initial.
        with torch.no_grad():
            nn.init.zeros_(self.vera_lambda_d[adapter_name]).fill_(d_initial)
            nn.init.zeros_(self.vera_lambda_b[adapter_name])

    def _move_adapter_to_device_of_base_layer(self, adapter_name):
        # Moves the adapter's parameters (vera_lambda_b, vera_lambda_d) to the same device as the base layer.
        device = self.base_layer.weight.device
        self.vera_lambda_b[adapter_name] = self.vera_lambda_b[adapter_name].to(device)
        self.vera_lambda_d[adapter_name] = self.vera_lambda_d[adapter_name].to(device)

    def set_adapter(self, adapters: List[str]):
        self.active_adapters = adapters


class Linear(nn.Linear, VeraLayer):
    def __init__(
        self,
        base_layer,
        vera_A: BufferDict,
        vera_B: BufferDict,
        adapter_name: str,
        r: int = 0,
        vera_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        **kwargs,
    ):
        # adapter name: The name of the adapter being used (could be useful for multi-task learning).
        super(nn.Linear, self).__init__()
        VeraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        # Calls update_layer to initialize or update the adapter-related parameters (vera_A, vera_B, etc.) and set up the layer for the adapter specified by adapter_name.
        self.update_layer(adapter_name, vera_A, vera_B, r, vera_dropout, init_weights, d_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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
        vera_A = self.vera_A[adapter]
        vera_B = self.vera_B[adapter]
        lambda_d = self.vera_lambda_d[adapter]
        lambda_b = self.vera_lambda_b[adapter]

        sliced_A = vera_A[:, : self.in_features]
        sliced_B = vera_B[: self.out_features, :]
        return torch.matmul(lambda_b.unsqueeze(-1) * sliced_B, lambda_d.unsqueeze(-1) * sliced_A)

    # x: torch.Tensor: This is the input tensor that will be passed through the model. It typically represents a batch of input data.
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # First, it performs a standard forward pass through the base_layer (which is the underlying nn.Linear or similar layer).
        result = self.base_layer(x, *args, **kwargs)
        #  Each active_adapter in self.active_adapters will be processed in the upcoming steps to modify the result.
        if not self._disable_adapters:
            for active_adapter in self.active_adapters:
                if active_adapter in self.vera_lambda_d:
                    sliced_A = self.vera_A[active_adapter][:, : self.in_features]
                    sliced_B = self.vera_B[active_adapter][: self.out_features, :]
                    lambda_d = self.vera_lambda_d[active_adapter]
                    lambda_b = self.vera_lambda_b[active_adapter]
                    result += F.linear(lambda_b * F.linear(lambda_d * x, sliced_A), sliced_B)
        return result
# operation effectively merges the contributions from the adapter into the final output, modifying it based on the learned parameters.