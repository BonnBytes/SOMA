import math
import warnings
from dataclasses import asdict
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D
from .layers import Linear, VeraLayer  # Import the necessary layers

class VeraModel(nn.Module):
    """
    Simplified VeraModel without using BaseTuner or tuner-utils. 
    """

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__()
        self.model = model
        self.config = config
        self.adapter_name = adapter_name
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.vera_A = None
        self.vera_B = None
        self.active_adapter = adapter_name

    def _find_dim(self) -> tuple[int, int]:
        largest_shape = None
        for key, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module_shape = module.out_features, module.in_features
            elif isinstance(module, Conv1D):
                # reverse to (in_channels, out_channels, kernel_size)
                module_shape = module.weight.shape[::-1]
            else:
                continue

            if largest_shape is None:
                largest_shape = module_shape
            else:
                largest_shape = tuple(max(a, b) for a, b in zip(largest_shape, module_shape))

        if largest_shape is None:
            raise ValueError("No compatible layers found for Vera.")
        return largest_shape

    def _kaiming_init(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        # torch.empty(shape) creates an empty tensor with the given shape without initializing its values. This tensor is used only for shape inspection to calculate the fan_in.
        # _calculate_correct_fan(tensor, "fan_in"): This function calculates the fan-in (the number of input units) for the layer. The fan-in is the number of input neurons (the dimension feeding into the current layer) and is important for determining the correct variance for initialization. The "fan_in" argument specifies that we are interested in calculating the number of input features to the layer.
        fan = _calculate_correct_fan(torch.empty(shape), "fan_in")
        # The gain is a scaling factor that depends on the activation function. For ReLU (Rectified Linear Unit), the gain is set to √2. This value helps scale the variance of the weights correctly, considering ReLU tends to zero out negative values.
        gain = math.sqrt(2)
        # gain / math.sqrt(fan): This computes the standard deviation (std) for the Kaiming initialization based on the gain and fan-in value. This standard deviation ensures that the weights are initialized with an appropriate scale for the ReLU activation.
        std = gain / math.sqrt(fan)
        # In Kaiming uniform initialization, weights are drawn from a uniform distribution between -bound and bound.
        # This is a common scaling factor when converting between a normal distribution and a uniform distribution, ensuring that the uniform initialization maintains the same variance as the normal distribution. The constant √3 comes from the relationship between the variance of a uniform distribution and its range.
        bound = math.sqrt(3.0) * std
        # The uniform distribution ensures that the initial weights are distributed between -bound and bound, with a variance that is compatible with the requirements of Kaiming initialization for ReLU.The generator is used to control the randomness.
        return torch.empty(shape).uniform_(-bound, bound, generator=generator)

    def _init_vera_A_vera_B(self) -> None:
        linear_out_dim, linear_in_dim = self._find_dim()
        # The device="cpu" argument specifies that the generator will run on the CPU. This is useful when you want to ensure that your random number generation is consistent across different devices, such as GPUs and CPUs.
        generator = torch.Generator(device="cpu").manual_seed(self.config.projection_prng_key)
        self.vera_A = self._kaiming_init((self.config.r, linear_in_dim), generator)
        self.vera_B = self._kaiming_init((linear_out_dim, self.config.r), generator)

    # In this function, it replaces the existing child module (identified by child_name) in parent with new_module.
    def _replace_module(self, parent, child_name, new_module):
        setattr(parent, child_name, new_module)
    # During fine-tuning, this method is called to replace standard layers with adapter-enabled layers that allow for low-rank fine-tuning.
    def create_and_replace(self, target, target_name, parent):
        r = self.config.r
        vera_A = self.vera_A
        vera_B = self.vera_B

        if isinstance(target, nn.Linear):
            # Replace with custom VeraLayer (Linear implementation)
            new_module = Linear(
                base_layer=target, 
                vera_A=self.vera_A, 
                vera_B=self.vera_B, 
                adapter_name=self.adapter_name, 
                r=r, 
                vera_dropout=self.config.vera_dropout
            )
            self._replace_module(parent, target_name, new_module)
        elif isinstance(target, Conv1D):
            # Handle Conv1D replacement (implementation needed for Conv1D)
            new_module = Conv1D(target.weight.shape[0], target.weight.shape[1])
            self._replace_module(parent, target_name, new_module)

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name

    def merge_and_unload(self):
        # Simplified merging logic
        for key, module in self.model.named_modules():
            if isinstance(module, Linear) and hasattr(module, 'merge'):
                module.merge()

        return self.model

    def delete_adapter(self, adapter_name: str):
        if self.active_adapter == adapter_name:
            self.active_adapter = None

    def unload(self):
        return self.merge_and_unload()

# Example usage:
# model = VeraModel(base_model, vera_config, adapter_name="default")
# model.create_and_replace(target_module, target_name, parent_module)
