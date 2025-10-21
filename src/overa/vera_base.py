"""VeRA Base Layer."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch as th


class VeraBase(ABC, th.nn.Module):
    """Vera Base layer."""

    def __init__(self, base_layer: th.nn.Module):
        """Initialise VeRA base.

        Args:
            base_layer (th.nn.Module): VeRA base linear layer.

        Raises:
            ValueError: Raise error if base projection is not square.
        """
        super(VeraBase, self).__init__()
        self.base_layer = base_layer
        self.vera_lambda_d: Optional[th.nn.Parameter] = None
        self.fixed_lambda_d: Optional[th.nn.Parameter] = None
        self.vera_lambda_b: Optional[th.nn.Parameter] = None
        self.A_vera: Optional[Any] = None
        self.B_vera: Optional[Any] = None
        self.in_feat, self.out_feat = self.__get_layer_dimensions()
        if self.in_feat != self.out_feat:
            raise ValueError("Non-square base projection is not allowed.")

    def __get_layer_dimensions(self) -> Tuple[int, int]:
        """Get base layer dimensions.

        Raises:
            ValueError: Error if layer isn't linear.

        Returns:
            Tuple[int, int]: Tuple containing layer dimensions.
        """
        if isinstance(self.base_layer, th.nn.Linear):
            return (self.base_layer.in_features, self.base_layer.out_features)
        raise ValueError("Only Linear layer is supported.")

    def __move_to_device(self):
        """Move VeRA parameters to model device."""
        device = self.base_layer.weight.device
        self.vera_lambda_d = self.vera_lambda_d.to(device)
        self.vera_lambda_b = self.vera_lambda_b.to(device)
        self.fixed_lambda_d = self.fixed_lambda_d.to(device)
        self.A_vera = self.A_vera.to(device)
        self.B_vera = self.B_vera.to(device)

    @abstractmethod
    def compute_bases(self):
        """Abstract method to compute base."""
        pass

    @abstractmethod
    def update_layer(self, *args):
        """Abstract method to update layer parameters."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Abstract methd to model forward."""
        pass
