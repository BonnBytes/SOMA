import warnings
import torch
import torch.nn as nn




class BaseKaimingLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.r = {}
        self.vera_lambda_b = nn.ParameterDict({})
        self.vera_lambda_d = nn.ParameterDict({})
        self.vera_A = None
        self.vera_B = None
        self.base_layer = base_layer
        self.in_features, self.out_features = self._get_layer_dimensions(base_layer)
        self.kwargs = kwargs


    def get_base_layer(self):
        return self.base_layer

    def _get_layer_dimensions(self, base_layer):
        if isinstance(base_layer, nn.Linear):
            return base_layer.in_features, base_layer.out_features
        raise ValueError("Unsupported layer type")

    def reset_vera_parameters(self, d_initial: float = 0.1):
        with torch.no_grad():
            nn.init.zeros_(self.vera_lambda_d).fill_(d_initial)
            nn.init.zeros_(self.vera_lambda_b)

    def _move_adapter_to_device_of_base_layer(self):
        device = self.base_layer.weight.device
        self.vera_lambda_b = self.vera_lambda_b.to(device)
        self.vera_lambda_d = self.vera_lambda_d.to(device)
        
    def update_layer(self,
        vera_A,
        vera_B,
        r,
        d_initial: float = 0.1
        ):
        """Define the parameter initialization specific to Kaiming Linear."""

        self.vera_lambda_b = nn.Parameter(torch.zeros(self.out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.ones(r), requires_grad=True)
        self.vera_A = nn.Parameter(vera_A[:, : self.in_features], requires_grad=False)
        self.vera_B = nn.Parameter(vera_B[: self.out_features, :], requires_grad=False)

        self._move_adapter_to_device_of_base_layer()

class KaimingLinear(nn.Linear, BaseKaimingLayer):
    """SVD-based VeRA Linear layer."""
    def __init__(
        self,
        base_layer,
        weight,
        vera_A,
        vera_B,     
        r: int = 0,
        **kwargs,
    ):
        BaseKaimingLayer.__init__(self, base_layer, **kwargs)
        self.update_layer(vera_A, vera_B, r)
        self.weight = weight
    

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        A= self.vera_A.to(x.device)
        B = self.vera_B.to(x.device)
        lambda_b, lambda_d = self.vera_lambda_b.to(x.device), self.vera_lambda_d.to(x.device)
        re = torch.diag(lambda_b) @ B @ torch.diag(lambda_d) @ A
        delta = x @ re

        return delta

