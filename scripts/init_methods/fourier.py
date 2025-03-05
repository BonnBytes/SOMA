import warnings
import torch
import torch.nn as nn


from layers.utils import random_permutation_matrix


class BaseFourierLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
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
<<<<<<< HEAD
            nn.init.zeros_(self.fixed_lambda_d).fill_(d_initial)
=======
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
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
        """Define the parameter initialization specific to Fourier Linear."""
<<<<<<< HEAD
        self.vera_lambda_b = nn.Parameter(torch.zeros(self.out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.ones(r), requires_grad=True)
        self.fixed_lambda_d = nn.Parameter(torch.ones(self.out_features - r), requires_grad=False)
=======
        self.vera_lambda_b = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.vera_lambda_d = nn.Parameter(torch.randn(r), requires_grad=True)
        self.fixed_lambda_d = nn.Parameter(torch.randn(self.out_features - r), requires_grad=False)
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
        # Generate fixed random permutation matrix Pi
        self.pi = random_permutation_matrix(self.out_features).to(dtype=torch.cdouble)

        self.vera_B = nn.Parameter(vera_B.T, requires_grad=False)
        self.vera_A = nn.Parameter(self.pi @ vera_A.T, requires_grad=False)

        self.reset_vera_parameters(d_initial)
        self._move_adapter_to_device_of_base_layer()

class FourierLinear(nn.Linear, BaseFourierLayer):
    """Fourier-based VeRA Linear layer."""
    def __init__(
        self,
        base_layer,
<<<<<<< HEAD
        weight,
=======
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
        vera_A,
        vera_B,     
        r: int = 0,
        **kwargs,
    ):
        BaseFourierLayer.__init__(self, base_layer, **kwargs)
        self.update_layer(vera_A, vera_B, r)
    

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        A, B = self.vera_A.to(x.device), self.vera_B.to(x.device)
        lambda_b, lambda_d = self.vera_lambda_b.to(x.device).to(dtype=torch.cdouble), torch.cat([self.vera_lambda_d.to(x.device), self.fixed_lambda_d.to(x.device)], dim=0).to(dtype=torch.cdouble)
        re = torch.diag(lambda_b) @ B @ torch.diag(lambda_d) @ A
        abs_value = torch.abs(x.to(dtype=torch.cdouble) @ re)
<<<<<<< HEAD
        return abs_value.to(torch.float32)
=======
        return result + abs_value.to(torch.float32)
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc

