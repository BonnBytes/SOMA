import warnings
import torch
import torch.nn as nn




class BaseSVDLayer(nn.Module):
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
<<<<<<< HEAD
        weight,
        d_initial: float = 0.1
        ):
        """Define the parameter initialization specific to SVD Linear."""
        self.vera_lambda_b = nn.Parameter(torch.zeros(self.out_features), requires_grad=True)

        U, S , Vt = torch.linalg.svd(weight, full_matrices=False)
=======
        d_initial: float = 0.1
        ):
        """Define the parameter initialization specific to SVD Linear."""
        self.vera_lambda_b = nn.Parameter(torch.ones(self.out_features), requires_grad=True)

        U, S , Vt = torch.linalg.svd(self.base_layer.weight, full_matrices=False)
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
        self.vera_B = nn.Parameter(U[:, :r], requires_grad=False)
        self.vera_lambda_d = nn.Parameter(S[:r], requires_grad=True)
        self.vera_A = nn.Parameter(Vt[:r, :], requires_grad=False)

<<<<<<< HEAD
        # self.reset_vera_parameters(d_initial)
=======
        self.reset_vera_parameters(d_initial)
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
        self._move_adapter_to_device_of_base_layer()

class SVDLinear(nn.Linear, BaseSVDLayer):
    """SVD-based VeRA Linear layer."""
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
        BaseSVDLayer.__init__(self, base_layer, **kwargs)
<<<<<<< HEAD
        self.weight = weight
        self.update_layer(vera_A, vera_B, r, weight)
    

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        A, B = self.vera_A.to(x.device), self.vera_B.to(x.device)
        lambda_b, lambda_d = self.vera_lambda_b.to(x.device), self.vera_lambda_d.to(x.device)
        re = torch.diag(lambda_b) @ B @ torch.diag(lambda_d) @ A
        delta = x @ re
        return delta
=======
        self.update_layer(vera_A, vera_B, r)
    

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        A, B = self.vera_A.to(x.device), self.vera_B.to(x.device)
        lambda_b, lambda_d = self.vera_lambda_b.to(x.device), self.vera_lambda_d.to(x.device)
        re = torch.diag(lambda_b) @ B @ torch.diag(lambda_d) @ A
        return result + x @ re
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc

