import torch
import torch.nn as nn
from init_methods.fourier import FourierLinear
from init_methods.wavelet import WaveletLinear
from init_methods.svd import SVDLinear
from init_methods.kaiming import KaimingLinear

def get_projection_layer(method):

    projection_layer_classes = {
        'fourier': FourierLinear,
        'wavelet': WaveletLinear,
        'svd': SVDLinear,
        'kaiming': KaimingLinear
    }
    if method not in projection_layer_classes:
        raise ValueError(f"Unsupported method: {method}")
    return projection_layer_classes[method]

class BaseMergedLinear(nn.Linear):
    def __init__(
        self,
        base_layer,
        weight,
        vera_A,
        vera_B,
        method,
        r: int = 0,
        scale: float = 0,
        num_heads: int = 12,
        embed_dim: int = 768,
        **kwargs,
    ):
        super().__init__(base_layer.in_features, base_layer.out_features)
        
        self.base_layer = base_layer
        self.dim = self.base_layer.out_features // 3
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        # Split the weight tensor into Q, K, V components
        weight_q, weight_k, weight_v = torch.split(weight, self.dim, dim=0)

        # Initialize projection layers
        q = nn.Linear(self.base_layer.in_features, self.dim)
        v = nn.Linear(self.base_layer.in_features, self.dim)

        # Assign weights and biases
        with torch.no_grad():
            q.weight.copy_(weight_q)
            v.weight.copy_(weight_v)

        # Define projection layers based on the selected method
        projection_class = get_projection_layer(method)
            
        # Initialize separate linear layers for q and v
        self.q_proj = projection_class(
            base_layer=q,
            vera_A=vera_A,
            vera_B=vera_B,
            r=r
        )
        
        self.v_proj = projection_class(
            base_layer=v,
            vera_A=vera_A,
            vera_B=vera_B,
            r=r
        )

        self.scale = scale

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = super().forward(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)
        K = torch.zeros_like(V)
        out = torch.cat([Q, K, V], dim=-1)
        result = result + self.scale * out
        return result
