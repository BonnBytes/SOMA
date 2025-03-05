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
        
<<<<<<< HEAD
        self.base_layer = base_layer   # Original qkv layer
=======
        self.base_layer = base_layer
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
        self.dim = self.base_layer.out_features // 3
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        # Split the weight tensor into Q, K, V components
        weight_q, weight_k, weight_v = torch.split(weight, self.dim, dim=0)

<<<<<<< HEAD
        # # Initialize projection layers
        # q = nn.Linear(self.base_layer.in_features, self.dim, bias=False)
        # v = nn.Linear(self.base_layer.in_features, self.dim, bias=False)

        # # Assign weights and biases
        # with torch.no_grad():
        #     q.weight.copy_(weight_q)
        #     v.weight.copy_(weight_v)
=======
        # Initialize projection layers
        q = nn.Linear(self.base_layer.in_features, self.dim)
        v = nn.Linear(self.base_layer.in_features, self.dim)

        # Assign weights and biases
        with torch.no_grad():
            q.weight.copy_(weight_q)
            v.weight.copy_(weight_v)
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc

        # Define projection layers based on the selected method
        projection_class = get_projection_layer(method)
            
        # Initialize separate linear layers for q and v
        self.q_proj = projection_class(
<<<<<<< HEAD
            base_layer=nn.Linear(self.base_layer.in_features, self.dim, bias=False),
            weight = weight_q,
=======
            base_layer=q,
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
            vera_A=vera_A,
            vera_B=vera_B,
            r=r
        )
        
        self.v_proj = projection_class(
<<<<<<< HEAD
            base_layer=nn.Linear(self.base_layer.in_features, self.dim, bias=False),
            weight = weight_v,
=======
            base_layer=v,
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
            vera_A=vera_A,
            vera_B=vera_B,
            r=r
        )

<<<<<<< HEAD
        # # Initialize adapters for Q and V
        # projection_class = get_projection_layer(method)
        # self.q_proj = projection_class(
        #     base_layer=nn.Linear(self.base_layer.in_features, self.dim, bias=False),
        #     vera_A=vera_A,
        #     vera_B=vera_B,
        #     r=r
        # )
        # self.v_proj = projection_class(
        #     base_layer=nn.Linear(self.base_layer.in_features, self.dim, bias=False),
        #     vera_A=vera_A,
        #     vera_B=vera_B,
        #     r=r
        # )

        self.scale = scale

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        original_qkv = self.base_layer(x)
        Q_orig, K_orig, V_orig = torch.split(original_qkv, self.dim, dim=-1)
        
        # Compute deltas
        delta_Q = self.q_proj(x)
        delta_V = self.v_proj(x)
        
        # Apply scale and combine
        new_Q = Q_orig + self.scale * delta_Q
        new_V = V_orig + self.scale * delta_V
        new_qkv = torch.cat([new_Q, K_orig, new_V], dim=-1)
        return new_qkv


=======
        self.scale = scale

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = super().forward(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)
        K = torch.zeros_like(V)
        out = torch.cat([Q, K, V], dim=-1)
        result = result + self.scale * out
        return result
>>>>>>> c4b8a3b014b7c0c3f6a3fb1c4d5afbe6842d1dfc
