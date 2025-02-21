import torch
import pytest
from scripts.train import VeraTrainer
from layers.merged import BaseMergedLinear
from layers.utils import random_permutation_matrix

@pytest.fixture
def trainer():
    return VeraTrainer(dataset='Cifar100', seed = 1, device= "cuda")

 # -------------------- Test Initialization Helpers --------------------
def test_kaiming_init(trainer):
    shape = (10,5)
    generator = torch.Generator(device="cuda").manual_seed(0)
    output_tensor = trainer._kaiming_init(shape, generator)
    assert output_tensor.shape == shape
    assert torch.is_tensor(output_tensor)

def test_fourier_matrix(trainer):
    n = 768
    matrix = trainer.fourier_matrix(n)
    assert matrix.shape == (n,n)
    assert torch.is_tensor(matrix)
    assert torch.is_complex(matrix)

# -------------------- Test Model Creation --------------------
def test_create_model(trainer):
    pre_trained_check_path = 'test_pretrained.pth'
    model, in_dim, out_dim = trainer.create_model(pre_trained_check_path)

    #Model should be a PyTorch module.
    assert isinstance(model, torch.nn.module) 

    assert hasattr(model, "head")
    assert model.head.out_features == trainer.num_classes

    # Check dimensions, Input and output dimensions should be a positive integer.
    assert isinstance(in_dim, int) and in_dim > 0
    assert isinstance(out_dim, int) and out_dim > 0

    assert os.path.exists(pre_trained_check_path)

    #Checkpoint should contain model state.
    assert "model_state_dict" in checkpoint

# -------------------- Test Model Modification --------------------
def test_freeze_parameters(trainer):
    """Check if parameters are correctly frozen."""
    pre_trained_check_path = 'test_pretrained.pth'
    model, out_dim, in_dim = trainer.create_model(pre_trained_check_path)
    model = trainer.apply_vera_to_attention(model, out_dim, in_dim, scale=0.1, rank=256)
    model = trainer.freeze_parameters(model)


    all_frozen = all(not param.requires_grad for name, param in model.named_parameters() if "vera_lambda" not in name)
    all_lambda = all(param.requires_grad for name, param in model.named_parameters() if "vera_lambda" in name)
    all_head = all(param.requires_grad for param in model.head.parameters())

    assert all_frozen
    assert all_lambda
    assert all_head

def test_apply_vera_to_attention(trainer):
    """Test if VeraTrainer modifies model correctly."""
    r = 256
    pre_trained_check_path = 'test_pretrained.pth'
    model, in_dim, out_dim = trainer.create_model(pre_trained_check_path)
    model = trainer.apply_vera_to_attention(model, out_dim, in_dim, scale=0.1, rank=r)
    
    #check if vera parameters exist in the modified model
    base_layers = [param for name, param in model.named_parameters() if ".attn.qkv.base_layer" in name]
    lambda_b_params = [param for name, param in model.named_parameters() if "vera_lambda_b" in name]
    lambda_d_params = [param for name, param in model.named_parameters() if "vera_lambda_d" in name]
    B_params = [param for name, param in model.named_parameters() if "vera_B" in name]
    A_params = [param for name, param in model.named_parameters() if "vera_A" in name]
    
    assert base_layers, "qkv base layers not found in model."
    assert lambda_b_params, "vera_lambda_b not found in model."
    assert lambda_d_params, "vera_lambda_d not found in model."
    assert B_params, "vera_B not found in model."
    assert A_params, "vera_A not found in model."

    for i in range(base_layers):
        #not sure if I should have shape[0] or shape[1]
        assert lambda_b_params[i].shape == (base_layers[i].shape[0],), f"vera_lambda_b at layer {i} has incorrect shape {lambda_b_params[i].shape}, expected {base_layers[i].shape[0]}"
        assert lambda_d_params[i].shape == (r,), f"vera_lambda_b at layer {i} has incorrect shape {lambda_d_params[i].shape}, expected {r}"
        assert B_params[i].shape == (base_layers[i].shape[0],r), f"vera_B at layer {i} has incorrect shape {lambda_d_params[i].shape}, expected {base_layers[i].shape[0]},{r}"
        assert A_params[i].shape == (r,base_layers[i].shape[1]), f"vera_A at layer {i} has incorrect shape {lambda_d_params[i].shape}, expected {r}, {base_layers[i].shape[1]}"

# -------------------- Test base merged layer --------------------
def test_base_merged_linear():
    """Test BaseMergedLinear initialization and forward pass."""
    base_layer = torch.nn.Linear(10, 30)
    weight = torch.randn(30, 10)
    vera_A = torch.randn(256, 10)
    vera_B = torch.randn(30, 256)

    layer = BaseMergedLinear(base_layer, weight, vera_A, vera_B, method="kaiming", r=256)
    
    x = torch.randn(5, 10)  # Input tensor
    output = layer(x)
    
    assert output.shape == (5, 30)

# -------------------- Test random permutation in utils.py --------------------
def test_random_permutation_matrix():
    """Test permutation matrix generation."""
    size = 5
    P = random_permutation_matrix(size)
    
    assert P.shape == (size, size)
    assert torch.allclose(P @ P.T, torch.eye(size, dtype=torch.float64)), "P should be orthogonal."