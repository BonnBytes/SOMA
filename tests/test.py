import torch
import pytest
import os
from scripts.train import VeraTrainer
from layers.merged import BaseMergedLinear
from layers.utils import random_permutation_matrix
from init_methods.wavelet import WaveletLinear

@pytest.fixture
def trainer():
    return VeraTrainer(dataset='Cifar100', seed=1, device="cuda")

# -------------------- Test Initialization Helpers --------------------
def test_kaiming_init(trainer):
    shape = (10, 5)
    generator = torch.Generator(device="cpu").manual_seed(0)
    output_tensor = trainer._kaiming_init(shape, generator)
    assert output_tensor.shape == shape
    assert torch.is_tensor(output_tensor)

def test_fourier_matrix(trainer):
    n = 768
    matrix = trainer.fourier_matrix(n)
    assert matrix.shape == (n, n)
    assert torch.is_tensor(matrix)
    assert torch.is_complex(matrix)

# -------------------- Test Model Creation --------------------
def test_create_model(trainer):
    pre_trained_check_path = 'test_pretrained.pth'
    model, in_dim, out_dim = trainer.create_model(pre_trained_check_path)
    
    # Check model type and head
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "head")
    assert model.head.out_features == trainer.num_classes
    
    # Check dimensions
    assert in_dim > 0 and out_dim > 0
    
    # Check checkpoint
    assert os.path.exists(pre_trained_check_path)
    checkpoint = torch.load(pre_trained_check_path)
    assert "head.weight" in checkpoint  # Verify model state is saved

# -------------------- Test Model Modification --------------------
def test_freeze_parameters(trainer):
    pre_trained_check_path = 'test_pretrained.pth'
    model, in_dim, out_dim = trainer.create_model(pre_trained_check_path)
    model = trainer.apply_vera_to_attention(model, out_dim, in_dim, scale=0.1, rank=256)
    model = trainer.freeze_parameters(model)
    
    # Check parameter freezing
    vera_params = any("vera_lambda" in name for name, p in model.named_parameters())
    head_params = all(p.requires_grad for p in model.head.parameters())
    base_params = all(not p.requires_grad for name, p in model.named_parameters() if "vera" not in name and "head" not in name)
    
    assert vera_params and head_params and base_params, "Freezing logic failed"

def test_apply_vera_to_attention(trainer):
    r = 256
    model, in_dim, out_dim = trainer.create_model('dummy_path')
    model = trainer.apply_vera_to_attention(model, out_dim, in_dim, scale=0.1, rank=r)
    
    # Verify parameter existence and shapes
    for name, param in model.named_parameters():
        if "vera_lambda_b" in name:
            assert param.shape == (out_dim,), f"Shape mismatch for {name}"
        elif "vera_lambda_d" in name:
            assert param.shape == (r,), f"Shape mismatch for {name}"
        elif "vera_B" in name:
            assert param.shape == (out_dim, r), f"Shape mismatch for {name}"
        elif "vera_A" in name:
            assert param.shape == (r, in_dim), f"Shape mismatch for {name}"

# -------------------- Test Base Merged Layer --------------------
def test_base_merged_linear():
    base_layer = torch.nn.Linear(10, 30)
    weight = torch.randn(30, 10)
    vera_A = torch.randn(256, 10)
    vera_B = torch.randn(30, 256)
    
    layer = BaseMergedLinear(
        base_layer, weight, vera_A, vera_B, method="kaiming", r=256, scale=0.1
    )
    x = torch.randn(5, 10)
    output = layer(x)
    assert output.shape == (5, 30)

# -------------------- Test Utils --------------------
def test_random_permutation_matrix():
    size = 5
    P = random_permutation_matrix(size)
    assert P.shape == (size, size)
    assert torch.allclose(P @ P.T, torch.eye(size), atol=1e-6), "Not orthogonal"

# -------------------- Test Wavelet Layer --------------------
def test_wavelet_linear_forward():
    base_layer = torch.nn.Linear(10, 30)
    weight = torch.randn(30, 10)
    vera_A = torch.randn(10, 10)
    vera_B = torch.randn(30, 10)
    wavelet_layer = WaveletLinear(base_layer, weight, vera_A, vera_B, r=10)
    x = torch.randn(5, 10)
    output = wavelet_layer(x)
    assert output.shape == (5, 30)

# -------------------- Test Training Dynamics --------------------
def test_update_parameters(trainer):
    model, in_dim, out_dim = trainer.create_model('dummy_path')
    model = trainer.apply_vera_to_attention(model, out_dim, in_dim, scale=0.1, rank=256)
    model = trainer.freeze_parameters(model)
    
    optimizer = trainer._get_optimizer(model, 'Cifar100')
    initial_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    
    # Perform dummy training step
    x = torch.randn(2, 3, 224, 224, device=trainer.device)
    y = torch.randint(0, 100, (2,), device=trainer.device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    
    # Check parameter updates
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.equal(initial_params[name], param), f"{name} not updated"