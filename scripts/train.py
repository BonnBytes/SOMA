import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from timm import create_model
from timm.models.vision_transformer import VisionTransformer, Attention
from buffer_dict import BufferDict  # Assuming this is implemented
from layers import Linear as VeraLinear
from layers import MergedLinear as VeraMergedLinear
from layers_svd import MergedLinear as VeraSVDLinear
from layers_wavelet import MergedLinear as VeraWaveletLinear
from layers_fourier import MergedLinear as VeraFourierLinear
from torch.nn.init import _calculate_correct_fan
import math
from tqdm import tqdm
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
from scipy.linalg import hadamard
import random
import matplotlib.pyplot as plt
import ptwt
import pywt
# Configuration for VERA
class VeraConfig:
    r = 256
    vera_dropout = 0.1

def _find_dim(model) -> tuple[int, int]:
        largest_shape = None
        for key, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_shape = module.out_features, module.in_features
            else:
                continue

            if largest_shape is None:
                largest_shape = module_shape
            else:
                largest_shape = tuple(max(a, b) for a, b in zip(largest_shape, module_shape))

        if largest_shape is None:
            raise ValueError("No compatible layers found for Vera.")
        return largest_shape

def _kaiming_init(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
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

    


# def matmul_wht(x, h_mat=None, inverse=False):
#     """
#     Welsh-Hadamard transform by matrix multiplication.
#     @ param x: The sequence to be transformed [batchsize, seq_len].
#     @ param inverse: If true computes the inverse transform.
#     """
#     n = x.shape[-1]

#     if h_mat is None:
#         h_mat = torch.from_numpy(hadamard(n).astype(np.float32))
#         if x.device.type == 'cuda':
#             h_mat = h_mat.cuda()
#     y = torch.nn.functional.linear(x, h_mat, bias=None)
#     if not inverse:
#         y = y/n
#     return y



# Function to apply VERA modifications to the attention layers
def apply_vera_to_attention(model, vera_config, linear_out_dim, linear_in_dim, scale):
    # print("yeeessssssssssssssssssssssss222222")
    # Check if linear_in_dim is a power of two
    new_in_dim = 0
    if (linear_in_dim & (linear_in_dim - 1)) != 0:  # Not a power of two
        new_in_dim = 2 ** math.ceil(math.log2(linear_in_dim))
    else:
        new_in_dim = linear_in_dim

    generator = torch.Generator(device="cpu").manual_seed(0)
    vera_A = _kaiming_init((vera_config.r, new_in_dim), generator)
    vera_B = _kaiming_init((linear_out_dim, vera_config.r), generator)
    # 

 
    # print(vera_A.shape)
    # print(vera_B.shape)

    # # SVD initialization
    # U_A, S_A, Vt_A = svd_init((vera_config.r, linear_in_dim), generator)
    # U_B, S_B, Vt_B = svd_init((linear_out_dim, vera_config.r), generator)
    
    # # Construct A and B matrices with perturbations for training
    # vera_A = U_A @ torch.diag(S_A) @ Vt_A  # A = U Σ V^T
    # vera_B = U_B @ torch.diag(S_B) @ Vt_B  # B = U Σ V^T
    
    qkv_layers = []
    proj_layers = []

    for name, _ in model.named_modules():
        if "qkv" in name:
            qkv_layers.append(name.split("."))
        if name[-5:] == ".proj" and "patch_embed" not in name:
            proj_layers.append(name.split("."))

    for idx, (*parent, k, _, _) in enumerate(qkv_layers):
        nm = '.'.join(qkv_layers[idx])
        mod = model.get_submodule(nm)
        b = True if mod.bias is not None else False
        new_mod = VeraMergedLinear(
            base_layer=model.get_submodule(".".join(parent))[int(k)].attn.qkv,
            vera_A=vera_A,
            vera_B=vera_B,
            r=vera_config.r,
            vera_dropout=vera_config.vera_dropout,
            scale = scale
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias if b else None
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod


    return model

# Function to apply VERA modifications to the attention layers
def apply_vera_to_attention_SVD(model, vera_config, linear_out_dim, linear_in_dim, scale):

    qkv_layers = []
    proj_layers = []

    for name, _ in model.named_modules():
        if "qkv" in name:
            qkv_layers.append(name.split("."))
        if name[-5:] == ".proj" and "patch_embed" not in name:
            proj_layers.append(name.split("."))

    for idx, (*parent, k, _, _) in enumerate(qkv_layers):
        nm = '.'.join(qkv_layers[idx])
        mod = model.get_submodule(nm)
        b = True if mod.bias is not None else False
        # print(f"mod.weight:{mod.weight.shape}")
        new_mod = VeraSVDLinear(
            base_layer=model.get_submodule(".".join(parent))[int(k)].attn.qkv,
            weight = mod.weight.data,
            r=vera_config.r,
            vera_dropout=vera_config.vera_dropout,
            scale = scale
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias if b else None
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod


    return model


# Function to apply VERA modifications to the attention layers
def apply_vera_to_attention_Wavelet(model, vera_config, linear_out_dim, linear_in_dim, scale):

    matrix_A = ptwt.matmul_transform.construct_boundary_a("haar", length=768, device="cpu").to_dense()
    matrix_B = ptwt.matmul_transform.construct_boundary_s("haar", length=768, device="cpu").to_dense()
    matrix_A[768//2:, :] = 0.0

    qkv_layers = []
    proj_layers = []

    for name, _ in model.named_modules():
        if "qkv" in name:
            qkv_layers.append(name.split("."))
        if name[-5:] == ".proj" and "patch_embed" not in name:
            proj_layers.append(name.split("."))

    for idx, (*parent, k, _, _) in enumerate(qkv_layers):
        nm = '.'.join(qkv_layers[idx])
        mod = model.get_submodule(nm)
        b = True if mod.bias is not None else False
        # print(f"mod.weight:{mod.weight.shape}")
        new_mod = VeraWaveletLinear(
            base_layer=model.get_submodule(".".join(parent))[int(k)].attn.qkv,
            vera_A=matrix_A,
            vera_B=matrix_B,
            r=vera_config.r,
            vera_dropout=vera_config.vera_dropout,
            scale = scale
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias if b else None
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod


    return model

# Construct Fourier transform matrix
def fourier_matrix(n):
    indices = torch.arange(n, dtype=torch.float64)
    real_part = torch.cos(-2 * torch.pi * torch.outer(indices, indices) / n)
    imag_part = torch.sin(-2 * torch.pi * torch.outer(indices, indices) / n)
    return torch.complex(real_part, imag_part)

# Function to apply VERA modifications to the attention layers
def apply_vera_to_attention_Fourier(model, vera_config, linear_out_dim, linear_in_dim, scale):

    matrix_A = fourier_matrix(768)
    matrix_B = matrix_A.conj().T / 768
    matrix_A_copy = matrix_A.clone()
    matrix_A_copy[768 // 2:, :] = 0.0

    qkv_layers = []
    proj_layers = []

    for name, _ in model.named_modules():
        if "qkv" in name:
            qkv_layers.append(name.split("."))
        if name[-5:] == ".proj" and "patch_embed" not in name:
            proj_layers.append(name.split("."))

    for idx, (*parent, k, _, _) in enumerate(qkv_layers):
        nm = '.'.join(qkv_layers[idx])
        mod = model.get_submodule(nm)
        b = True if mod.bias is not None else False
        # print(f"mod.weight:{mod.weight.shape}")
        new_mod = VeraFourierLinear(
            base_layer=model.get_submodule(".".join(parent))[int(k)].attn.qkv,
            vera_A=matrix_A_copy,
            vera_B=matrix_B,
            r=vera_config.r,
            vera_dropout=vera_config.vera_dropout,
            scale = scale
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias if b else None
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod


    return model

# Freeze all parameters except lambda_b and lambda_d in the VERA layers
def freeze_except_vera(model):
    for name, param in model.named_parameters():
        if "vera_lambda_b" in name or "vera_lambda_d" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

# Training loop
def train_model(model, dataloader, test_dataloader, optimizer, device,scheduler, num_epochs=10):
    model.to(device)
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    for ep in (pbar:=tqdm(range(num_epochs))):
        batch_losses = []
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = loss_fn(out, y)
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

            pbar.set_description(f"Lossy: {loss.item()}")
        
        epoch_losses.append(np.mean(batch_losses))
        if scheduler is not None:
            scheduler.step(ep)
        acc = test(model, test_dataloader)
        print(f"Epoch: {ep}, Accuracy: {acc}")
        #pbar.set_description(f"Accuracy: {str(acc*100)}")

    model = model.cpu()
    return model, epoch_losses

@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = []
    ex = []
    model = model.cuda()
    for batch in dl:
        x, y =  batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        out = torch.argmax(out, dim=1)
        correct = (out == y).float()
        acc.append(correct.sum())
        ex.append(len(y))
    ac = sum(acc) / sum(ex)
    return round(ac.cpu().item(), 4)

# Main script
def main():
#SBATCH --partition=A100medium
    # name = "cifar"
    # train_dl, test_dl = get_data(name)
    # num_classes = get_classes_num(name)
    dataset = 'Cifar100'
    num_classes = 0
    if dataset == 'Cifar100':
        num_classes = 100
    elif dataset == 'Food101':
        num_classes = 101
    elif dataset == 'RESISC45':
        num_classes = 45

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 100,10,1,0.1,0.01
    scale_array = [0.001]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for scale in scale_array:
        # Device setup

        # Load pretrained ViT model
        model = create_model("vit_base_patch16_224_in21k", pretrained=True, drop_path_rate=0.1)

        # for name, module in model.named_modules():
        #     print(name, type(module))

        # Change the classification head.
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
        for param in model.head.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in model.parameters())
        total_params = round(total_params / 1000000, 4)
        print(f"Total #Parameters: {total_params}M")
        
        # Apply VERA to the attention layers
        vera_config = VeraConfig()
        linear_out_dim, linear_in_dim = _find_dim(model)
        # print(linear_in_dim)
        # print(linear_out_dim)

        
        print(f"\n\n--------------------------------\n\n")
        print(f"scale is: {scale}\n")
        model = apply_vera_to_attention_Wavelet(model, vera_config, linear_out_dim, linear_in_dim, scale)

        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.data.shape}")

        # Freeze all parameters except lambda_b and lambda_d
        freeze_except_vera(model)
    
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params = round(trainable_params / 1000000, 4)
        
        print(f"Trainable #Parameters: {trainable_params}M")
        model.reset_classifier(num_classes)
        if dataset == 'Cifar100':
            # Dataset and DataLoader for CIFAR-100
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

            # Test dataset and dataloader
            test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        elif dataset == 'Food101':
            # Dataset and DataLoader for Food101
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resizing images to 224x224 for ViT input
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            # Training dataset and DataLoader
            train_dataset = datasets.Food101(root="./data", split="train", download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

            # Test dataset and DataLoader
            test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        elif dataset == 'RESISC45':
            num_classes = 45

            # Dataset and DataLoader for RESISC45
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            # Download RESISC45 from torchvision.datasets if supported, or manually provide paths to the dataset
            train_dataset = datasets.Food101(root="./data", split="train", download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

            test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

        
        # Define loss function and optimizer
        optimizer = torch.optim.AdamW(
            [
                {"params": model.head.parameters(), "lr": 4e-3},  # Higher learning rate for the classification head
                {"params": (p for n, p in model.named_parameters() if n.startswith("vera_lambda") and p.requires_grad)},
            ],
            lr=2e-2,  # Default learning rate for other layers
            weight_decay=0.01,
        )
      
        # Train the model
        # Linear learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,   # Start with the initial learning rate
        end_factor=0.01,    # Final learning rate as a fracxtion of the initial learning rate
        total_iters=1000   # Number of epochs for the linear decay
        )
      
        # scheduler = CosineLRScheduler(optimizer, t_initial=100)
        model, epoch_losses = train_model(model,train_loader, test_loader, optimizer, device, scheduler)

        
        # # Plot the loss convergence
        # plt.plot(range(len(epoch_losses1)), epoch_losses1, label="Kaiming Uniform")
        # plt.plot(range(len(epoch_losses1)), epoch_losses2, label="SVD")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("Loss Convergence Across Different Initialization Strategies for Shared Matrices")
        # plt.legend()
        # plt.show()
        # plt.savefig("/home/s33zganj/VERA1/convergence_plot.png")

        facc = test(model, test_loader)
        print(f"Final Accuracy: {facc*100}")

if __name__ == "__main__":
    main()
