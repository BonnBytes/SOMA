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
import debugpy
from res_dataloader import RESISC45Dataset
# debugpy.listen(("127.0.0.1", 5678))  # Replace with remote machine IP and port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
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
def apply_vera_to_attention(model, vera_config, linear_out_dim, linear_in_dim, scale,rank):
    # print("yeeessssssssssssssssssssssss222222")
    # Check if linear_in_dim is a power of two
    new_in_dim = 0
    if (linear_in_dim & (linear_in_dim - 1)) != 0:  # Not a power of two
        new_in_dim = 2 ** math.ceil(math.log2(linear_in_dim))
    else:
        new_in_dim = linear_in_dim

    generator = torch.Generator(device="cpu").manual_seed(0)
    vera_A = _kaiming_init((rank, new_in_dim), generator)
    vera_B = _kaiming_init((linear_out_dim, rank), generator)
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
            r=rank,
            vera_dropout=vera_config.vera_dropout,
            scale = scale
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias if b else None
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod


    return model

# Function to apply VERA modifications to the attention layers
def apply_vera_to_attention_SVD(model, vera_config, linear_out_dim, linear_in_dim, scale, rank):

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
            r=rank,
            vera_dropout=vera_config.vera_dropout,
            scale = scale
            )

        new_mod.weight = mod.weight
        new_mod.bias = mod.bias if b else None
        model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod


    return model


# Function to apply VERA modifications to the attention layers
def apply_vera_to_attention_Wavelet(model, vera_config, linear_out_dim, linear_in_dim, scale, rank):
     #coeffs = pywt.dwt(torch.eye(768), 'sym4')
     #matrix_A = torch.tensor(coeffs[0])  # Approximation coefficients
     #matrix_B = torch.pinverse(matrix_A)  # Pseudo-inverse for reconstruction
    matrix_A = ptwt.matmul_transform.construct_boundary_a("sym4", length=768, device="cpu").to_dense()
    matrix_B = ptwt.matmul_transform.construct_boundary_s("sym4", length=768, device="cpu").to_dense()
    matrix_A[768//2:, :] = 0.0
    
    # print("Sum of matrix_A before permutation:", torch.sum(matrix_A).item())
    # print("Number of non-zero elements in matrix_A:", torch.count_nonzero(matrix_A).item())
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
            r=rank,
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
def apply_vera_to_attention_Fourier(model, vera_config, linear_out_dim, linear_in_dim, scale, rank):

    matrix_A = fourier_matrix(768).real
    matrix_B = fourier_matrix(768).real.T / 768
    matrix_A_copy = matrix_A.clone()
    matrix_A_copy[768 // 2:, :] = 0.0

    #matrix_A_copy[768 // 2:, :] = torch.complex(torch.zeros_like(matrix_A_copy[768 // 2:, :].real),
                                   #          torch.zeros_like(matrix_A_copy[768 // 2:, :].imag))
    
    # print(f"Is Complex matrix A: {torch.is_complex(matrix_A)}")
    # print(f"Is Complex matrix A copy: {torch.is_complex(matrix_A_copy)}")

    

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
            r=rank,
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


# def get_all_eigenvalues(model_state_dict, is_pretrained):

#     eigenvalues = []
#     layers = []
#     for name,module in model.named_modules():
#         print(name)
#         if "qkv" in name :
#             layers.append(name.split("."))
    
#     for l in layers:
        
#         nm = '.'.join(l)
#         mod = model.get_submodule(nm)
        
#         #     dim = mod.out_features
#         # else:
#         if is_pretrained == False:
#             # dim = mod.out_features 
#             weight = mod.weight.data.cpu().detach().numpy()

#             eigvals = np.linalg.eigvalsh(weight)
#             eigenvalues.append(eigvals)
#         else:
#             dim = mod.out_features // 3

#             weight_q, weight_k, weight_v = torch.split(mod.weight.data, dim, dim=0)
            
#             # # Get base weight matrix
#             # layer_weight = mod.weight.data

#             # Move to CPU and convert to numpy for eigenvalue calculation
#             for weight in [weight_q, weight_k, weight_v]:

#                 weight = weight.cpu().detach().numpy()

#                 eigvals = np.linalg.eigvalsh(weight)
#                 eigenvalues.append(eigvals)


#     return np.concatenate(eigenvalues)

def plot_eigen_hist(finetuned_eig, save_path="/home/s33zganj/VERA1/eigen_hist_wavelet.png"):
    """Plot comparison histogram of eigenvalues using Sturges' rule for bin selection."""
    plt.figure(figsize=(10, 6))

    # Calculate number of bins using Sturges' rule
    # num_bins_pretrained = int(np.ceil(np.log2(len(pretrained_eig)) + 1))
    bins = int(np.ceil(np.log2(len(finetuned_eig)) + 1))
    
    # Use the larger bin count to ensure consistency
    # bins = max(num_bins_pretrained, num_bins_finetuned)

    # plt.hist(pretrained_eig, bins=bins, alpha=0.2, label='Pretrained', density=True)
    plt.hist(finetuned_eig, bins=bins, alpha=0.5, label='Finetuned')

    plt.xlabel('Eigenvalue Magnitude')
    plt.ylabel('Frequency')
    plt.title('Eigenvalue Distribution of Finetuned Model')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

        
def get_all_eigenvalues(scale, checkpoint_path, is_pretrained = True):

    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    # Move tensors to the same device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # Print keys and shape of values
    for key, value in model_state_dict.items(): 
        print(f"Key: {key}, Shape: {value.shape}")
    for block_idx in range(12):
        print(model_state_dict[f'blocks.{block_idx}.attn.qkv.q_proj.vera_lambda_b'])

    eigenvalues = []
    qkv_weight = None

    for block_idx in range(12):
        attn_key = f'blocks.{block_idx}.attn.qkv'

        if is_pretrained == True:
            qkv_weight = model_state_dict[attn_key+ '.base_layer.weight'].to(device)

        elif is_pretrained == False:
                     
            if attn_key + '.weight' not in model_state_dict:
                print(f'layer {attn_key}.weight doesnt exists')
                continue
            
            # Get qkv weights
            # qkv_weight = model_state_dict[attn_key + '.weight']
            
            q_weight = None
            k_weight = None
            v_weight = None
            
            for name in ["q_proj", "v_proj"]:
                base_weight_key = attn_key + f'.{name}.base_layer.weight'
                
                if base_weight_key not in model_state_dict:
                    print(f'layer {base_weight_key} doesnt exists')
                    continue
                base_weight = model_state_dict[base_weight_key].to(device)

                lambda_b = model_state_dict[attn_key + f'.{name}.vera_lambda_b'].to(device)
                vera_B = model_state_dict[attn_key + f'.{name}.vera_B'].to(device)

                lambda_d = model_state_dict[attn_key + f'.{name}.vera_lambda_d'].to(device)
                
                vera_A = model_state_dict[attn_key + f'.{name}.vera_A'].to(device)

                if 'fixed_lambda_d' in model_state_dict.get(attn_key + f'.{name}.fixed_lambda_d', []):
                    fixed_lambda_d = model_state_dict[attn_key + f'.{name}.fixed_lambda_d'].to(device)
                    lambda_d = torch.cat([lambda_d, fixed_lambda_d], dim=0)


                
                # Compute adaptation: Λ_b * B * Λ_d * A
                scaled_B = vera_B * lambda_b.unsqueeze(1)  # Row scaling
                scaled_B = scaled_B * lambda_d.unsqueeze(0)  # Column scaling
                adaptation = scaled_B @ vera_A  # [768, 768]
                
                # Merge with base weights
                # model_state_dict[base_weight_key] = base_weight + adaptation
                if name == "q_proj":
                    q_weight = adaptation
                    k_weight = torch.zeros_like(q_weight, device=device)  
                elif name == "v_proj":
                    v_weight =adaptation

            
            qkv_delta_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            print(f'lambda_b:  {lambda_b}')
            print(f'vera_B:  {vera_B}')
            print(f'lambda_d:  {lambda_d}')
            print(f'vera_A:  {vera_A}')
            print("Sum of vera A :", torch.sum(vera_A).item())
            print("Number of non-zero elements in vera A:", torch.count_nonzero(vera_A).item())
            # model_state_dict[attn_key + '.weight'] = model_state_dict[attn_key + '.base_layer.weight'] + scale * qkv_delta_weight
            qkv_weight = model_state_dict[attn_key + '.base_layer.weight'].to(device) + scale * qkv_delta_weight


        
        output_dim = qkv_weight.shape[0] // 3
        q_weight, k_weight, v_weight = torch.split(qkv_weight, output_dim,  dim=0)
        for weight in [q_weight, k_weight, v_weight]:
            weight = weight.cpu().detach().numpy()
            eigvals = np.linalg.eigvalsh(weight)
            eigenvalues.append(eigvals)

    return np.concatenate(eigenvalues)

def save_checkpoint(checkpoint_path, model, optimizer, facc):
    # Saving model's checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': 10,
        'accuracy': facc

    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")       

# Main script
def main():
#SBATCH --partition=A100medium
    # name = "cifar"
    # train_dl, test_dl = get_data(name)
    # num_classes = get_classes_num(name)
    
    dataset_classes = {"Cifar100": 100, "Food101": 101, "RESISC45": 45}
    dataset = "Cifar100"
    num_classes = dataset_classes.get(dataset, 0)
    if not num_classes:
        raise ValueError(f"Unsupported dataset: {dataset}")

    seed =1
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    # 100,10,1,0.1,0.01
    scale_array = [0.01]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2,4,8,16,32,64,128,256,512
    rank = [256]


    for r in rank:
        svd_acc, wavelet_acc, fourier_acc, kaiming_acc = [], [], [], []
        for scale in scale_array:
            print(f"Training with scale: {scale}")
            
            # pre_trained_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/kaiming_vera_model.pth", True)
            
            # finetuned_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/kaiming_vera_model.pth", False)
            # plot_eigen_hist(pre_trained_eigenvalues,finetuned_eigenvalues)

            
            # kaiming Initialization
            model_kaiming = create_model("vit_base_patch16_224_in21k", pretrained=True, drop_path_rate=0.1)
            
            # pretrained_eig = get_all_eigenvalues(model_kaiming, True)
            
            model_kaiming.head = torch.nn.Linear(model_kaiming.head.in_features, num_classes)
            for param in model_kaiming.head.parameters():
                param.requires_grad = True
            
            total_params = sum(p.numel() for p in model_kaiming.parameters())
            total_params = round(total_params / 1000000, 4)
            print(f"Total #Parameters: {total_params}M")
            vera_config = VeraConfig()
            linear_out_dim, linear_in_dim = _find_dim(model_kaiming)
            model_kaiming = apply_vera_to_attention(model_kaiming, vera_config, linear_out_dim, linear_in_dim, scale, r)
            freeze_except_vera(model_kaiming)

            trainable_params = sum(p.numel() for p in model_kaiming.parameters() if p.requires_grad)
            trainable_params = round(trainable_params / 1000000, 4)
            print(f"Trainable #Parameters: {trainable_params}M")

            finetuned_model, optimizer, facc = train_and_evaluate(model_kaiming, dataset, device, num_classes,seed)
            kaiming_acc.append(facc)
            if scale == 0.01:
                save_checkpoint("/home/s33zganj/VERA1/checkpoints/kaiming_vera_model.pth", finetuned_model, optimizer, facc)
            
            # # finetuned_eig = get_all_eigenvalues(finetuned_model, False)
            # # plot_eigen_hist(pretrained_ei,finetuned_eig)

            # SVD Initialization
            print("SVD")
            model_svd = create_model("vit_base_patch16_224_in21k", pretrained=True, drop_path_rate=0.1)
            model_svd.head = torch.nn.Linear(model_svd.head.in_features, num_classes)
            for param in model_svd.head.parameters():
                param.requires_grad = True
            
            total_params = sum(p.numel() for p in model_svd.parameters())
            total_params = round(total_params / 1000000, 4)
            print(f"Total #Parameters: {total_params}M")
            vera_config = VeraConfig()
            linear_out_dim, linear_in_dim = _find_dim(model_svd)
            model_svd = apply_vera_to_attention_SVD(model_svd, vera_config, linear_out_dim, linear_in_dim, scale, r)
            freeze_except_vera(model_svd)

            trainable_params = sum(p.numel() for p in model_svd.parameters() if p.requires_grad)
            trainable_params = round(trainable_params / 1000000, 4)
            print(f"Trainable #Parameters: {trainable_params}M")

            finetuned_model, optimizer,facc = train_and_evaluate(model_svd, dataset, device, num_classes,seed)

            svd_acc.append(facc)

            if scale == 0.01:
                save_checkpoint(f"/home/s33zganj/VERA1/checkpoints/svd_vera_model.pth", finetuned_model, optimizer, facc)
            # pre_trained_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/svd_vera_model.pth", True)

            # finetuned_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/svd_vera_model.pth", False)
            # plot_eigen_hist(pre_trained_eigenvalues,finetuned_eigenvalues)
            

            # Wavelet Initialization
            print("Wavelet")
            model_wavelet = create_model("vit_base_patch16_224_in21k", pretrained=True, drop_path_rate=0.1)
            model_wavelet.head = torch.nn.Linear(model_wavelet.head.in_features, num_classes)
            for param in model_wavelet.head.parameters():
                param.requires_grad = True
            vera_config = VeraConfig()
            linear_out_dim, linear_in_dim = _find_dim(model_wavelet)
            model_wavelet = apply_vera_to_attention_Wavelet(model_wavelet, vera_config, linear_out_dim, linear_in_dim, scale, r)
            freeze_except_vera(model_wavelet)

            trainable_params = sum(p.numel() for p in model_wavelet.parameters() if p.requires_grad)
            trainable_params = round(trainable_params / 1000000, 4)
            print(f"Trainable #Parameters: {trainable_params}M")

            finetuned_model,optimizer, facc = train_and_evaluate(model_wavelet, dataset, device, num_classes,seed)
            wavelet_acc.append(facc)
            
            if scale == 0.01:
                save_checkpoint("/home/s33zganj/VERA1/checkpoints/wavelet_vera_model.pth", finetuned_model, optimizer, facc)
            # pre_trained_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/wavelet_vera_model.pth", True)

            # finetuned_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/wavelet_vera_model.pth", False)
            # plot_eigen_hist(finetuned_eigenvalues)

           
            # Fourier Initialization
            print("Fourier")
            model_fourier = create_model("vit_base_patch16_224_in21k", pretrained=True, drop_path_rate=0.1)
            model_fourier.head = torch.nn.Linear(model_fourier.head.in_features, num_classes)
            for param in model_fourier.head.parameters():
                param.requires_grad = True
            vera_config = VeraConfig()
            linear_out_dim, linear_in_dim = _find_dim(model_fourier)
            model_fourier = apply_vera_to_attention_Fourier(model_fourier, vera_config, linear_out_dim, linear_in_dim, scale, r)
            freeze_except_vera(model_fourier)

            trainable_params = sum(p.numel() for p in model_fourier.parameters() if p.requires_grad)
            trainable_params = round(trainable_params / 1000000, 4)
            print(f"Trainable #Parameters: {trainable_params}M")

            finetuned_model,optimizer, facc = train_and_evaluate(model_fourier, dataset, device, num_classes,seed)
            fourier_acc.append(facc)

            if scale == 0.01:
                save_checkpoint("/home/s33zganj/VERA1/checkpoints/fourier_vera_model.pth", finetuned_model, optimizer, facc)
            # pre_trained_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/fourier_vera_model.pth", True)
            # finetuned_eigenvalues = get_all_eigenvalues(scale,"/home/s33zganj/VERA1/checkpoints/fourier_vera_model.pth", False)
            # plot_eigen_hist(pre_trained_eigenvalues,finetuned_eigenvalues)


        # print(f"r:{r}")
        # print(f"kaiming: {kaiming_acc}")
        # print(f"svd: {svd_acc}")
        # print(f"wavelet: {wavelet_acc}")
        # print(f"fourier: {fourier_acc}")
        
       
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for acc, label in zip([svd_acc, wavelet_acc, fourier_acc], ["SVD", "Wavelet", "Fourier"]):
    #     plt.plot(scale_array, acc, label=label)
    # # plt.plot(scale_array, svd_acc, label='SVD')
    # # plt.plot(scale_array, wavelet_acc, label='Wavelet')
    # # plt.plot(scale_array, fourier_acc, label='Fourier')
    # plt.xlabel("Scale")
    # plt.ylabel("Final Accuracy")
    # plt.title("Comparison of Final Accuracies Across Different Initializations and Scales")
    # plt.legend()
    # plt.savefig("svd_final_accuracies.png")
    # plt.show()

    # k = [3.0764578605246973, 1.0027095282169254, 0.7898217181036787, 0.6785733015598793, 0.6189487880403105, 0.5638910320348587, 0.5270251410247953, 0.4727066457107076, 0.4436694593115947, 0.4144661062521296]
    # svd = [3.0838970436769375, 0.9858864544869383, 0.7745950173424638, 0.6680979913218504, 0.6188348691846194, 0.5466031939823084, 0.5396989724817514, 0.4772227857014923, 0.46063630696793406, 0.39959743593242497]
    # wav = [3.0568344847625477, 1.0276026250246693, 0.764103465277673, 0.6629967455742667, 0.6276995491698238, 0.5612173108882723, 0.527207456879168, 0.4947872884261012, 0.4547367881659878, 0.4365393567931664]
    # f= [3.094612689731676, 1.023395275625655, 0.7851022787230171, 0.6758109690196327, 0.624611132837477, 0.574745422292105, 0.5232566969265686, 0.46964792623166757, 0.4655607103308368, 0.4205849422672627]

    #     # Plotting
    # plt.figure(figsize=(10, 6))
    # for acc, label in zip([k, svd, wav, f], ["Kaiming","SVD", "Wavelet", "Fourier"]):
    #     plt.plot(range(len(k)), acc, label=label)
    # # plt.plot(scale_array, svd_acc, label='SVD')
    # # plt.plot(scale_array, wavelet_acc, label='Wavelet')
    # # plt.plot(scale_array, fourier_acc, label='Fourier')
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss Convergence Across Different Initialization Strategies for Shared Matrices")
    # plt.legend()
    # plt.savefig("Loss_convergence2.png")
    # plt.show()

    # # rank accuracy plot
    # # n_parameters = [0.0185,0.0185,0.0186,0.0188, 0.0192,0.02,0.0215,0.0246,0.0307]
    # k = [0.846,0.8502,0.8469,0.8439,0.8345,0.8485,0.842,0.848,0.8402]
    # svd = [0.8623,0.8649,0.8649,0.8633,0.8635,0.8623,0.8623,0.8612,0.863]
    # wav = [0.8666,0.8604,0.8595,0.8601,0.8612,0.8616,0.8594,0.8637,0.8643]
    # f = [0.8602,0.8603,0.8639,0.8676,0.8621,0.8639,0.8617,0.8625,0.8614]
    # # Plot the rank-accuracy curves
    # plt.figure(figsize=(8, 6))
    # plt.plot(rank, k, marker='o', linestyle='-', label='Kaiming Init')
    # plt.plot(rank, svd, marker='s', linestyle='-', label='SVD Init')
    # plt.plot(rank, wav, marker='^', linestyle='-', label='Wavelet Init')
    # plt.plot(rank, f, marker='d', linestyle='-', label='Fourier Init')

    # # Labels and title
    # plt.xlabel('Rank')
    # plt.ylabel('Accuracy')
    # plt.title('Rank-Accuracy Plot for Different Initializations')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("rank_accuracy.png")
    # plt.show()

def train_and_evaluate(model, dataset, device, num_classes,seed):
    model.reset_classifier(num_classes)
    # Dataset and Dataloader preparation
    if dataset == 'Cifar100':
        # Dataset and DataLoader for CIFAR-100
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,worker_init_fn=lambda _: np.random.seed(seed))

        # Test dataset and dataloader
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))
    
    elif dataset == 'Food101':
        # Dataset and DataLoader for Food101
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resizing images to 224x224 for ViT input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Training dataset and DataLoader
        train_dataset = datasets.Food101(root="./data", split="train", download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))

        # Test dataset and DataLoader
        test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))

    elif dataset == 'RESISC45':
        num_classes = 45
        
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

        dataset = RESISC45Dataset(root_dir='/home/s33zganj/VERA1/data/NWPU-RESISC45', transform=transform)

        # Split into training and validation sets (optional)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))
            
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": 4e-3}, 
         {"params": (p for n, p in model.named_parameters() if "vera_lambda" in n and p.requires_grad)}],
        lr=2e-2, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=1000)
    
    # Training
    model, epoch_losses = train_model(model, train_loader, test_loader, optimizer, device, scheduler)
    
    

    # # Plot the loss convergence
        # plt.plot(range(len(epoch_losses1)), epoch_losses1, label="Kaiming Uniform")
        # plt.plot(range(len(epoch_losses1)), epoch_losses2, label="SVD")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("Loss Convergence Across Different Initialization Strategies for Shared Matrices")
        # plt.legend()
        # plt.show()
        # plt.savefig("/home/s33zganj/VERA1/convergence_plot.png")
    # Evaluation

    facc = test(model, test_loader)
    print(f"Final Accuracy: {facc*100}")

    

    
    return model,optimizer, facc

if __name__ == "__main__":
    main()



