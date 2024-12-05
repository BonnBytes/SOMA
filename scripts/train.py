import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from timm import create_model
from timm.models.vision_transformer import VisionTransformer, Attention
from buffer_dict import BufferDict  # Assuming this is implemented
from layers import Linear as VeraLinear
from layers import MergedLinear as VeraMergedLinear
from torch.nn.init import _calculate_correct_fan
import math
from vtab import get_data, get_classes_num
from tqdm import tqdm
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
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

# Function to apply VERA modifications to the attention layers
def apply_vera_to_attention(model, vera_config, linear_out_dim, linear_in_dim):
    # print("yeeessssssssssssssssssssssss222222")
    generator = torch.Generator(device="cpu").manual_seed(0)
    vera_A = _kaiming_init((vera_config.r, linear_in_dim), generator)
    vera_B = _kaiming_init((linear_out_dim, vera_config.r), generator)

    
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
    for ep in (pbar:=tqdm(range(num_epochs))):
        for i, batch in enumerate(dataloader):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step(ep)
        acc = test(model, test_dataloader)
        print(f"Epoch: {ep}, Accuracy: {acc}")
        pbar.set_description(f"Accuracy: {str(acc*100)}")
    model = model.cpu()
    return model

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

    # name = "cifar"
    # train_dl, test_dl = get_data(name)
    # num_classes = get_classes_num(name)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ViT model
    model = create_model("vit_base_patch16_224_in21k", pretrained=True, drop_path_rate=0.1)
    # for name, module in model.named_modules():
    #     print(name, type(module))

    # Change the classification head.
    model.head = torch.nn.Linear(model.head.in_features, 100)
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
    model = apply_vera_to_attention(model, vera_config, linear_out_dim, linear_in_dim )

    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.data.shape}")

    # Freeze all parameters except lambda_b and lambda_d
    freeze_except_vera(model)

    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = round(trainable_params / 1000000, 4)
    
    print(f"Trainable #Parameters: {trainable_params}M")
    model.reset_classifier(100)
    # Dataset and DataLoader for CIFAR-100
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Test dataset and dataloader
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Define loss function and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.01)
    # Train the model
    # scheduler = None
    scheduler = CosineLRScheduler(optimizer, t_initial=100)
    model = train_model(model,train_loader, test_loader, optimizer, device, scheduler)
    facc = test(model, test_loader)
    print(f"Final Accuracy: {facc*100}")

if __name__ == "__main__":
    main()
