import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from timm import create_model
from layers.merged import BaseMergedLinear as BaseMerged
from torch.nn.init import _calculate_correct_fan
import math
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from res_dataloader import RESISC45Dataset
from pcam_dataloader import PCAMDataset
import pywt
import ptwt
import gdown
import h5py

class VeraTrainer:
    def __init__(self, dataset="Cifar100", seed=1, device="cuda"):
        self.dataset_classes = {
            "Cifar100": 100, 
            "Food101": 101, 
            "RESISC45": 45,
            "Flowers102": 102,
            "PCAM": 2,
            "CelebA": 2
        }
        self.dataset = dataset
        self.num_classes = self.dataset_classes.get(dataset, 0)
        self.seed = seed
        self.device = device
        self._set_seeds()
        
    def _set_seeds(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------- Initialization Helpers --------------------
    @staticmethod
    def _kaiming_init(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        fan = _calculate_correct_fan(torch.empty(shape), "fan_in")
        gain = math.sqrt(2)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        return torch.empty(shape).uniform_(-bound, bound, generator=generator)

    @staticmethod
    def fourier_matrix(n: int) -> torch.Tensor:
        indices = torch.arange(n, dtype=torch.float64)
        real_part = torch.cos(-2 * torch.pi * torch.outer(indices, indices) / n)
        imag_part = torch.sin(-2 * torch.pi * torch.outer(indices, indices) / n)
        return torch.complex(real_part, imag_part)

    # -------------------- Model Creation --------------------

    def create_model(self, pre_trained_check_path):
        model = create_model("vit_base_patch16_224_in21k", pretrained=True, drop_path_rate=0.1)
        torch.save(model.state_dict(), pre_trained_check_path)
        # Modify model head and set requires_grad to True for the head parameters
        model.head = torch.nn.Linear(model.head.in_features, self.num_classes)
        self._param_numbers(model)
        out_dim, in_dim = self._find_dim(model)
        return model, in_dim, out_dim

    # -------------------- Model Modification --------------------

    @staticmethod
    def freeze_parameters(model):

        for name, param in model.named_parameters():
            if "vera_lambda_b" in name or "vera_lambda_d" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params = round(trainable_params / 1000000, 4)
        print(f"Trainable #Parameters: {trainable_params}M")
        
        for param in model.head.parameters():
            param.requires_grad = True

        return model

    def _modify_qkv_layers(self, model, linear_out_dim, linear_in_dim,
                         scale, rank, method, matrix_A=None, matrix_B=None):
        qkv_layers = []
        for name, _ in model.named_modules():
            if "qkv" in name:
                qkv_layers.append(name.split("."))

        for idx, (*parent, k, _, _) in enumerate(qkv_layers):
            nm = '.'.join(qkv_layers[idx])
            mod = model.get_submodule(nm)
            b = mod.bias is not None
            
            new_mod = BaseMerged(
                base_layer=model.get_submodule(".".join(parent))[int(k)].attn.qkv,
                weight=mod.weight.data,
                vera_A=matrix_A,
                vera_B=matrix_B,
                method = method,
                r=rank,
                scale=scale

            )
            new_mod.weight = mod.weight
            new_mod.bias = mod.bias if b else None
            model.get_submodule(".".join(parent))[int(k)].attn.qkv = new_mod
        
        return model

    def apply_vera_to_attention(self, model, linear_out_dim, linear_in_dim, scale, rank):
        generator = torch.Generator(device="cpu").manual_seed(0)
        vera_A = self._kaiming_init((rank, linear_in_dim), generator)
        vera_B = self._kaiming_init((linear_out_dim, rank), generator)
        return self._modify_qkv_layers(model, linear_out_dim, linear_in_dim,
                                     scale, rank, 'kaiming', vera_A, vera_B)

    def apply_vera_to_attention_SVD(self, model, linear_out_dim, linear_in_dim, scale, rank):
        return self._modify_qkv_layers(model, linear_out_dim, linear_in_dim,
                                      scale, rank, 'svd')

    def apply_vera_to_attention_Wavelet(self, model, linear_out_dim, linear_in_dim, scale, rank):
        matrix_A = ptwt.matmul_transform.construct_boundary_a("sym4", length=768, device="cpu").to_dense()
        matrix_B = ptwt.matmul_transform.construct_boundary_s("sym4", length=768, device="cpu").to_dense()
        matrix_A[768//2:, :] = 0.0
        return self._modify_qkv_layers(model, linear_out_dim, linear_in_dim,
                                      scale, rank, 'wavelet', matrix_A, matrix_B)

    def apply_vera_to_attention_Fourier(self, model, linear_out_dim, linear_in_dim, scale, rank):
        matrix_A = self.fourier_matrix(768)
        matrix_B = torch.conj(matrix_A.T) / 768
        matrix_A_copy = matrix_A.clone()
        matrix_A_copy[768 // 2:, :] = 0.0
        return self._modify_qkv_layers(model, linear_out_dim, linear_in_dim,
                                      scale, rank, 'fourier', matrix_A_copy, matrix_B)


    # -------------------- Training Core --------------------
    def train_and_evaluate(self, model, dataset_type, num_epochs=10):
        model.reset_classifier(self.num_classes)
        train_loader, test_loader = self._get_data_loaders(dataset_type)
        
        optimizer = self._get_optimizer(model)
        total_iters = num_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_iters
        )
        
        model, _ = self._train_model(model, train_loader, test_loader, optimizer, scheduler, num_epochs)
        facc = self.test(model, test_loader)
        return model, optimizer, facc

    # -------------------- Helper Functions --------------------
    def _get_data_loaders(self, dataset_type):
        transform = self._get_transforms(dataset_type)
        
        if dataset_type == 'Cifar100':
            train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        elif dataset_type == 'Food101':
            train_dataset = datasets.Food101(root="./data", split="train", download=True, transform=transform)
            test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=transform)
        elif dataset_type == 'RESISC45':
            full_dataset = RESISC45Dataset(root_dir='/home/s33zganj/VERA1/data/NWPU-RESISC45', transform=transform)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        elif dataset_type == 'Flowers102':
            train_dataset = datasets.Flowers102(root="./data", split="train", download=True, transform=transform)
            test_dataset = datasets.Flowers102(root="./data", split="test", download=True, transform=transform)
        elif dataset_type == 'PCAM':
            try:
                y_train = np.array(h5py.File('/home/s33zganj/VERA1/scripts/data/pcam/camelyonpatch_level_2_split_train_y.h5', 'r')['y'])
                x_train = np.array(h5py.File('/home/s33zganj/VERA1/scripts/data/pcam/camelyonpatch_level_2_split_train_x.h5', 'r')['x'])
                y_test = np.array(h5py.File('/home/s33zganj/VERA1/scripts/data/pcam/camelyonpatch_level_2_split_test_y.h5', 'r')['y'])
                x_test = np.array(h5py.File('/home/s33zganj/VERA1/scripts/data/pcam/camelyonpatch_level_2_split_test_x.h5', 'r')['x'])
            except OSError:
                raise NotImplementedError("Error while loading the dataset, make sure files are available.")

            train_dataset = PCAMDataset(x_train, y_train, transform=transform)
            test_dataset = PCAMDataset(x_test, y_test, transform=transform)

        elif dataset_type == 'CelebA':
            get_smile = lambda attr: attr[31]
            train_dataset = datasets.CelebA(root="/home/s33zganj/VERA1/scripts/data", split="train", target_type="attr", target_transform=get_smile, download=False, transform=transform)
            test_dataset = datasets.CelebA(root="/home/s33zganj/VERA1/scripts/data", split="test", target_type="attr", target_transform=get_smile, download=False, transform=transform)

        
        train_loader = self._create_loader(train_dataset)
        test_loader = self._create_loader(test_dataset, shuffle=False)
        return train_loader, test_loader

    def _get_transforms(self, dataset_type):
        if dataset_type in ['RESISC45', 'Flowers102']:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _create_loader(self, dataset, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=shuffle,
            worker_init_fn=lambda _: np.random.seed(self.seed)
        )

    def _get_optimizer(self, model):
        return torch.optim.AdamW(
            [
                {"params": model.head.parameters(), "lr": 4e-3},
                {"params": (p for n, p in model.named_parameters() if "vera_lambda" in n and p.requires_grad)}
            ],
            lr=2e-2, weight_decay=0.01
        )

    def _find_dim(self, model) -> tuple[int, int]:
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

    @staticmethod
    def _save_checkpoint(checkpoint_path, model, optimizer, facc):
        # Saving model's checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
            'accuracy': facc

        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


    def _get_all_eigenvalues(self, scale, checkpoint_path, is_pretrained = True):

        model_state_dict = None
        if is_pretrained == True:
            # Load checkpoint and model state dict
            model_state_dict = torch.load(checkpoint_path)

        else:
            # Load checkpoint and model state dict
            checkpoint = torch.load(checkpoint_path)
            model_state_dict = checkpoint['model_state_dict']

        # Move tensors to the same device
        device = "cuda" if torch.cuda.is_available() else "cpu"


        eigenvalues = []
        weights = []
        qkv_weight = None

        for block_idx in range(12):
            attn_key = f'blocks.{block_idx}.attn.qkv'

            if is_pretrained == True:
                qkv_weight = model_state_dict[attn_key+ '.weight'].to(device)

            elif is_pretrained == False:
                        
                if attn_key + '.weight' not in model_state_dict:
                    print(f'layer {attn_key}.weight doesnt exists')
                    continue
                
                # Get qkv weights       
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

                    fixed_lambda_d_key = attn_key + f'.{name}.fixed_lambda_d'
                    if fixed_lambda_d_key in model_state_dict:                        
                        fixed_lambda_d = model_state_dict[attn_key + f'.{name}.fixed_lambda_d'].to(device)
                        lambda_d = torch.cat([lambda_d, fixed_lambda_d], dim=0)

                    # Compute adaptation: Λ_b * B * Λ_d * A
                    scaled_B = vera_B * lambda_b.unsqueeze(1)  # Row scaling
                    scaled_B = scaled_B * lambda_d.unsqueeze(0)  # Column scaling
                    adaptation = scaled_B @ vera_A  # [768, 768]
                    
                    # Merge with base weights
                    if name == "q_proj":
                        q_weight = adaptation
                        k_weight = torch.zeros_like(q_weight, device=device)  
                    elif name == "v_proj":
                        v_weight =adaptation

                qkv_delta_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                qkv_weight = model_state_dict[attn_key + '.base_layer.weight'].to(device) + scale * qkv_delta_weight
            
            qkv_weight = qkv_weight.cpu().detach().numpy()
            weights.append(qkv_weight)
            singular_vals = np.linalg.svd(qkv_weight, compute_uv=False)
            eigenvalues.append(singular_vals)

        return np.concatenate(eigenvalues),  np.concatenate(weights)

    def eing_his(self, scale, checkpoint_path, pre_trained_eigenvalues, pre_trained_weights, his_path_eig, his_path_weight):
        finetuned_eigenvalues, finetuned_weights = self._get_all_eigenvalues(scale, checkpoint_path, False)
        self.plot_eigen_hist(pre_trained_eigenvalues, finetuned_eigenvalues, pre_trained_weights, finetuned_weights, his_path_eig, his_path_weight)
        return


    def _param_numbers(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        total_params = round(total_params / 1000000, 4)
        print(f"Total #Parameters: {total_params}M")
        return



    # -------------------- Training/Evaluation --------------------
    def _train_model(self, model, train_loader, test_loader, optimizer, scheduler, num_epochs=10):
        model.to(self.device)
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        epoch_losses = []
        
        for ep in (pbar := tqdm(range(num_epochs))):
            batch_losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                # # Only update weights if learning rate is not too small
                 #if optimizer.param_groups[0]['lr'] > 1e-6:
                    # optimizer.step()
                optimizer.step()
                batch_losses.append(loss.item())
                pbar.set_description(f"Loss: {loss.item():.4f}")
            
            epoch_losses.append(np.mean(batch_losses))
            if scheduler is not None:
                scheduler.step()
            acc = self.test(model, test_loader)
            print(f"Epoch {ep}: Accuracy={acc:.4f}")
        
        return model.cpu(), epoch_losses

    @torch.no_grad()
    def test(self, model, test_loader):
        model.eval().to(self.device)
        correct, total = 0, 0
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = torch.argmax(model(x), dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return round(correct / total, 4)

    # -------------------- Visualization --------------------
    @staticmethod
    def plot_accuracy_comparison(acc_dict, title="Accuracy Comparison", save_path="accuracy_comparison.png"):
        plt.figure(figsize=(10, 6))
        for label, acc in acc_dict.items():
            plt.plot(acc, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend()
        plt.savefig(save_path)
        plt.close()


    def plot_eigen_hist(self, pre_trained_eigenvalues, finetuned_eigenvalues, pre_trained_weights, finetuned_weights, save_path_eig ,save_path_weight, title="Eigenvalue Distribution"):
        plt.figure(figsize=(10, 6))

        # Compute bin edges using a finer resolution
        min_val = min(np.min(pre_trained_eigenvalues), np.min(finetuned_eigenvalues))
        max_val = max(np.max(pre_trained_eigenvalues), np.max(finetuned_eigenvalues))
        num_bins = max(
            int(np.ceil(np.log2(len(pre_trained_eigenvalues)) + 1)),
            int(np.ceil(np.log2(len(finetuned_eigenvalues)) + 1))
        )

        # Create finer bin edges
        bins = np.linspace(min_val, max_val, num_bins * 5)  # Increase bin resolution

        plt.hist(pre_trained_eigenvalues, bins=bins, alpha=0.3, label='Pretrained',
                density=True, edgecolor='blue', linewidth=1.0)

        plt.hist(finetuned_eigenvalues, bins=bins, alpha=0.8, label='Finetuned',
                density=True, edgecolor='red', linewidth=1.0)


        plt.xlabel('Eigenvalue Magnitude')
        plt.ylabel('Frequency')

        plt.title(title)
        plt.legend()
        plt.savefig(save_path_eig, dpi=300)  # Higher resolution for better clarity
        plt.close()

        plt.figure(figsize=(10, 6))

        # Compute bin edges using a finer resolution
        min_val = min(np.min(pre_trained_weights), np.min(finetuned_weights))
        max_val = max(np.max(pre_trained_weights), np.max(finetuned_weights))
        num_bins = max(
            int(np.ceil(np.log2(len(pre_trained_weights)) + 1)),
            int(np.ceil(np.log2(len(finetuned_weights)) + 1))
        )

        # Create finer bin edges
        bins = np.linspace(min_val, max_val, num_bins * 5)  # Increase bin resolution

        plt.hist(pre_trained_weights, bins=bins, alpha=0.3, label='Pretrained',
                density=True, edgecolor='blue', linewidth=1.0)

        plt.hist(finetuned_weights, bins=bins, alpha=0.8, label='Finetuned',
                density=True, edgecolor='red', linewidth=1.0)


        plt.xlabel('Weight Magnitude')
        plt.ylabel('Frequency')

        plt.title('Weight Distribution')
        plt.legend()
        plt.savefig(save_path_weight, dpi=300)  # Higher resolution for better clarity
        plt.close()


# Main Execution

if __name__ == "__main__":
    dataset_name = "CelebA"
    trainer = VeraTrainer(dataset=dataset_name, seed=1)
    scale_array = [0.01]
    rank = [256]
    results = {}

    for r in rank:
        for scale in scale_array:
            print(f"\nTraining with rank={r}, scale={scale}")
            pre_trained_check_path = f"/home/s33zganj/VERA1/checkpoints/vit_base_patch16_224_in21k_{dataset_name}.pth"


            ## Kaiming
            model, in_dim, out_dim = trainer.create_model(pre_trained_check_path)
            model = trainer.apply_vera_to_attention(model, out_dim, in_dim, scale, r)
            model = trainer.freeze_parameters(model)
            finetuned_model, optimizer, acc = trainer.train_and_evaluate(model, dataset_name)
            results.setdefault('Kaiming', []).append(acc)
            checkpoint_path = f"/home/s33zganj/VERA1/checkpoints/kaiming_vera_model_{dataset_name}.pth"
            trainer._save_checkpoint(checkpoint_path, finetuned_model, optimizer, acc)
            pretrained_eigenvalues, pretrained_weights = trainer._get_all_eigenvalues(scale, pre_trained_check_path, True)
            his_path_eig = f"/home/s33zganj/VERA1/histograms/eigenvalues/eigen_hist_kaiming_{dataset_name}.png"
            his_path_weight = f"/home/s33zganj/VERA1/histograms/weights/weight_hist_kaiming_{dataset_name}.png"
            trainer.eing_his(scale, checkpoint_path, pretrained_eigenvalues,pretrained_weights, his_path_eig, his_path_weight)

            ## SVD
            model, _, _ = trainer.create_model(pre_trained_check_path)
            model = trainer.apply_vera_to_attention_SVD(model, 768, 768, scale, r)
            model = trainer.freeze_parameters(model)
            finetuned_model, optimizer, acc = trainer.train_and_evaluate(model, dataset_name)
            results.setdefault('SVD', []).append(acc)
            checkpoint_path = f"/home/s33zganj/VERA1/checkpoints/svd_vera_model_{dataset_name}.pth"
            trainer._save_checkpoint(checkpoint_path, finetuned_model, optimizer, acc)
            pretrained_eigenvalues, pretrained_weights = trainer._get_all_eigenvalues(scale, pre_trained_check_path, True)
            his_path_eig = f"/home/s33zganj/VERA1/histograms/eigenvalues/eigen_hist_svd_{dataset_name}.png"
            his_path_weight = f"/home/s33zganj/VERA1/histograms/weights/weight_hist_svd_{dataset_name}.png"
            trainer.eing_his(scale, checkpoint_path, pretrained_eigenvalues, pretrained_weights, his_path_eig, his_path_weight)

            # Wavelet
            model, _, _ = trainer.create_model(pre_trained_check_path)
            model = trainer.apply_vera_to_attention_Wavelet(model, 768, 768, scale, r)
            model = trainer.freeze_parameters(model)
            finetuned_model, optimizer, acc = trainer.train_and_evaluate(model, dataset_name)
            results.setdefault('Wavelet', []).append(acc)
            checkpoint_path = f"/home/s33zganj/VERA1/checkpoints/wavelet_vera_model_{dataset_name}.pth"
            trainer._save_checkpoint(checkpoint_path, finetuned_model, optimizer, acc)
            pretrained_eigenvalues, pretrained_weights = trainer._get_all_eigenvalues(scale, pre_trained_check_path, True)
            his_path_eig = f"/home/s33zganj/VERA1/histograms/eigenvalues/eigen_hist_wavelet_{dataset_name}.png"
            his_path_weight = f"/home/s33zganj/VERA1/histograms/weights/weight_hist_wavelet_{dataset_name}.png"
            trainer.eing_his(scale, checkpoint_path, pretrained_eigenvalues, pretrained_weights, his_path_eig, his_path_weight)

            ## Fourier
            model, _, _ = trainer.create_model(pre_trained_check_path)
            model = trainer.apply_vera_to_attention_Fourier(model, 768, 768, scale, r)
            model = trainer.freeze_parameters(model)
            finetuned_model, optimizer, acc = trainer.train_and_evaluate(model, dataset_name)
            results.setdefault('Fourier', []).append(acc)
            checkpoint_path = f"/home/s33zganj/VERA1/checkpoints/fourier_vera_model_{dataset_name}.pth"
            trainer._save_checkpoint(checkpoint_path, finetuned_model, optimizer, acc)
            pretrained_eigenvalues, pretrained_weights = trainer._get_all_eigenvalues(scale, pre_trained_check_path, True)
            his_path_eig = f"/home/s33zganj/VERA1/histograms/eigenvalues/eigen_hist_fourier_{dataset_name}.png"
            his_path_weight = f"/home/s33zganj/VERA1/histograms/weights/weight_hist_fourier_{dataset_name}.png"
            trainer.eing_his(scale, checkpoint_path,  pretrained_eigenvalues, pretrained_weights, his_path_eig, his_path_weight)





    #VeraTrainer.plot_accuracy_comparison(results)
