import torch
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
# from resisc_dataloader import RESISC45Dataset

# -------------------- Helper Functions --------------------
def _create_10_per_class_subset(dataset, sd, dname):
    # Extract targets if available
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # Extract labels manually if 'targets' attribute is missing
        fname = f"./targets_{dname}.pt"
        if os.path.exists(fname):
            print("Target file exists, loading pt file")
            targets = torch.load(fname)
        else:
            targets = [dataset[i][1] for i in tqdm(range(len(dataset)))]
            torch.save(targets, fname)

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Select 10 samples per class using the seed for reproducibility
    selected_indices = []
    print(f"Using dataset seed: {sd}")
    for label in class_indices:
        indices = class_indices[label]
        np.random.seed(sd)
        shuffled_indices = np.random.permutation(indices)
        selected_indices.extend(shuffled_indices[:10].tolist())

    return torch.utils.data.Subset(dataset, selected_indices)

def _split_resisc45(full_dataset, seed):
    """Splits RESISC45 into 10 samples per class for training and the rest for testing."""
    # Extract targets from the dataset
    fname = f"./targets_resisc.pt"
    if os.path.exists(fname):
        print("Target file exists, loading pt file")
        targets = torch.load(fname)
    else:
        targets = [full_dataset[i][1] for i in range(len(full_dataset))]

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Split each class into 10 training samples and remaining for testing
    train_indices = []
    test_indices = []
    for label in class_indices:
        indices = class_indices[label]
        np.random.seed(14)
        shuffled_indices = np.random.permutation(indices)
        train_indices.extend(shuffled_indices[:10].tolist())
        test_indices.extend(shuffled_indices[10:].tolist())

    # return (
    # torch.utils.data.Subset(full_dataset, train_indices),
    # torch.utils.data.Subset(full_dataset, test_indices)
    # )
    return train_indices, test_indices

def get_dl(dataset_type, seed):
    transform, test_transform = _get_transforms(dataset_type)

    if dataset_type == 'cifar':
        original_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        train_dataset = _create_10_per_class_subset(original_train, seed, "cifar")
        test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    elif dataset_type == 'food':
        original_train = datasets.Food101(root="./data", split="train", download=True, transform=transform)
        train_dataset = _create_10_per_class_subset(original_train, seed, "food")
        test_dataset = datasets.Food101(root="./data", split="test", download=True, transform=test_transform)
    elif dataset_type == 'resisc':
        # full_dataset = RESISC45Dataset(root_dir='./data/NWPU-RESISC45', transform=None)
        # train_indices, test_indices = _split_resisc45(full_dataset, seed)

        # train_dataset = torch.utils.data.Subset(
        # RESISC45Dataset(root_dir='./data/NWPU-RESISC45', transform=transform),
        # train_indices
        # )
        # test_dataset = torch.utils.data.Subset(
        # RESISC45Dataset(root_dir='./data/NWPU-RESISC45', transform=test_transform),
        # test_indices
        # )
        pass
    elif dataset_type == 'flowers':
        original_train = datasets.Flowers102(root="./data", split="train", download=True, transform=transform)
        train_dataset = _create_10_per_class_subset(original_train, seed, "flowers")
        test_dataset = datasets.Flowers102(root="./data", split="test", download=True, transform=test_transform)



    train_loader = _create_loader(train_dataset, drop=True, bs=64)
    test_loader = _create_loader(test_dataset, shuffle=False, bs=128)
    return train_loader, test_loader

def _get_transforms(dataset_type):
    transform = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform, test_transform


def _create_loader(dataset, shuffle=True, drop=False, bs=64):
    return torch.utils.data.DataLoader(
    dataset, batch_size=bs, shuffle=shuffle,
    drop_last=drop,
    num_workers=4, pin_memory=True)
