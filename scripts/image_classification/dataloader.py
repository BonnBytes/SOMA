"""ViT Dataloader."""

import os
from collections import defaultdict
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# from resisc_dataloader import RESISC45Dataset


def _create_10_per_class_subset(
    dataset: Dataset, sd: int, dname: str
) -> Dataset:
    """Compute 10 samples per class.

    Args:
        dataset (Dataset): Dataset object.
        sd (int): Seed.
        dname (str): Dataset name.

    Returns:
        Dataset: Subset dataset.
    """
    # Extract targets if available
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    else:
        # Extract labels manually if 'targets' attribute is missing
        fname = f"./data/targets_{dname}.pt"
        if os.path.exists(fname):
            print("Target file exists, loading pt file")
            targets = torch.load(fname)
        else:
            targets = [dataset[i][1] for i in tqdm(range(len(dataset)))]  # type: ignore
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


def _split_resisc45(full_dataset: Dataset, seed: int) -> Tuple[Any, Any]:
    """Split RESISC for train and testing.

    Args:
        full_dataset (Dataset): RESISC dataset.
        seed (int): Seed.

    Returns:
        Tuple[Any, Any]: Tuple containing train and test indices.
    """
    # Extract targets from the dataset
    fname = "./targets_resisc.pt"
    if os.path.exists(fname):
        print("Target file exists, loading pt file")
        targets = torch.load(fname)
    else:
        targets = [full_dataset[i][1] for i in range(len(full_dataset))]  # type: ignore

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


def get_dl(dataset_type: str, seed: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepare dataloaders.

    Args:
        dataset_type (str): Dataset name.
        seed (int): Seed.
        batch_size (int): Batch size.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple containing train and test dataloaders.
    """
    transform, test_transform = _get_transforms()

    if dataset_type == "cifar":
        original_train = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        train_dataset = _create_10_per_class_subset(
            original_train, seed, "cifar"
        )
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=test_transform
        )
    elif dataset_type == "food":
        original_train = datasets.Food101(
            root="./data", split="train", download=True, transform=transform
        )
        train_dataset = _create_10_per_class_subset(
            original_train, seed, "food"
        )
        test_dataset = datasets.Food101(
            root="./data",
            split="test",
            download=True,
            transform=test_transform,
        )
    elif dataset_type == "resisc":
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
    elif dataset_type == "flowers":
        original_train = datasets.Flowers102(
            root="./data", split="train", download=True, transform=transform
        )
        train_dataset = _create_10_per_class_subset(
            original_train, seed, "flowers"
        )
        test_dataset = datasets.Flowers102(
            root="./data",
            split="test",
            download=True,
            transform=test_transform,
        )

    train_loader = _create_loader(train_dataset, drop=False, bs=batch_size)
    test_loader = _create_loader(test_dataset, shuffle=False, bs=2048)
    return train_loader, test_loader


def _get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Prepare data transformations.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Tuple containing train and test transforms.
    """
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=3),
            transforms.RandAugment(num_ops=2, magnitude=5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            # transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
        ]
    )
    return transform, test_transform


def _create_loader(
    dataset: Dataset,
    shuffle: Optional[bool] = True,
    drop: Optional[bool] = False,
    bs: Optional[int] = 64,
) -> DataLoader:
    """Create dataloader object.

    Args:
        dataset (Dataset): Dataset object.
        shuffle (Optional[bool], optional): Shuffle. Defaults to True.
        drop (Optional[bool], optional): Drop last. Defaults to False.
        bs (Optional[int], optional): Batch size. Defaults to 64.

    Returns:
        DataLoader: _description_
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=drop,
        num_workers=4,
        pin_memory=True,
    )
