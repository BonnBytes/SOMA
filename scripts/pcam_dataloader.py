import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PCAMDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        """
        Args:
            x_data (np.array): Numpy array containing the image data (e.g., x_train).
            y_data (np.array): Numpy array containing the labels (e.g., y_train).
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.x_data[idx]).convert("RGB")
        label = int(self.y_data[idx, 0, 0, 0])

        if self.transform:
            image = self.transform(image)

        return image, label
