import torch

def random_permutation_matrix(size):
    """Generate a random permutation matrix of given size."""
    perm = torch.randperm(size)
    I = torch.eye(size, dtype=torch.float64)
    return I[perm]


