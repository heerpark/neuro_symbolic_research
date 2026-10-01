"""MNIST-Addition dataset construction (standard split, N-digit numbers).

Standard sizes: 60000 // (2*N) train instances, 10000 // (2*N) test instances,
each instance pairing two N-digit numbers (2*N raw MNIST images) and their sum.
Validation is 10% of the train instance pool, split off at the instance level
(so no raw image is ever shared between train/val/test).
"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets

MNIST_ROOT_DEFAULT = "./data/mnist_raw"


def load_mnist(root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (images[N,1,28,28] float in [0,1], labels[N] long)."""
    ds = datasets.MNIST(root=root, train=train, download=True)
    images = ds.data.float().div(255.0).unsqueeze(1)
    labels = ds.targets.long()
    return images, labels


def _make_addition_instances(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_digits: int,
    generator: torch.Generator,
):
    n = images.shape[0]
    n_instances = n // (2 * n_digits)
    perm = torch.randperm(n, generator=generator)[: n_instances * 2 * n_digits]
    idx = perm.view(n_instances, 2 * n_digits)
    op1_idx = idx[:, :n_digits]
    op2_idx = idx[:, n_digits:]

    images_op1 = images[op1_idx]
    images_op2 = images[op2_idx]
    digit_labels_op1 = labels[op1_idx]
    digit_labels_op2 = labels[op2_idx]

    weights = 10 ** torch.arange(n_digits - 1, -1, -1)
    num1 = (digit_labels_op1 * weights).sum(dim=1)
    num2 = (digit_labels_op2 * weights).sum(dim=1)
    sum_label = num1 + num2

    return images_op1, images_op2, digit_labels_op1, digit_labels_op2, sum_label


class MNISTAdditionDataset(Dataset):
    def __init__(self, images_op1, images_op2, digit_labels_op1, digit_labels_op2, sum_label):
        self.images_op1 = images_op1
        self.images_op2 = images_op2
        self.digit_labels_op1 = digit_labels_op1
        self.digit_labels_op2 = digit_labels_op2
        self.sum_label = sum_label

    def __len__(self):
        return self.images_op1.shape[0]

    def __getitem__(self, i):
        return {
            "images_op1": self.images_op1[i],
            "images_op2": self.images_op2[i],
            "digit_labels_op1": self.digit_labels_op1[i],
            "digit_labels_op2": self.digit_labels_op2[i],
            "sum_label": self.sum_label[i],
        }


def build_splits(
    n_digits: int,
    seed: int,
    root: str = MNIST_ROOT_DEFAULT,
    val_frac: float = 0.10,
) -> dict:
    """Returns {"train": MNISTAdditionDataset, "val": ..., "test": ...}, fully
    deterministic given (n_digits, seed, root, val_frac)."""
    generator = torch.Generator().manual_seed(seed)

    train_images, train_labels = load_mnist(root, train=True)
    test_images, test_labels = load_mnist(root, train=False)

    train_full = _make_addition_instances(train_images, train_labels, n_digits, generator)
    test_full = _make_addition_instances(test_images, test_labels, n_digits, generator)

    n_train_instances = train_full[0].shape[0]
    perm = torch.randperm(n_train_instances, generator=generator)
    n_val = int(round(n_train_instances * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def _subset(full, idx):
        return MNISTAdditionDataset(*(tensor[idx] for tensor in full))

    return {
        "train": _subset(train_full, train_idx),
        "val": _subset(train_full, val_idx),
        "test": MNISTAdditionDataset(*test_full),
    }
