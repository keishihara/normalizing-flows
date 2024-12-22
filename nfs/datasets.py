import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from sklearn import datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor


class FlowDataset:
    """Base class for datasets"""

    ndim: int = 2

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError


class DatasetMoons(FlowDataset):
    """Two half-moons"""

    def sample(self, n: int) -> torch.Tensor:
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)


class DatasetMixture(FlowDataset):
    """4 mixture of gaussians"""

    def sample(self, n: int) -> torch.Tensor:
        assert n % 4 == 0
        r = np.r_[
            np.random.randn(n // 4, 2) * 0.5 + np.array([0, -2]),
            np.random.randn(n // 4, 2) * 0.5 + np.array([0, 0]),
            np.random.randn(n // 4, 2) * 0.5 + np.array([2, 2]),
            np.random.randn(n // 4, 2) * 0.5 + np.array([-2, 2]),
        ]
        return torch.from_numpy(r.astype(np.float32))


class DatasetSIGGRAPH(FlowDataset):
    """
    Created by Eric from https://blog.evjang.com/2018/01/nf2.html
    Source: https://github.com/ericjang/normalizing-flows-tutorial/blob/master/siggraph.pkl
    """

    def __init__(self):
        with open(Path(__file__).parents[1] / "data" / "siggraph.pkl", "rb") as f:
            XY = np.array(pickle.load(f), dtype=np.float32)
            XY -= np.mean(XY, axis=0)  # center
        self.XY = torch.from_numpy(XY)

    def sample(self, n: int) -> torch.Tensor:
        X = self.XY[np.random.randint(self.XY.shape[0], size=n)]
        return X


class FlowMNIST(Dataset):
    """Simple MNIST wrapper.

    Args:
        root: Where to save dataset.
        train: Whether it is train or valid dataset. Defaults to True.
        transform: Image transforms. Defaults to None.
    """

    def __init__(self, root: str, train: bool = True, transform: Callable | None = None):
        """Inits MNIST dataset wrapper."""
        self.dataset = MNIST(root=root, train=train, transform=transform, download=True)

    def __len__(self) -> int:
        """Returns dataset length.

        Returns:
            Dataset length.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Returns MNIST dataset sample.

        Args:
            index: Sample index.

        Returns:
            Dict with sampled image.
        """
        image, _ = self.dataset[index]

        return {"image": image}


def get_mnist_dataloader(
    root: str = "./data",
    batch_size: int = 64,
    train: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """Returns MNIST dataloader, prepared for generative model training.

    Args:
        root: Where to save dataset. Defaults to "./data".
        batch_size: Batch size. Defaults to 64.

    Returns:
        MNIST dataloader.
    """
    transforms = Compose(
        [
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1),  # Normalize to [-1, 1]
        ]
    )

    dataset = FlowMNIST(root=root, train=train, transform=transforms)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=True)

    return dataloader
