"""
Plot the data distribution of a specific dataset.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

from nfs import DATASETS_2D
from nfs.datasets import FlowDataset

ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, choices=DATASETS_2D.keys(), required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    args = parser.parse_args()

    dataset: FlowDataset = DATASETS_2D[args.dataset]()
    x = dataset.sample(args.num_samples)

    plt.figure(figsize=(4, 4))
    plt.scatter(x[:, 0], x[:, 1], s=5, alpha=0.5)
    plt.axis("equal")
    plt.title(f"{args.dataset}: {args.num_samples:,} samples")

    path = ASSETS_DIR / f"{args.dataset}.png"
    plt.savefig(path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
