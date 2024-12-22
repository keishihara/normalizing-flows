"""
Plot the step-wise flows of a specific model.
"""

import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import collections as mc
from torch.distributions import SigmoidTransform, TransformedDistribution, Uniform

from nfs import DATASETS_2D
from nfs.datasets import Dataset
from nfs.flows import AffineConstantFlow, AffineHalfFlow
from nfs.models import NormalizingFlowModel

ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename(filename: str) -> dict[str, str]:
    pattern = r"(?P<model_name>.+?)_(?P<dataset>.+?)_step(?P<n_steps>\d+)\.pth"
    match = re.match(pattern, Path(filename).name)
    if match:
        return match.groupdict()
    else:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    args.checkpoint = Path(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parsed = parse_filename(args.checkpoint)
    print(parsed)

    dataset_name = parsed["dataset"]
    dataset: Dataset = DATASETS_2D[dataset_name]()

    # Logistic distribution as prior
    prior = Uniform(low=torch.zeros(dataset.ndim).to(device), high=torch.ones(dataset.ndim).to(device))
    transform = SigmoidTransform().inv
    prior = TransformedDistribution(prior, [transform])

    if parsed["model_name"] == "nice":
        flows = [AffineHalfFlow(dim=dataset.ndim, parity=i % 2, scale=False) for i in range(4)]
        flows.append(AffineConstantFlow(dim=dataset.ndim, shift=False))  # The last scaling layer
    elif parsed["model_name"] == "rnvp":
        flows = [AffineHalfFlow(dim=dataset.ndim, parity=i % 2) for i in range(9)]
    else:
        raise ValueError(f"Unknown model name: {parsed['model_name']}")

    model = NormalizingFlowModel(prior, flows)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(device)
    model.eval()

    # Visualize the step-wise flow in the full net

    x = dataset.sample(128)

    # plot the coordinate warp
    ng = 20
    xx, yy = np.linspace(-3, 3, ng), np.linspace(-3, 3, ng)
    xv, yv = np.meshgrid(xx, yy)
    xy = np.stack([xv, yv], axis=-1)
    in_circle = np.sqrt((xy**2).sum(axis=2)) <= 3  # seems appropriate since we use radial distributions as priors
    xy = xy.reshape((ng * ng, 2))
    xy = torch.from_numpy(xy.astype(np.float32)).to(device)

    zs, log_det = model.inverse(xy)
    zs = zs.cpu()
    log_det = log_det.cpu()

    backward_flow_names = [type(f).__name__ for f in model.flow.flows[::-1]]
    nz = len(zs)

    _, axs = plt.subplots(nz - 1, 2, figsize=(6, 3 * (nz - 1)))

    for i in range(nz - 1):
        z0 = zs[i].detach().numpy()
        z1 = zs[i + 1].detach().numpy()

        # plot how the samples travel at this stage
        axs[i, 0].scatter(z0[:, 0], z0[:, 1], c="r", s=3, label=f"z{i}")
        axs[i, 0].scatter(z1[:, 0], z1[:, 1], c="b", s=3, label=f"z{i+1}")
        axs[i, 0].quiver(z0[:, 0], z0[:, 1], z1[:, 0] - z0[:, 0], z1[:, 1] - z0[:, 1], units="xy", scale=1, alpha=0.5)
        axs[i, 0].axis([-3, 3, -3, 3])
        axs[i, 0].set_title(f"layer {i} -> {i+1} ({backward_flow_names[i]})")
        axs[i, 0].legend()

        q = z1.reshape((ng, ng, 2))
        # y coords
        p1 = np.reshape(q[1:, :, :], (ng**2 - ng, 2))
        p2 = np.reshape(q[:-1, :, :], (ng**2 - ng, 2))
        inc = np.reshape(in_circle[1:, :] | in_circle[:-1, :], (ng**2 - ng,))
        p1, p2 = p1[inc], p2[inc]
        lcy = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.5, color="k")
        # x coords
        p1 = np.reshape(q[:, 1:, :], (ng**2 - ng, 2))
        p2 = np.reshape(q[:, :-1, :], (ng**2 - ng, 2))
        inc = np.reshape(in_circle[:, 1:] | in_circle[:, :-1], (ng**2 - ng,))
        p1, p2 = p1[inc], p2[inc]
        lcx = mc.LineCollection(zip(p1, p2), linewidths=1, alpha=0.5, color="k")
        # draw the lines
        axs[i, 1].add_collection(lcy)
        axs[i, 1].add_collection(lcx)
        axs[i, 1].axis([-3, 3, -3, 3])
        axs[i, 1].set_title(f"grid warp at the end of {i+1}")

        # draw the data too
        axs[i, 1].scatter(x[:, 0], x[:, 1], c="r", s=5, alpha=0.5, label="data")
        axs[i, 1].legend()

    plt.tight_layout()
    path = ASSETS_DIR / f"{args.checkpoint.stem}_step_wise_flows.png"
    plt.savefig(path)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
