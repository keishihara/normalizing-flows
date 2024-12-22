from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.distributions import SigmoidTransform, TransformedDistribution, Uniform

from nfs import DATASETS_2D
from nfs.datasets import FlowDataset
from nfs.flows import AffineConstantFlow, AffineHalfFlow
from nfs.models import NormalizingFlowModel
from nfs.utils import (
    compute_kl_divergence_kde,
    compute_kl_divergence_logistic,
    set_seed_everywhere,
)

ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="moons")
    parser.add_argument("--num-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed_everywhere(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset: {args.dataset}")
    dataset: FlowDataset = DATASETS_2D[args.dataset]()

    # Logistic distribution as prior
    prior = Uniform(low=torch.zeros(dataset.ndim).to(device), high=torch.ones(dataset.ndim).to(device))
    transform = SigmoidTransform().inv
    prior = TransformedDistribution(prior, [transform])

    # NICE
    flows = [AffineHalfFlow(dim=dataset.ndim, parity=i % 2, scale=False) for i in range(4)]
    flows.append(AffineConstantFlow(dim=dataset.ndim, shift=False))  # The last scaling layer

    model = NormalizingFlowModel(prior, flows)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training

    for global_step in range(args.num_steps):
        model.train()

        xs = dataset.sample(args.batch_size).to(device)
        if torch.isnan(xs).any() or torch.isinf(xs).any():
            print("Invalid values in input data, skipping this step")
            continue

        zs, prior_log_prob, log_det_jacobian = model(xs)
        log_prob = prior_log_prob + log_det_jacobian
        loss = -log_prob.sum()  # NLL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 1000 == 0 or global_step == args.num_steps - 1:
            # Compute KL-divergence between prior and zs
            model.eval()
            xs = dataset.sample(1024).to(device)
            with torch.inference_mode():
                ps = model.prior.sample(sample_shape=(1024,)).squeeze().cpu().numpy()
                zs = model(xs)[0][-1].cpu().numpy()
            kl_div = compute_kl_divergence_logistic(zs, ps)
            print(f"Step {global_step}, loss: {loss.item():.4f}, kl-div (prior & z): {kl_div:.4f}")

    print(f"Final loss: {loss.item():.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), CKPT_DIR / f"nice_{args.dataset}_step{args.num_steps}.pth")

    model.eval()

    # Plotting

    # Inference i.e. x -> z
    xs = dataset.sample(1024).to(device)
    with torch.inference_mode():
        ps = model.prior.sample(sample_shape=(1024,)).squeeze().cpu().numpy()
        zs, prior_log_prob, log_det_jacobian = model(xs)
    zs = zs[-1]
    xs = xs.detach().cpu().numpy()
    zs = zs.detach().cpu().numpy()
    kl_div = compute_kl_divergence_logistic(zs, ps)
    print(f"KL-divergence between prior and zs: {kl_div:.4f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(ps[:, 0], ps[:, 1], s=5, alpha=0.5, c="g", label="prior")
    plt.scatter(zs[:, 0], zs[:, 1], s=5, alpha=0.5, c="r", label="$x \\rightarrow z$")
    plt.scatter(xs[:, 0], xs[:, 1], s=5, alpha=0.5, c="b", label="data")
    plt.legend()
    plt.title(f"Inference $\\mathcal{{f}}: x \\rightarrow z$ (KL-div: {kl_div:.4f})")
    plt.axis("equal")

    # Sampling i.e. z -> x
    with torch.inference_mode():
        ys = model.sample(1024)
    ys = ys[-1].detach().cpu().numpy()
    kl_div = compute_kl_divergence_kde(xs, ys, bandwidth=0.2)
    print(f"KL-divergence between data and samples: {kl_div:.4f}")
    plt.subplot(122)
    plt.scatter(xs[:, 0], xs[:, 1], c="b", s=5, alpha=0.5, label="data")
    plt.scatter(ys[:, 0], ys[:, 1], c="r", s=5, alpha=0.5, label="$z \\rightarrow x$")
    plt.legend()
    plt.title(f"Sampling $\\mathcal{{f}}^{{-1}}: z \\rightarrow x$ (KL-div: {kl_div:.4f})")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"nice_{args.dataset}_step{args.num_steps}.png")


if __name__ == "__main__":
    main()
