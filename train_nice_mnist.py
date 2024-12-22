from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch import optim
from torch.distributions import SigmoidTransform, TransformedDistribution, Uniform

from nfs.datasets import get_mnist_dataloader
from nfs.flows import AffineConstantFlow, AffineHalfFlow
from nfs.models import NormalizingFlowModel
from nfs.utils import (
    compute_kl_divergence_logistic,
    set_seed_everywhere,
)

ASSETS_DIR = Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
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
    train_loader = get_mnist_dataloader("./data", args.batch_size, train=True, shuffle=True)
    eval_loader = get_mnist_dataloader("./data", args.batch_size)
    sample = train_loader.dataset[0]
    ndim = sample["image"].numel()
    print(f"Dataset shape: {sample['image'].shape}, ndim: {ndim}")
    # Logistic distribution as prior
    prior = Uniform(low=torch.zeros(ndim).to(device), high=torch.ones(ndim).to(device))
    transform = SigmoidTransform().inv
    prior = TransformedDistribution(prior, [transform])

    # NICE
    flows = [AffineHalfFlow(dim=ndim, parity=i % 2, scale=False) for i in range(4)]
    flows.append(AffineConstantFlow(dim=ndim, shift=False))  # The last scaling layer

    model = NormalizingFlowModel(prior, flows)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training

    steps_per_epoch = len(train_loader)
    epoch = 0
    dataiter = iter(train_loader)
    for global_step in range(args.num_steps):
        model.train()
        try:
            xs = next(dataiter)["image"].to(device)
        except StopIteration:
            dataiter = iter(train_loader)
            xs = next(dataiter)["image"].to(device)
            epoch += 1

        if torch.isnan(xs).any() or torch.isinf(xs).any():
            print("Invalid values in input data, skipping this step")
            continue

        xs = xs.view(xs.size(0), -1)
        zs, prior_log_prob, log_det_jacobian = model(xs)
        log_prob = prior_log_prob + log_det_jacobian
        loss = -log_prob.sum()  # NLL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % steps_per_epoch == 0 or global_step == args.num_steps - 1:
            # Compute KL-divergence between prior and zs
            model.eval()

            zs = []
            for batch in eval_loader:
                xs = batch["image"].view(batch["image"].size(0), -1).to(device)
                with torch.inference_mode():
                    zs.append(model(xs)[0][-1].cpu())
            zs = torch.cat(zs).numpy()
            ps = model.prior.sample(sample_shape=(len(zs),)).squeeze().cpu().numpy()
            kl_div = compute_kl_divergence_logistic(zs, ps)
            print(f"Epoch {epoch}, step {global_step}, loss: {loss.item():.4f}, kl-div (prior & z): {kl_div:.4f}")

    print(f"Final loss: {loss.item():.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), CKPT_DIR / f"nice_mnist_step{args.num_steps}.pth")


if __name__ == "__main__":
    main()
