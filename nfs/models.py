import torch
import torch.nn as nn
from torch.distributions import Distribution


class NormalizingFlow(nn.Module):
    def __init__(self, flows: list[nn.Module]):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m, *_ = x.shape
        log_det_jacobian = torch.zeros(m, device=x.device)
        z = x
        zs = [z]
        for flow in self.flows:
            z, ldj = flow(z)
            log_det_jacobian += ldj
            zs.append(z)
        return torch.stack(zs), log_det_jacobian

    def inverse(self, z: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        m, _ = z.shape
        log_det_jacobian = torch.zeros(m, device=z.device)
        x = z
        xs = [x]
        for flow in reversed(self.flows):
            x, ldj = flow.inverse(x)
            log_det_jacobian += ldj
            xs.append(x)
        return torch.stack(xs), log_det_jacobian


class NormalizingFlowModel(nn.Module):
    def __init__(self, prior: Distribution, flows: list[nn.Module]):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zs, log_det_jacobian = self.flow(x)
        prior_log_prob = self.prior.log_prob(zs[-1])
        prior_log_prob = prior_log_prob.to(x.device).view(x.size(0), -1).sum(1)
        return zs, prior_log_prob, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xs, log_det_jacobian = self.flow.inverse(z)
        return xs, log_det_jacobian

    def sample(self, n: int) -> torch.Tensor:
        z = self.prior.sample(sample_shape=(n,)).to(next(self.parameters()).device)
        xs, _ = self.flow.inverse(z)
        return xs
