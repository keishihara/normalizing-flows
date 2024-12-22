import torch
import torch.nn as nn


class AffineConstantFlow(nn.Module):
    """
    Affine constant flow.

    Scales and shifts the flow by learned constants per distribution.
    In NICE paper there is a scaling layer which is a special case of this where the shift is None.
    """

    def __init__(self, dim: int, shift: bool = True):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.randn(1, dim))
        self.shift = nn.Parameter(torch.randn(1, dim)) if shift else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale
        shift = self.shift if self.shift is not None else x.new_zeros(x.size())
        z = x * torch.exp(scale) + shift
        log_det_jacobian = torch.sum(scale, dim=1)
        return z, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale
        shift = self.shift if self.shift is not None else z.new_zeros(z.size())
        x = (z - shift) * torch.exp(-scale)
        log_det_jacobian = torch.sum(-scale, dim=1)
        return x, log_det_jacobian


class AffineHalfFlow(nn.Module):
    """
    Affine half flow.

    As seen in RealNVP paper, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transformed as a function of the other half.

    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim: int, parity: int, n_hidden: int = 24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        if scale:
            self.s_cond = nn.Sequential(
                nn.Linear(self.dim // 2, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, self.dim // 2),
            )
        else:
            self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if shift:
            self.t_cond = nn.Sequential(
                nn.Linear(self.dim // 2, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, self.dim // 2),
            )
        else:
            self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0  # untouched half
        z1 = x1 * torch.exp(s) + t  # transform this half a a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det_jacobian = torch.sum(s, dim=1)
        return z, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0  # untouched half
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det_jacobian = torch.sum(-s, dim=1)
        return x, log_det_jacobian
