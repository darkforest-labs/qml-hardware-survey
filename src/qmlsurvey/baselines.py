"""
Classical baselines matched on parameter count to the hybrid model. Every
RunRecord includes a baseline score so quantum results can never be reported
without context.
"""
from __future__ import annotations

import torch
from torch import nn


class MatchedMLP(nn.Module):
    """Plain MLP with approximately the same parameter count as `target_params`."""

    def __init__(self, in_features: int, out_features: int, target_params: int):
        super().__init__()
        # Solve for hidden so that (in*h + h) + (h*out + out) ≈ target_params.
        # h * (in + out + 1) ≈ target_params - out
        denom = max(1, in_features + out_features + 1)
        hidden = max(2, (target_params - out_features) // denom)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
