"""Temporal quantile transformer for hourly rental counts."""

from __future__ import annotations

import torch
from torch import nn


class QuantileTransformer(nn.Module):
    """Encoder over the last 24 hours. Predicts the median and a 90% upper bound.

    The last step is the hour being forecast. Its own rental count is masked.
    Weather and calendar for that hour stay visible, which matches day-ahead
    planning when a forecast for the hour is already known.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 24,
        d_model: int = 64,
        nhead: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        raw = self.head(self.norm(h[:, -1]))
        q50 = raw[:, 0]
        q90 = q50 + torch.nn.functional.softplus(raw[:, 1])
        return q50, q90


def pinball(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
    weight: torch.Tensor,
) -> torch.Tensor:
    error = target - pred
    loss = torch.maximum(quantile * error, (quantile - 1.0) * error)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
