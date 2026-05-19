from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class CausalConv1d(nn.Module):
    """One-dimensional convolution that never reads future positions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.downsample(x))


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network for direct multi-horizon forecasting."""

    def __init__(
        self,
        horizon_count: int,
        input_channels: int = 1,
        channels: tuple[int, ...] = (32, 32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if horizon_count <= 0:
            raise ValueError("horizon_count must be positive")
        if kernel_size <= 1:
            raise ValueError("kernel_size must be greater than 1")

        layers: list[nn.Module] = []
        in_channels = input_channels
        for block_idx, out_channels in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_channels, horizon_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [batch, lookback]")
        features = self.tcn(x.unsqueeze(1))
        last_features = features[:, :, -1]
        return self.head(last_features)
