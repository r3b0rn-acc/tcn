from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StandardScaler1D:
    mean_: float
    scale_: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler1D":
        values = np.asarray(values, dtype=float)
        scale = float(np.std(values))
        if scale <= np.finfo(float).eps:
            scale = 1.0
        return cls(mean_=float(np.mean(values)), scale_=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale_ + self.mean_


def parse_horizons(text: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    validate_horizons(horizons)
    return horizons


def validate_horizons(horizons: Sequence[int]) -> None:
    if not horizons:
        raise ValueError("at least one horizon is required")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")


def make_origins(
    n_observations: int,
    split_idx: int,
    lookback: int,
    horizons: Sequence[int],
    max_test_origins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create chronological train and rolling-origin test indices."""

    validate_horizons(horizons)
    max_horizon = max(horizons)
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if split_idx <= lookback + max_horizon:
        raise ValueError("training split is too short for the requested lookback/horizon")
    if n_observations - split_idx <= max_horizon:
        raise ValueError("test split is too short for the requested horizon")

    train_origins = np.arange(lookback - 1, split_idx - max_horizon, dtype=int)
    test_origins = np.arange(split_idx - 1, n_observations - max_horizon, dtype=int)
    if max_test_origins is not None and max_test_origins > 0:
        test_origins = test_origins[:max_test_origins]
    return train_origins, test_origins


def make_windows_for_origins(
    input_values: np.ndarray,
    target_values: np.ndarray,
    origins: Sequence[int],
    lookback: int,
    horizons: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised windows where X uses input history and y uses raw targets."""

    input_values = np.asarray(input_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    origins = np.asarray(origins, dtype=int)
    validate_horizons(horizons)

    if input_values.ndim != 1 or target_values.ndim != 1:
        raise ValueError("input_values and target_values must be one-dimensional")
    if len(input_values) != len(target_values):
        raise ValueError("input_values and target_values must have equal length")

    x = np.empty((len(origins), lookback), dtype=float)
    y = np.empty((len(origins), len(horizons)), dtype=float)
    for row, origin in enumerate(origins):
        start = origin - lookback + 1
        if start < 0:
            raise ValueError("origin is too early for lookback")
        x[row] = input_values[start : origin + 1]
        y[row] = [target_values[origin + horizon] for horizon in horizons]
    return x, y
