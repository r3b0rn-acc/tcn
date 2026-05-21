from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.seasonal import STL


MAD_NORMAL_SCALE = 1.4826


@dataclass(frozen=True)
class PreprocessingConfig:
    filter_outliers: bool = False
    smooth: str | None = None
    stl: bool = False
    outlier_window: int = 24
    outlier_threshold: float = 3.5
    moving_average_window: int = 3
    exp_alpha: float = 0.3


@dataclass(frozen=True)
class STLAdjustment:
    adjusted: np.ndarray
    seasonal: np.ndarray


def causal_mad_outlier_filter(
    values: np.ndarray,
    window: int = 24,
    threshold: float = 3.5,
    min_history: int = 6,
) -> np.ndarray:
    """Replace outliers using only past observations.

    For point t the local median and MAD are computed from observations
    [t-window, ..., t-1]. The current value and future values are not used in
    the robust statistics, so the transformation is causal.
    """

    values = _as_float_array(values)
    filtered = values.copy()
    if window <= 0:
        raise ValueError("window must be positive")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    for idx in range(len(values)):
        start = max(0, idx - window)
        history = values[start:idx]
        history = history[np.isfinite(history)]
        if len(history) < min_history:
            continue

        median = float(np.median(history))
        mad = float(np.median(np.abs(history - median)))
        scale = MAD_NORMAL_SCALE * mad
        if scale <= np.finfo(float).eps:
            continue

        if abs(values[idx] - median) > threshold * scale:
            filtered[idx] = median
    return filtered


def causal_moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Causal moving average over the current and previous observations."""

    values = _as_float_array(values)
    if window <= 0:
        raise ValueError("window must be positive")

    smoothed = np.empty_like(values, dtype=float)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed[idx] = float(np.mean(values[start : idx + 1]))
    return smoothed


def exponential_smoothing(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Simple causal exponential smoothing."""

    values = _as_float_array(values)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if len(values) == 0:
        return values.copy()

    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def apply_preprocessing(values: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Apply causal filtering and smoothing before neural-network input."""

    if config.stl:
        raise ValueError("STL adjustment needs split_idx; use stl_seasonal_adjustment")

    processed = _as_float_array(values).copy()
    if config.filter_outliers:
        processed = causal_mad_outlier_filter(
            processed,
            window=config.outlier_window,
            threshold=config.outlier_threshold,
        )

    if config.smooth is None or config.smooth == "none":
        return processed
    if config.smooth == "moving_average":
        return causal_moving_average(processed, window=config.moving_average_window)
    if config.smooth == "exponential":
        return exponential_smoothing(processed, alpha=config.exp_alpha)
    raise ValueError(f"Unknown smoothing method: {config.smooth!r}")


def stl_seasonal_adjustment(
    values: np.ndarray,
    seasonality: int,
    fit_until: int,
    robust: bool = True,
) -> STLAdjustment:
    """Remove an STL seasonal component estimated only on the training slice."""

    values = _as_float_array(values)
    if seasonality <= 1:
        raise ValueError("seasonality must be greater than 1 for STL adjustment")
    if fit_until <= 0 or fit_until > len(values):
        raise ValueError("fit_until must be within the values length")

    if fit_until < seasonality * 2:
        seasonal = np.zeros_like(values, dtype=float)
        return STLAdjustment(adjusted=values.copy(), seasonal=seasonal)

    train_values = values[:fit_until]
    result = STL(train_values, period=seasonality, robust=robust).fit()
    seasonal_train = np.asarray(result.seasonal, dtype=float)
    seasonal = np.empty_like(values, dtype=float)
    seasonal[:fit_until] = seasonal_train

    profile = _seasonal_phase_profile(seasonal_train, seasonality)
    for idx in range(fit_until, len(values)):
        seasonal[idx] = profile[idx % seasonality]

    return STLAdjustment(adjusted=values - seasonal, seasonal=seasonal)


def _seasonal_phase_profile(seasonal: np.ndarray, seasonality: int) -> np.ndarray:
    profile = np.empty(seasonality, dtype=float)
    for phase in range(seasonality):
        phase_values = seasonal[phase::seasonality]
        profile[phase] = float(np.mean(phase_values)) if len(phase_values) else 0.0
    return profile


def _as_float_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    return values
