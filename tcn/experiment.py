from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tcn.baselines import (
    forecast_arima,
    forecast_ets,
    forecast_naive,
    forecast_sarima,
)
from tcn.dataset import make_origins, make_windows_for_origins
from tcn.metrics import build_metrics_table, metric_rows
from tcn.preprocessing import (
    PreprocessingConfig,
    apply_preprocessing,
    stl_seasonal_adjustment,
)
from tcn.train import TCNForecastResult, TCNTrainingConfig, train_tcn_forecaster


@dataclass(frozen=True)
class ExperimentConfig:
    lookback: int = 36
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    test_size: float = 0.2
    seasonality: int = 12
    outlier_window: int = 24
    outlier_threshold: float = 3.5
    moving_average_window: int = 3
    exp_alpha: float = 0.3
    max_test_origins: int | None = 60
    include_statistical_baselines: bool = True
    tcn: TCNTrainingConfig = TCNTrainingConfig()


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    actuals: np.ndarray
    origins: np.ndarray
    tcn_results: dict[str, TCNForecastResult]
    split_idx: int


def run_experiment(series: pd.Series, config: ExperimentConfig) -> ExperimentResult:
    values = np.asarray(series.values, dtype=float)
    split_idx = int(len(values) * (1.0 - config.test_size))
    train_origins, test_origins = make_origins(
        n_observations=len(values),
        split_idx=split_idx,
        lookback=config.lookback,
        horizons=config.horizons,
        max_test_origins=config.max_test_origins,
    )
    _, actuals = make_windows_for_origins(values, values, test_origins, config.lookback, config.horizons)

    predictions: dict[str, np.ndarray] = {}
    tcn_results: dict[str, TCNForecastResult] = {}
    rows: list[dict[str, float | int | str]] = []
    train_values = values[:split_idx]

    for model_name, preprocessing in _tcn_variants(config).items():
        target_values = values
        seasonal_adjustment = None
        if preprocessing.stl:
            seasonal_adjustment = stl_seasonal_adjustment(
                values,
                seasonality=config.seasonality,
                fit_until=split_idx,
            )
            processed_inputs = seasonal_adjustment.adjusted
            target_values = seasonal_adjustment.adjusted
        else:
            processed_inputs = apply_preprocessing(values, preprocessing)

        tcn_config = _with_experiment_shape(config.tcn, config.lookback, config.horizons)
        result = train_tcn_forecaster(
            model_name=model_name,
            input_values=processed_inputs,
            target_values=target_values,
            split_idx=split_idx,
            train_origins=train_origins,
            test_origins=test_origins,
            config=tcn_config,
        )
        model_predictions = result.predictions
        if seasonal_adjustment is not None:
            model_predictions = model_predictions + _seasonal_targets(
                seasonal_adjustment.seasonal,
                test_origins,
                config.horizons,
            )
            result.predictions = model_predictions
            result.actuals = actuals

        predictions[model_name] = model_predictions
        tcn_results[model_name] = result
        rows.extend(
            metric_rows(
                model_name,
                actuals,
                model_predictions,
                train_values=train_values,
                horizons=config.horizons,
                seasonality=config.seasonality,
            )
        )

    baseline_predictions = _run_baselines(values, config, train_origins, test_origins)
    for model_name, forecast in baseline_predictions.items():
        predictions[model_name] = forecast
        rows.extend(
            metric_rows(
                model_name,
                actuals,
                forecast,
                train_values=train_values,
                horizons=config.horizons,
                seasonality=config.seasonality,
            )
        )

    return ExperimentResult(
        metrics=build_metrics_table(rows),
        predictions=predictions,
        actuals=actuals,
        origins=test_origins,
        tcn_results=tcn_results,
        split_idx=split_idx,
    )


def _tcn_variants(config: ExperimentConfig) -> dict[str, PreprocessingConfig]:
    base = dict(
        outlier_window=config.outlier_window,
        outlier_threshold=config.outlier_threshold,
        moving_average_window=config.moving_average_window,
        exp_alpha=config.exp_alpha,
    )
    return {
        "Raw": PreprocessingConfig(filter_outliers=False, smooth=None, **base),
        "Filter": PreprocessingConfig(filter_outliers=True, smooth=None, **base),
        "Smooth-MA": PreprocessingConfig(filter_outliers=False, smooth="moving_average", **base),
        "Smooth-ES": PreprocessingConfig(filter_outliers=False, smooth="exponential", **base),
        "Filter+MA": PreprocessingConfig(filter_outliers=True, smooth="moving_average", **base),
        "Filter+ES": PreprocessingConfig(filter_outliers=True, smooth="exponential", **base),
        "STL+TCN": PreprocessingConfig(stl=True, **base),
    }


def _run_baselines(
    values: np.ndarray,
    config: ExperimentConfig,
    train_origins: np.ndarray,
    test_origins: np.ndarray,
) -> dict[str, np.ndarray]:
    forecasts = {
        "naive": forecast_naive(values, config.horizons, test_origins),
    }
    if config.include_statistical_baselines:
        forecasts["arima"] = forecast_arima(values, config.horizons, test_origins)
        forecasts["sarima"] = forecast_sarima(
            values,
            config.horizons,
            test_origins,
            seasonality=config.seasonality,
        )
        forecasts["ets"] = forecast_ets(values, config.horizons, test_origins, seasonality=config.seasonality)
    return forecasts


def _seasonal_targets(
    seasonal: np.ndarray,
    origins: np.ndarray,
    horizons: tuple[int, ...],
) -> np.ndarray:
    return np.asarray(
        [[seasonal[int(origin) + horizon] for horizon in horizons] for origin in origins],
        dtype=float,
    )


def _with_experiment_shape(
    config: TCNTrainingConfig,
    lookback: int,
    horizons: tuple[int, ...],
) -> TCNTrainingConfig:
    return TCNTrainingConfig(
        lookback=lookback,
        horizons=horizons,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        validation_fraction=config.validation_fraction,
        patience=config.patience,
        channels=config.channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        gradient_clip_norm=config.gradient_clip_norm,
        device=config.device,
        require_gpu=config.require_gpu,
        seed=config.seed,
    )
