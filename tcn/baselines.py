from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tcn.dataset import make_windows_for_origins


def forecast_naive(values: np.ndarray, horizons: Sequence[int], origins: Sequence[int]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray([[values[origin] for _ in horizons] for origin in origins], dtype=float)


def forecast_seasonal_naive(
    values: np.ndarray,
    horizons: Sequence[int],
    origins: Sequence[int],
    seasonality: int,
) -> np.ndarray:
    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    values = np.asarray(values, dtype=float)
    forecasts = np.empty((len(origins), len(horizons)), dtype=float)
    for row, origin in enumerate(origins):
        for col, horizon in enumerate(horizons):
            source = origin + horizon
            while source > origin:
                source -= seasonality
            if source < 0:
                source = origin
            forecasts[row, col] = values[source]
    return forecasts


def forecast_linear_autoregression(
    values: np.ndarray,
    horizons: Sequence[int],
    train_origins: Sequence[int],
    test_origins: Sequence[int],
    lookback: int,
) -> np.ndarray:
    x_train, y_train = make_windows_for_origins(values, values, train_origins, lookback, horizons)
    x_test, _ = make_windows_for_origins(values, values, test_origins, lookback, horizons)
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def forecast_arima(
    values: np.ndarray,
    horizons: Sequence[int],
    origins: Sequence[int],
    order: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    return _rolling_statsmodels_forecast(values, horizons, origins, "arima", order=order)


def forecast_sarima(
    values: np.ndarray,
    horizons: Sequence[int],
    origins: Sequence[int],
    seasonality: int,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    return _rolling_statsmodels_forecast(
        values,
        horizons,
        origins,
        "sarima",
        order=order,
        seasonal_order=seasonal_order + (seasonality,),
    )


def forecast_ets(
    values: np.ndarray,
    horizons: Sequence[int],
    origins: Sequence[int],
    seasonality: int,
) -> np.ndarray:
    return _rolling_statsmodels_forecast(values, horizons, origins, "ets", seasonality=seasonality)


def _rolling_statsmodels_forecast(
    values: np.ndarray,
    horizons: Sequence[int],
    origins: Sequence[int],
    model_type: str,
    **kwargs: object,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_horizon = max(horizons)
    forecasts = np.empty((len(origins), len(horizons)), dtype=float)
    warnings_to_ignore = (ConvergenceWarning, UserWarning, RuntimeWarning)

    for row, origin in enumerate(origins):
        history = values[: origin + 1]
        try:
            with warnings.catch_warnings():
                for warning_class in warnings_to_ignore:
                    warnings.simplefilter("ignore", warning_class)
                if model_type == "arima":
                    result = ARIMA(history, order=kwargs["order"]).fit()
                    full_forecast = np.asarray(result.forecast(steps=max_horizon), dtype=float)
                elif model_type == "sarima":
                    result = SARIMAX(
                        history,
                        order=kwargs["order"],
                        seasonal_order=kwargs["seasonal_order"],
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=50)
                    full_forecast = np.asarray(result.forecast(steps=max_horizon), dtype=float)
                elif model_type == "ets":
                    full_forecast = _fit_ets(history, int(kwargs["seasonality"]), max_horizon)
                else:
                    raise ValueError(f"Unknown model_type: {model_type!r}")
        except Exception:
            full_forecast = np.full(max_horizon, values[origin], dtype=float)
        forecasts[row] = [full_forecast[horizon - 1] for horizon in horizons]
    return forecasts


def _fit_ets(history: np.ndarray, seasonality: int, steps: int) -> np.ndarray:
    use_seasonal = seasonality > 1 and len(history) >= seasonality * 3
    try:
        model = ExponentialSmoothing(
            history,
            trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=seasonality if use_seasonal else None,
            initialization_method="estimated",
        )
        result = model.fit(optimized=True)
    except Exception:
        model = ExponentialSmoothing(history, trend="add", initialization_method="estimated")
        result = model.fit(optimized=True)
    return np.asarray(result.forecast(steps), dtype=float)
