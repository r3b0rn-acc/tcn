from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_values: np.ndarray,
    seasonality: int = 1,
) -> np.ndarray:
    """Mean absolute scaled error with seasonal naive in-sample scaling."""

    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    train_values = np.asarray(train_values, dtype=float)
    if len(train_values) <= seasonality:
        raise ValueError("not enough train values for MASE denominator")
    denominator = float(np.mean(np.abs(train_values[seasonality:] - train_values[:-seasonality])))
    if denominator <= np.finfo(float).eps:
        denominator = 1.0
    return mae(y_true, y_pred) / denominator


def metric_rows(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_values: np.ndarray,
    horizons: Sequence[int],
    seasonality: int,
) -> list[dict[str, float | int | str]]:
    mae_values = mae(y_true, y_pred)
    rmse_values = rmse(y_true, y_pred)
    mase_values = mase(y_true, y_pred, train_values=train_values, seasonality=seasonality)

    rows: list[dict[str, float | int | str]] = []
    for idx, horizon in enumerate(horizons):
        rows.append(
            {
                "model": model_name,
                "horizon": int(horizon),
                "MAE": float(mae_values[idx]),
                "RMSE": float(rmse_values[idx]),
                "MASE": float(mase_values[idx]),
            }
        )

    rows.append(
        {
            "model": model_name,
            "horizon": "mean",
            "MAE": float(np.mean(mae_values)),
            "RMSE": float(np.mean(rmse_values)),
            "MASE": float(np.mean(mase_values)),
        }
    )
    return rows


def build_metrics_table(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["horizon", "MASE", "RMSE"], ignore_index=True)
