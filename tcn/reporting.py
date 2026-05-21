from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from tcn.experiment import ExperimentResult


def build_predictions_frame(
    series: pd.Series,
    result: ExperimentResult,
    horizons: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = series.index
    for model_name, predictions in result.predictions.items():
        for origin_row, origin in enumerate(result.origins):
            for horizon_col, horizon in enumerate(horizons):
                target_idx = int(origin + horizon)
                rows.append(
                    {
                        "model": model_name,
                        "horizon": int(horizon),
                        "origin_index": int(origin),
                        "origin_date": index[int(origin)],
                        "target_index": target_idx,
                        "target_date": index[target_idx],
                        "actual": float(result.actuals[origin_row, horizon_col]),
                        "prediction": float(predictions[origin_row, horizon_col]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["horizon", "model", "target_date"], ignore_index=True)


def save_prediction_plots(
    series: pd.Series,
    predictions_frame: pd.DataFrame,
    horizons: Sequence[int],
    output_dir: str | Path,
    context_points: int = 24,
    metrics: pd.DataFrame | None = None,
    top_n_models: int = 2,
) -> list[Path]:
    if top_n_models <= 0:
        raise ValueError("top_n_models must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for horizon in horizons:
        horizon_frame = predictions_frame[predictions_frame["horizon"] == int(horizon)]
        if horizon_frame.empty:
            continue
        top_models = _top_models_for_horizon(horizon_frame, int(horizon), metrics, top_n_models)
        horizon_frame = horizon_frame[horizon_frame["model"].isin(top_models)]
        if horizon_frame.empty:
            continue

        start_idx = max(0, int(horizon_frame["origin_index"].min()) - context_points)
        end_idx = int(horizon_frame["target_index"].max())
        actual_window = series.iloc[start_idx : end_idx + 1]

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(
            actual_window.index,
            actual_window.values,
            color="black",
            linewidth=2.2,
            label="actual",
        )

        for model_name in top_models:
            model_frame = horizon_frame[horizon_frame["model"] == model_name]
            if model_frame.empty:
                continue
            model_frame = model_frame.sort_values("target_date")
            ax.plot(
                model_frame["target_date"],
                model_frame["prediction"],
                linewidth=1.4,
                alpha=0.85,
                label=str(model_name),
            )

        ax.axvline(
            series.index[int(horizon_frame["origin_index"].min())],
            color="0.55",
            linestyle="--",
            linewidth=1.0,
            label="first forecast origin",
        )
        ax.set_title(f"Actual vs Top {len(top_models)} Forecasts, horizon={horizon}")
        ax.set_xlabel("Date")
        ax.set_ylabel(series.name or "value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncols=2)
        fig.autofmt_xdate()
        fig.tight_layout()

        file_path = output_path / f"actual_vs_predictions_h{int(horizon)}.png"
        fig.savefig(file_path, dpi=160)
        plt.close(fig)
        saved_paths.append(file_path)

    return saved_paths


def _top_models_for_horizon(
    horizon_frame: pd.DataFrame,
    horizon: int,
    metrics: pd.DataFrame | None,
    top_n_models: int,
) -> list[str]:
    if metrics is not None and not metrics.empty:
        horizon_values = metrics["horizon"]
        metric_rows = metrics[(horizon_values == horizon) | (horizon_values.astype(str) == str(horizon))]
        metric_rows = metric_rows[metric_rows["model"].isin(horizon_frame["model"].unique())]
        sort_columns = [column for column in ("MASE", "RMSE", "MAE") if column in metric_rows.columns]
        if not metric_rows.empty and sort_columns:
            return list(metric_rows.sort_values(sort_columns).head(top_n_models)["model"].astype(str))

    errors = horizon_frame.assign(abs_error=(horizon_frame["actual"] - horizon_frame["prediction"]).abs())
    ranking = errors.groupby("model", sort=True)["abs_error"].mean().sort_values()
    return list(ranking.head(top_n_models).index.astype(str))
