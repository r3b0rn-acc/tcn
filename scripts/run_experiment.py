from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from tcn.data import load_fred_series
from tcn.dataset import parse_horizons
from tcn.experiment import ExperimentConfig, run_experiment
from tcn.train import TCNTrainingConfig


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    channels = tuple(int(part.strip()) for part in args.channels.split(",") if part.strip())

    series = load_fred_series(
        series_id=args.fred_series,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=not args.no_cache,
    )

    tcn_config = TCNTrainingConfig(
        lookback=args.lookback,
        horizons=horizons,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        device=args.device,
        require_gpu=args.require_gpu,
        seed=args.seed,
    )
    config = ExperimentConfig(
        lookback=args.lookback,
        horizons=horizons,
        test_size=args.test_size,
        seasonality=args.seasonality,
        smoothing_method=args.smoothing_method,
        outlier_window=args.outlier_window,
        outlier_threshold=args.outlier_threshold,
        moving_average_window=args.moving_average_window,
        exp_alpha=args.exp_alpha,
        max_test_origins=args.max_test_origins,
        include_statistical_baselines=not args.no_statistical_baselines,
        tcn=tcn_config,
    )

    print(f"Series: {series.name}, observations: {len(series)}, range: {series.index.min().date()}..{series.index.max().date()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    result = run_experiment(series, config)
    used_devices = sorted({item.device for item in result.tcn_results.values()})
    print(f"TCN device(s): {', '.join(used_devices)}")
    print(f"Train observations: {result.split_idx}, test origins: {len(result.origins)}")
    print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.metrics.to_csv(output_path, index=False)
        print(f"Saved metrics to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TCN preprocessing variants on a real FRED economic time series."
    )
    parser.add_argument("--fred-series", default="CPIAUCSL", help="FRED series id, e.g. CPIAUCSL, UNRATE, INDPRO")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--no-cache", action="store_true", help="Download FRED data even if cached CSV exists")
    parser.add_argument("--lookback", type=int, default=36)
    parser.add_argument("--horizons", default="1,3,6,12")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seasonality", type=int, default=12)
    parser.add_argument("--smoothing-method", choices=["moving_average", "exponential"], default="moving_average")
    parser.add_argument("--outlier-window", type=int, default=24)
    parser.add_argument("--outlier-threshold", type=float, default=3.5)
    parser.add_argument("--moving-average-window", type=int, default=3)
    parser.add_argument("--exp-alpha", type=float, default=0.3)
    parser.add_argument("--max-test-origins", type=int, default=60)
    parser.add_argument("--no-statistical-baselines", action="store_true", help="Skip ARIMA, SARIMA and ETS")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--channels", default="32,32,32")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cuda", "gpu", "cpu"], default="auto")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is unavailable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/metrics.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
