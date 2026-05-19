from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def load_fred_series(
    series_id: str = "CPIAUCSL",
    start_date: str | None = None,
    end_date: str | None = None,
    cache_dir: str | Path = "data/raw",
    use_cache: bool = True,
) -> pd.Series:
    """Load an economic time series from FRED as a sorted float Series.

    The FRED graph CSV endpoint does not require an API key and is suitable for
    reproducible examples with public macroeconomic data.
    """

    cache_path = Path(cache_dir) / f"{series_id}.csv"
    if use_cache and cache_path.exists():
        frame = pd.read_csv(cache_path)
    else:
        response = requests.get(FRED_CSV_URL, params={"id": series_id}, timeout=30)
        response.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(response.text, encoding="utf-8")
        frame = pd.read_csv(cache_path)

    if "observation_date" not in frame.columns or series_id not in frame.columns:
        raise ValueError(f"Unexpected FRED CSV format for series {series_id!r}")

    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame[series_id] = pd.to_numeric(frame[series_id].replace(".", pd.NA), errors="coerce")
    frame = frame.dropna(subset=[series_id]).sort_values("observation_date")

    if start_date is not None:
        frame = frame[frame["observation_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["observation_date"] <= pd.Timestamp(end_date)]

    series = frame.set_index("observation_date")[series_id].astype(float)
    series.name = series_id

    if len(series) < 50:
        raise ValueError(
            f"Series {series_id!r} has only {len(series)} observations after filtering; "
            "use a longer range."
        )
    return series
