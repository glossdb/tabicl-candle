"""Extract the E2.1 walk-forward matrices from the tfmeval harness —
stage 3 of the fidelity gate reproduces the recorded band-calibration
numbers from the Rust side, and this script freezes exactly what the
harness fed TabICL: per (corpus, series, t), the median-filled feature
matrix for months < t, the target values, the month-t feature row, and
the actual. Protocol copied from tfmeval experiments/e2_1_bands.py
(walk-forward, causal; features t, moy, lag1, lag3m, lag12).

Runs in the tfmeval environment (needs its corpora on disk — generate()
is manifest-gated and reuses them):

    uv run --project ../tfmeval python scripts/extract_bands_walk.py
"""

import json
import os
import sys
from pathlib import Path

TFMEVAL = Path(__file__).resolve().parents[2] / "tfmeval"
sys.path.insert(0, str(TFMEVAL / "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

from tfmeval import corpora  # noqa: E402

METRICS = ("revenue", "gross_profit", "expenses", "operating_income",
           "cash_balance", "ar_balance", "dso", "free_cash_flow")
FEATS = ["t", "moy", "lag1", "lag3m", "lag12"]


def metric_series(path: Path, metric: str) -> list[float]:
    gt = yaml.safe_load(open(path / "ground_truth.yaml"))
    for m in gt["metrics"]:
        if m["id"] == metric:
            vals = m["values"]["month"]
            return [float(vals[k]) for k in sorted(vals)]
    raise KeyError(metric)


def segment_revenue(path: Path) -> dict[str, list[float]]:
    c = corpora.load(path, tables=["sales_order_lines", "sales_orders", "customers"])
    df = (
        c.tables["sales_order_lines"]
        .join(c.tables["sales_orders"], on="order_id")
        .join(c.tables["customers"], on="customer_id")
        .with_columns(pl.col("order_date").cast(pl.Utf8).str.slice(0, 7).alias("month"))
        .group_by(["segment", "month"])
        .agg(pl.col("line_amount").sum().alias("value"))
        .sort(["segment", "month"])
    )
    return {
        str(seg): sub.sort("month")["value"].to_list()
        for (seg,), sub in df.partition_by("segment", as_dict=True).items()
    }


def feature_rows(series: list[float]) -> pd.DataFrame:
    rows = []
    for t in range(1, len(series)):
        rows.append({
            "t": t, "moy": t % 12 + 1, "lag1": series[t - 1],
            "lag3m": float(np.mean(series[max(0, t - 3): t])),
            "lag12": series[t - 12] if t >= 12 else np.nan,
            "value": series[t],
        })
    return pd.DataFrame(rows)


def main() -> None:
    corpora_specs = [
        (dict(strategy="clean", seed=42, months=12), 6),
        (dict(strategy="clean", seed=43, months=12), 6),
        (dict(strategy="clean", seed=44, months=12), 6),
        (dict(strategy="clean", seed=42, months=24), 8),
    ]

    index, labels = [], []
    train_x_all, train_y_all, test_x, actual = [], [], [], []
    offset = 0
    series_id = 0

    def walk(series: list[float], start: int, grain: int, label: dict) -> None:
        nonlocal offset, series_id
        frame = feature_rows(series)
        for t in range(start, len(series)):
            train = frame[frame.t < t].copy()
            test = frame[frame.t == t].copy()
            med = train[FEATS].median(numeric_only=True).fillna(0.0)
            train[FEATS] = train[FEATS].fillna(med)
            test[FEATS] = test[FEATS].fillna(med)
            n = len(train)
            index.append([grain, label["seed"], label["months"], t, offset, n, series_id])
            train_x_all.append(train[FEATS].to_numpy(dtype=np.float64))
            train_y_all.append(train["value"].to_numpy(dtype=np.float64))
            test_x.append(test[FEATS].to_numpy(dtype=np.float64)[0])
            actual.append(float(series[t]))
            offset += n
        labels.append(label)
        series_id += 1

    for spec, start in corpora_specs:
        path = corpora.generate(**spec)
        for metric in METRICS:
            walk(metric_series(path, metric), start, 0,
                 {"metric": metric, "seed": spec["seed"], "months": spec["months"]})
        for seg, series in segment_revenue(path).items():
            walk(series, start, 1,
                 {"metric": f"revenue@{seg}", "seed": spec["seed"], "months": spec["months"]})
        print(f"extracted {path.name}", flush=True)

    np.savez_compressed(
        f"{ROOT}/fixtures/bands_walk.npz",
        index=np.asarray(index, dtype=np.int64),
        train_x_all=np.concatenate(train_x_all, axis=0),
        train_y_all=np.concatenate(train_y_all),
        test_x=np.asarray(test_x),
        actual=np.asarray(actual),
    )
    with open(f"{ROOT}/fixtures/bands_walk_series.json", "w") as fh:
        json.dump(labels, fh, indent=1)
    print(f"bands_walk.npz: {len(index)} fits over {series_id} series, "
          f"{offset} pooled train rows")


if __name__ == "__main__":
    main()
