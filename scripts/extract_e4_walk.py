"""Extract the E4 what-if fit matrices from the tfmeval harness —
every TabICL fit the recorded run performed, matrix-exact: the fine
grid (seed 42, 7 factors, held-out 1.15 + out-of-support 1.6/2.0), the
coarse grid (seed 43, 3 factors, held-out only), and the two-lever
interaction (baseline + singles -> joint). The experiment module is
imported directly so the data assembly cannot drift from the protocol.

Writes fixtures/e4_walk.npz:
  grid fits (16): index (phase, metric, factor*100, seed, grid),
    flattened train_x/train_y with offsets, test_x (6,2), truth (6,),
    base_vals (6,)
  interaction fits (5): metric codes, train (18,3), test (6,3),
    truth (6,), additive (6,)

Codes: phase 1=in_support 2=out_of_support 3=interaction; grid
1=fine 2=coarse; metric 0=revenue 1=gross_profit 2=cash_balance
3=ar_balance 4=dso 5=free_cash_flow.

    uv run --project ../tfmeval python scripts/extract_e4_walk.py
"""

import os
import sys
from pathlib import Path

TFMEVAL = Path(__file__).resolve().parents[2] / "tfmeval"
sys.path.insert(0, str(TFMEVAL / "src"))
sys.path.insert(0, str(TFMEVAL / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRIC_CODE = {"revenue": 0, "gross_profit": 1, "cash_balance": 2,
               "ar_balance": 3, "dso": 4, "free_cash_flow": 5}
FEATS = ["factor", "month_index"]


def main() -> None:
    os.chdir(TFMEVAL)  # corpora paths are relative to the tfmeval root
    import e4_whatif as e4
    import yaml
    from tfmeval import corpora

    grid_fits = []
    for seed, factors, label in [(42, [1.0, *e4.GRID], "fine"),
                                 (43, [1.0, 0.85, 1.30], "coarse")]:
        base = corpora.generate(strategy="clean", seed=seed, months=12)
        for metric in e4.METRICS:
            train = e4.build_rows(seed, factors, metric)
            base_series = e4.monthly_series(base, metric)
            months = e4.post_months(base_series)
            base_vals = np.asarray([base_series[m] for m in months])

            targets = [(e4.HELD, 1)]
            if label == "fine":
                targets += [(f, 2) for f in e4.OOS]
            for f, phase in targets:
                truth_series = e4.monthly_series(e4.lever_path(seed, f), metric)
                test = pd.DataFrame({
                    "factor": f,
                    "month_index": [sorted(truth_series).index(m) for m in months],
                    "value": [truth_series[m] for m in months],
                })
                grid_fits.append(dict(
                    index=[phase, METRIC_CODE[metric], int(round(f * 100)),
                           seed, 1 if label == "fine" else 2],
                    train_x=train[FEATS].to_numpy(dtype=float),
                    train_y=train.value.to_numpy(dtype=float),
                    test_x=test[FEATS].to_numpy(dtype=float),
                    truth=test.value.to_numpy(dtype=float),
                    base=base_vals,
                ))

    # Interaction: replicate run_interaction's assembly exactly.
    seed = 42
    base = corpora.generate(strategy="clean", seed=seed, months=12)
    single_a = e4.lever_path(seed, 1.15)
    single_b = corpora.generate(strategy="clean", seed=seed, months=12,
                                lever={"type": "rate", "driver": "collection_lag",
                                       "period_k": e4.K, "factor": 1.5})
    pair = corpora.generate(strategy="clean", seed=seed, months=12,
                            levers=[{"type": "rate", "driver": "price",
                                     "period_k": e4.K, "factor": 1.15},
                                    {"type": "rate", "driver": "collection_lag",
                                     "period_k": e4.K, "factor": 1.5}])

    inter_fits = []
    for metric in ("revenue", "cash_balance", "ar_balance", "dso", "free_cash_flow"):
        frames = []
        for fp, fl, path in ((1.0, 1.0, base), (1.15, 1.0, single_a),
                             (1.0, 1.5, single_b)):
            series = e4.monthly_series(path, metric)
            months = e4.post_months(series)
            frames.append(pd.DataFrame({
                "factor_price": fp, "factor_lag": fl,
                "month_index": [sorted(series).index(m) for m in months],
                "value": [series[m] for m in months]}))
        train = pd.concat(frames, ignore_index=True)

        joint = e4.monthly_series(pair, metric)
        months = e4.post_months(joint)
        truth = np.asarray([joint[m] for m in months])
        test = pd.DataFrame({"factor_price": 1.15, "factor_lag": 1.5,
                             "month_index": [sorted(joint).index(m) for m in months]})

        base_s = e4.monthly_series(base, metric)
        a_s = e4.monthly_series(single_a, metric)
        b_s = e4.monthly_series(single_b, metric)
        additive = np.asarray([base_s[m] + (a_s[m] - base_s[m]) + (b_s[m] - base_s[m])
                               for m in months])

        feats = ["factor_price", "factor_lag", "month_index"]
        inter_fits.append(dict(
            metric=METRIC_CODE[metric],
            train_x=train[feats].to_numpy(dtype=float),
            train_y=train.value.to_numpy(dtype=float),
            test_x=test[feats].to_numpy(dtype=float),
            truth=truth,
            additive=additive,
        ))

    grid_index = np.asarray([f["index"] for f in grid_fits], dtype=np.int64)
    grid_offsets, off = [], 0
    for f in grid_fits:
        n = len(f["train_y"])
        grid_offsets.append([off, n])
        off += n
    out = dict(
        grid_index=grid_index,
        grid_offsets=np.asarray(grid_offsets, dtype=np.int64),
        grid_train_x=np.concatenate([f["train_x"] for f in grid_fits]),
        grid_train_y=np.concatenate([f["train_y"] for f in grid_fits]),
        grid_test_x=np.stack([f["test_x"] for f in grid_fits]),
        grid_truth=np.stack([f["truth"] for f in grid_fits]),
        grid_base=np.stack([f["base"] for f in grid_fits]),
        inter_metric=np.asarray([f["metric"] for f in inter_fits], dtype=np.int64),
        inter_train_x=np.stack([f["train_x"] for f in inter_fits]),
        inter_train_y=np.stack([f["train_y"] for f in inter_fits]),
        inter_test_x=np.stack([f["test_x"] for f in inter_fits]),
        inter_truth=np.stack([f["truth"] for f in inter_fits]),
        inter_additive=np.stack([f["additive"] for f in inter_fits]),
    )
    np.savez_compressed(f"{ROOT}/fixtures/e4_walk.npz", **out)
    print(f"e4_walk.npz: {len(grid_fits)} grid fits "
          f"(train rows {sorted(set(o[1] for o in grid_offsets))}), "
          f"{len(inter_fits)} interaction fits")


if __name__ == "__main__":
    main()
