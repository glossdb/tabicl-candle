"""Full-default sklearn oracle over the E4 fit matrices: one
TabICLRegressor with the recorded run's configuration (defaults:
n_estimators=8, norms none+power, latin feature shuffles,
random_state=0) per fit. Captures the concrete member configs the
seeded generator produced (norm + feature permutation per member —
narrow fits yield fewer than 8 distinct members) and the final
averaged bands, written to fixtures/e4_ensemble.npz for the Rust
ensemble test to grade against with injected configs.

Also prints the comparison against the recorded tfmeval figures — the
recorded run is the same configuration, so this rerun should sit on
top of it up to device drift.

    uv run python scripts/gen_e4_ensemble_oracle.py
"""

import json
import os

import numpy as np
from tabicl import TabICLRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "..", "tfmeval", "output", "results", "e4_whatif.jsonl")

ALPHAS = [0.05, 0.10, 0.50, 0.90, 0.95]
METRIC = {0: "revenue", 1: "gross_profit", 2: "cash_balance",
          3: "ar_balance", 4: "dso", 5: "free_cash_flow"}
PHASE = {1: "e4_1_in_support", 2: "e4_2_out_of_support"}
NORM_CODE = {"none": 0, "power": 1}


def bands_for(train_x, train_y, test_x):
    est = TabICLRegressor(use_amp=False, device="cpu", random_state=0)
    est.fit(train_x, train_y)
    members = []
    for norm, configs in est.ensemble_generator_.ensemble_configs_.items():
        for shuffle, y_pattern in configs:
            assert y_pattern is None
            members.append([NORM_CODE[norm], *shuffle])
    q = np.asarray(
        est.predict(test_x, output_type="quantiles", alphas=ALPHAS), dtype=float
    )
    if q.shape[0] != len(ALPHAS):
        q = q.T
    return q, np.asarray(members, dtype=np.int64)


def main() -> None:
    d = np.load(f"{ROOT}/fixtures/e4_walk.npz")
    recorded = {}
    with open(RESULTS) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("engine") != "tabicl":
                continue
            if row["phase"] == "e4_3_interaction":
                recorded[("inter", row["metric"])] = row
            else:
                recorded[(row["phase"], row["metric"], row["factor"], row["grid"])] = row

    grid_bands, grid_members = [], []
    for i, idx in enumerate(d["grid_index"]):
        phase, metric, f100, seed, grid = (int(v) for v in idx)
        off, n = (int(v) for v in d["grid_offsets"][i])
        q, members = bands_for(d["grid_train_x"][off:off + n],
                               d["grid_train_y"][off:off + n], d["grid_test_x"][i])
        grid_bands.append(q)
        grid_members.append(members)
        truth, base = d["grid_truth"][i], d["grid_base"][i]
        p50 = q[2]
        mape = round(float(np.median(np.abs(p50 - truth) / np.abs(truth))), 4)
        key = (PHASE[phase], METRIC[metric], f100 / 100,
               "fine" if grid == 1 else "coarse")
        rec = recorded[key]
        print(f"{key[1]:<13} f={key[2]:<4} {key[3]:<6} members {len(members)} "
              f"mape {mape:<7} | recorded mape {rec['mape_p50']}")

    inter_bands, inter_members = [], []
    for i, mcode in enumerate(d["inter_metric"]):
        q, members = bands_for(d["inter_train_x"][i], d["inter_train_y"][i],
                               d["inter_test_x"][i])
        inter_bands.append(q)
        inter_members.append(members)
        truth = d["inter_truth"][i]
        err = round(float(np.median(np.abs(q[2] - truth))), 2)
        rec = recorded[("inter", METRIC[int(mcode)])]
        print(f"interaction {METRIC[int(mcode)]:<15} members {len(members)} "
              f"err {err:<9} | recorded err {rec['median_abs_err_model']}")

    assert len({m.shape for m in grid_members}) == 1, "ragged grid member sets"
    assert len({m.shape for m in inter_members}) == 1, "ragged inter member sets"
    np.savez_compressed(
        f"{ROOT}/fixtures/e4_ensemble.npz",
        grid_bands=np.stack(grid_bands),
        grid_members=np.stack(grid_members),
        inter_bands=np.stack(inter_bands),
        inter_members=np.stack(inter_members),
    )
    print(f"e4_ensemble.npz: grid {np.stack(grid_bands).shape} "
          f"members {np.stack(grid_members).shape}, "
          f"inter {np.stack(inter_bands).shape} "
          f"members {np.stack(inter_members).shape}")


if __name__ == "__main__":
    main()
