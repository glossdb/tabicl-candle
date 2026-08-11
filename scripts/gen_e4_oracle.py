"""Pinned-member sklearn oracle over the E4 fit matrices: one
TabICLRegressor (n_estimators=1, norm "none", no shuffles) per fit,
bands at the protocol alphas, written to fixtures/e4_pinned.npz for the
Rust test to grade against.

Also prints the verdict comparison: the recorded run used full
defaults (8-member inner ensembles) — this is the experiment that
answers whether the pinned member carries the POINT read (mape_p50,
effect recovery), the one read class where ensembling plausibly
matters. Recorded figures are read from the tfmeval results archive.

    uv run python scripts/gen_e4_oracle.py
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


def bands_for(train_x, train_y, test_x):
    est = TabICLRegressor(
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        use_amp=False,
        device="cpu",
        random_state=0,
    )
    est.fit(train_x, train_y)
    ((shuffle, y_pattern),) = est.ensemble_generator_.ensemble_configs_["none"]
    assert y_pattern is None and list(shuffle) == list(range(len(shuffle)))
    q = np.asarray(
        est.predict(test_x, output_type="quantiles", alphas=ALPHAS), dtype=float
    )
    if q.shape[0] != len(ALPHAS):
        q = q.T
    return q  # (5, 6)


def grade(q, truth, base):
    p50 = q[2]
    true_delta = truth - base
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(true_delta != 0, (p50 - base) / true_delta, np.nan)
    width = np.abs(q[3] - q[1])
    lo80, hi80 = np.minimum(q[1], q[3]), np.maximum(q[1], q[3])
    lo90, hi90 = np.minimum(q[0], q[4]), np.maximum(q[0], q[4])
    return {
        "mape_p50": round(float(np.median(np.abs(p50 - truth) / np.abs(truth))), 4),
        "effect_recovery_median": round(float(np.nanmedian(rec)), 4),
        "coverage_80": round(float(np.mean((truth >= lo80) & (truth <= hi80))), 4),
        "coverage_90": round(float(np.mean((truth >= lo90) & (truth <= hi90))), 4),
        "median_width_rel": round(float(np.median(width / np.abs(truth))), 4),
    }


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

    grid_bands = []
    for i, idx in enumerate(d["grid_index"]):
        phase, metric, f100, seed, grid = (int(v) for v in idx)
        off, n = (int(v) for v in d["grid_offsets"][i])
        q = bands_for(d["grid_train_x"][off:off + n], d["grid_train_y"][off:off + n],
                      d["grid_test_x"][i])
        grid_bands.append(q)
        g = grade(q, d["grid_truth"][i], d["grid_base"][i])
        key = (PHASE[phase], METRIC[metric], f100 / 100,
               "fine" if grid == 1 else "coarse")
        rec = recorded[key]
        print(f"{key[1]:<13} f={key[2]:<4} {key[3]:<6} "
              f"mape {g['mape_p50']:<7} rec {g['effect_recovery_median']:<8} "
              f"cov80 {g['coverage_80']:<7} | recorded "
              f"mape {rec['mape_p50']:<7} rec {rec['effect_recovery_median']:<8} "
              f"cov80 {rec['coverage_80']}")

    inter_bands = []
    for i, mcode in enumerate(d["inter_metric"]):
        q = bands_for(d["inter_train_x"][i], d["inter_train_y"][i], d["inter_test_x"][i])
        inter_bands.append(q)
        truth = d["inter_truth"][i]
        p50 = q[2]
        lo80, hi80 = np.minimum(q[1], q[3]), np.maximum(q[1], q[3])
        err_model = round(float(np.median(np.abs(p50 - truth))), 2)
        cov80 = round(float(np.mean((truth >= lo80) & (truth <= hi80))), 4)
        rec = recorded[("inter", METRIC[int(mcode)])]
        print(f"interaction {METRIC[int(mcode)]:<15} err {err_model:<9} "
              f"cov80 {cov80:<7} | recorded err {rec['median_abs_err_model']:<9} "
              f"cov80 {rec['coverage_80']} (additive err "
              f"{rec['median_abs_err_additive']})")

    np.savez_compressed(
        f"{ROOT}/fixtures/e4_pinned.npz",
        grid_bands=np.stack(grid_bands),
        inter_bands=np.stack(inter_bands),
    )
    print(f"e4_pinned.npz: grid {np.stack(grid_bands).shape}, "
          f"inter {np.stack(inter_bands).shape}")


if __name__ == "__main__":
    main()
