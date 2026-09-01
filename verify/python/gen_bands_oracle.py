"""Pinned-member oracle for the E2.1 band walk: run the sklearn
TabICLRegressor with the graded pin (n_estimators=1, norm "none") over
every extracted fit and record the five band levels. The Rust side must
match these tightly (same math); the distance between this summary and
the recorded default-ensemble summary is the measured ensembling
effect, which stage 3 reports.

    uv run python verify/python/gen_bands_oracle.py
"""

import os

import numpy as np
from tabicl import TabICLRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALPHAS = [0.05, 0.10, 0.50, 0.90, 0.95]


def main() -> None:
    fx = np.load(f"{ROOT}/verify/experiments/fixtures/bands_walk.npz")
    index = fx["index"]
    train_x_all, train_y_all = fx["train_x_all"], fx["train_y_all"]
    test_x, actual = fx["test_x"], fx["actual"]

    bands = np.zeros((len(index), len(ALPHAS)))
    for i, (grain, seed, months, t, off, n, sid) in enumerate(index):
        est = TabICLRegressor(
            n_estimators=1,
            norm_methods="none",
            feat_shuffle_method="none",
            use_amp=False,
            device="cpu",
            random_state=0,
        )
        est.fit(train_x_all[off : off + n], train_y_all[off : off + n])
        out = est.predict(test_x[i : i + 1], output_type="quantiles", alphas=ALPHAS)
        bands[i] = np.asarray(out, dtype=np.float64).reshape(-1)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(index)}", flush=True)

    np.savez_compressed(f"{ROOT}/verify/experiments/fixtures/bands_pinned.npz", bands=bands)

    # Summary in the harness's own terms, per grain
    for grain, name in ((0, "month"), (1, "segment")):
        m = index[:, 0] == grain
        lo80, hi80 = np.sort(bands[m][:, [1, 3]], axis=1).T
        lo90, hi90 = np.sort(bands[m][:, [0, 4]], axis=1).T
        a = actual[m]
        cov80 = np.mean((lo80 <= a) & (a <= hi80))
        cov90 = np.mean((lo90 <= a) & (a <= hi90))
        medw = np.median(np.abs(hi80 - lo80) / np.maximum(1e-9, np.abs(a)))
        print(f"pinned {name}: n={m.sum()} cov80={cov80:.4f} "
              f"cov90={cov90:.4f} medw80={medw:.4f}")


if __name__ == "__main__":
    main()
