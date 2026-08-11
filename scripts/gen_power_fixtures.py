"""Golden fixtures for the Yeo-Johnson power stage: sklearn's
PowerTransformer(method="yeo-johnson", standardize=True) over matrices
covering the branches the port must reproduce — mixed signs, all
positive, all negative, a constant column (lambda pinned to 1), a tiny
sample, and heavy outliers. Saves lambdas and transformed train/test
per case to fixtures/power.npz.

    uv run python scripts/gen_power_fixtures.py
"""

import os

import numpy as np
from sklearn.preprocessing import PowerTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r = np.random.RandomState(7)
    mixed = r.randn(40, 5) * [1.0, 3.0, 0.2, 10.0, 1.0] + [0.0, -2.0, 5.0, 0.0, 100.0]
    pos = r.lognormal(mean=1.0, sigma=0.8, size=(40, 4))
    neg = -r.lognormal(mean=0.5, sigma=1.2, size=(30, 3))
    const = r.randn(25, 4)
    const[:, 2] = 3.5
    tiny = r.randn(6, 3)
    outlier = r.randn(50, 4)
    outlier[3, 0] += 60.0
    outlier[17, 2] -= 45.0
    out = {}
    for name, x in [("mixed", mixed), ("pos", pos), ("neg", neg),
                    ("const", const), ("tiny", tiny), ("outlier", outlier)]:
        n_test = max(4, x.shape[0] // 4)
        test = r.randn(n_test, x.shape[1]) * np.abs(x).std(axis=0) + x.mean(axis=0)
        if name == "pos":
            test = np.abs(test) + 1e-3
        if name == "neg":
            test = -np.abs(test) - 1e-3
        out[name] = (x, test)
    return out


def main() -> None:
    arrays = {}
    for name, (train, test) in cases().items():
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        out_train = pt.fit_transform(train.copy())
        out_test = pt.transform(test.copy())
        arrays[f"{name}_train"] = train
        arrays[f"{name}_test"] = test
        arrays[f"{name}_lambdas"] = pt.lambdas_.astype(np.float64)
        arrays[f"{name}_out_train"] = out_train
        arrays[f"{name}_out_test"] = out_test
        print(f"{name:<8} lambdas {np.round(pt.lambdas_, 4)}")
    np.savez_compressed(f"{ROOT}/fixtures/power.npz", **arrays)
    print("power.npz written")


if __name__ == "__main__":
    main()
