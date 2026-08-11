"""Golden fixture for stage 2 of the fidelity gate: the sklearn wrapper
pinned to one member (n_estimators=1, norm method "none", no shuffles —
the Shuffler short-circuits to identity for a single estimator). Records
the wrapper's preprocessing intermediates and its end-to-end predictions
so the Rust wrapper can be graded piecewise without Python present.

The synthetic data is adversarial on purpose: a skewed column, injected
outliers in train and test (the two-stage soft clip), NaNs in train and
test (mean imputation), and a constant column (the unique-value filter).

    uv run python scripts/gen_wrapper_fixtures.py
"""

import os

import numpy as np
from tabicl import TabICLRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The band levels the evaluation harness uses (tfmeval conditional.py).
ALPHAS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def main() -> None:
    rng = np.random.default_rng(7)
    n_train, n_test = 60, 20
    n = n_train + n_test

    f0 = rng.normal(1.0, 3.0, n)
    f1 = rng.lognormal(0.0, 1.0, n)  # skewed
    f2 = rng.uniform(-2.0, 2.0, n)
    f3 = np.full(n, 3.25)  # constant -> dropped by the unique filter
    f4 = rng.normal(0.0, 1.0, n)
    f4[[3, 41, 67]] = [9.0, -11.0, 14.0]  # outliers in train and test
    f5 = rng.normal(5.0, 2.0, n)
    f5[[7, 22, 71]] = np.nan  # NaNs in train and test
    X = np.stack([f0, f1, f2, f3, f4, f5], axis=1)
    y = (
        0.7 * f0
        - 1.3 * np.log1p(f1)
        + 0.5 * f2 * f4
        + 0.2 * np.nan_to_num(f5)
        + rng.normal(0.0, 0.3, n)
    )

    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train]

    est = TabICLRegressor(
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        use_amp=False,
        device="cpu",
        random_state=42,
    )
    est.fit(X_train, y_train)

    # Guard the pin: exactly one member, identity shuffle, no y pattern.
    assert list(est.ensemble_generator_.ensemble_configs_.keys()) == ["none"]
    ((shuffle, y_pattern),) = est.ensemble_generator_.ensemble_configs_["none"]
    assert y_pattern is None and list(shuffle) == list(range(len(shuffle)))

    # The member's model inputs, straight from the wrapper's own plane.
    enc_test = est.X_encoder_.transform(X_test)
    ((Xs, ys),) = est.ensemble_generator_.transform(enc_test, mode="both").values()

    out = est.predict(X_test, output_type=["mean", "quantiles", "raw_quantiles"], alphas=ALPHAS)

    np.savez_compressed(
        f"{ROOT}/fixtures/wrapper_regressor.npz",
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        model_x=Xs.astype(np.float64),
        model_y=ys.astype(np.float64),
        y_mean=np.array([est.y_scaler_.mean_[0]], dtype=np.float64),
        y_scale=np.array([est.y_scaler_.scale_[0]], dtype=np.float64),
        mean=out["mean"].astype(np.float32),
        quantiles=out["quantiles"].astype(np.float32),
        raw_quantiles=out["raw_quantiles"].astype(np.float32),
        alphas=np.array(ALPHAS),
    )
    print(
        f"wrapper_regressor.npz: model_x {Xs.shape}, "
        f"mean {out['mean'].shape}, quantiles {out['quantiles'].shape}"
    )


if __name__ == "__main__":
    main()
