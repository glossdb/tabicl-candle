"""Golden fixture for the chain-rule density read over a mixed
numerical/categorical surface, inner estimators pinned to the graded
member (n_estimators=1, norm "none"). The categorical conditionals run
through the classifier wrapper: sorted classes, searchsorted lookup,
1e-10 clip for unseen classes, 0.0 contribution for missing
observations.

Same replay mechanics as gen_density_fixtures.py: permutations and the
empty-conditioning noise stream are recorded, per-permutation log
densities alongside the final scores.

    uv run python scripts/gen_density_mixed_fixtures.py
"""

import os

import numpy as np
from tabicl import TabICLUnsupervised
from tabicl._sklearn.preprocessing import Shuffler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    rng = np.random.default_rng(23)
    n_train, n_test, d = 48, 16, 5
    n = n_train + n_test

    a = rng.normal(0.0, 1.0, n)
    b = 0.8 * a + 0.6 * rng.normal(0.0, 1.0, n)  # correlated with a
    c = rng.lognormal(0.0, 0.7, n)  # skewed
    # cat1: three contiguous classes driven by a
    cat1 = np.digitize(a + 0.3 * rng.normal(0.0, 1.0, n), [-0.5, 0.5]).astype(np.float64)
    # cat2: non-contiguous class values, weakly driven by b
    cat2 = np.array([1.0, 3.0, 7.0, 9.0])[np.digitize(b, [-1.0, 0.0, 1.0])]
    X = np.stack([a, b, cat1, c, cat2], axis=1).astype(np.float32)

    X_train, X_test = X[:n_train].copy(), X[n_train:].copy()
    X_train[[5, 17], 3] = np.nan  # numerical: target rows dropped
    X_train[[2, 9, 30], 4] = np.nan  # categorical: target rows dropped
    X_test[[1, 7], 4] = np.nan  # missing observation -> 0.0 contribution
    X_test[3, 4] = 5.0  # class unseen in train -> log(1e-10)
    assert 5.0 not in X_train[:, 4]

    model = TabICLUnsupervised(
        n_estimators=1,
        categorical_features=[2, 4],
        batch_size=8,
        random_state=42,
        device="cpu",
        estimator_params={
            "norm_methods": "none",
            "feat_shuffle_method": "none",
            "use_amp": False,
        },
    )
    model.fit(X_train)
    assert model.categorical_features_ == [2, 4]

    # Replay score_samples piecewise: same permutations, same rng.
    perms = Shuffler(d, random_state=42).shuffle(4)
    rng_replay = np.random.default_rng(42)
    log_densities = np.stack(
        [model._compute_log_density(X_test, np.asarray(p), rng_replay) for p in perms]
    )
    scores = np.exp(log_densities.mean(axis=0))
    assert np.allclose(
        scores, model.score_samples(X_test, n_permutations=4), atol=1e-12
    )

    # Record the noise stream the oracle consumed: one train draw and
    # one test draw per permutation, sized by the first feature's
    # non-NaN training rows.
    rng_noise = np.random.default_rng(42)
    noise = {}
    for k, p in enumerate(perms):
        n_tr = int((~np.isnan(X_train[:, p[0]])).sum())
        noise[f"noise_train_{k}"] = rng_noise.standard_normal((n_tr, 1)).astype(np.float32)
        noise[f"noise_test_{k}"] = rng_noise.standard_normal((n_test, 1)).astype(np.float32)

    np.savez_compressed(
        f"{ROOT}/fixtures/density_mixed_scores.npz",
        x_train=X_train,
        x_test=X_test,
        categorical=np.array([2, 4], dtype=np.int64),
        perms=np.asarray(perms, dtype=np.int64),
        log_densities=log_densities,
        scores=scores,
        **noise,
    )
    print(f"density_mixed_scores.npz: perms {np.asarray(perms).shape}, scores {scores.shape}")
    print(f"log density range: [{log_densities.min():.3f}, {log_densities.max():.3f}]")


if __name__ == "__main__":
    main()
