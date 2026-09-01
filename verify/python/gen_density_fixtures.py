"""Golden fixture for the chain-rule density read
(TabICLUnsupervised.score_samples), numerical columns only, inner
estimators pinned to the graded member (n_estimators=1, norm "none").

Two oracle quirks are recorded rather than ported:

- The permutations come from the sklearn Shuffler's default latin
  method, which ignores n_permutations and returns n_features
  permutations, seeded by Python's Mersenne Twister. The Rust API takes
  permutations as an input; this fixture records the oracle's.
- The empty-conditioning dummy feature is numpy-Generator Gaussian
  noise, drawn train-then-test once per permutation (only the first
  feature of a permutation has no conditioning). The Rust API takes a
  noise provider; this fixture records the oracle's stream by replaying
  the identically-seeded generator.

Per-permutation log densities are recorded alongside the final scores
so the Rust side can be graded piecewise.

    uv run python verify/python/gen_density_fixtures.py
"""

import os

import numpy as np
from tabicl import TabICLUnsupervised
from tabicl._sklearn.preprocessing import Shuffler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    rng = np.random.default_rng(11)
    n_train, n_test, d = 48, 16, 4
    n = n_train + n_test

    a = rng.normal(0.0, 1.0, n)
    b = 0.8 * a + 0.6 * rng.normal(0.0, 1.0, n)  # correlated with a
    c = rng.lognormal(0.0, 0.7, n)  # skewed
    e = rng.normal(2.0, 1.5, n)
    X = np.stack([a, b, c, e], axis=1).astype(np.float32)

    X_train, X_test = X[:n_train].copy(), X[n_train:].copy()
    X_train[[5, 17], 3] = np.nan  # target rows dropped, conditioning imputed

    model = TabICLUnsupervised(
        n_estimators=1,
        categorical_features=[],
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

    # Replay score_samples piecewise: same permutations, same rng.
    perms = Shuffler(d, random_state=42).shuffle(4)
    rng_replay = np.random.default_rng(42)
    log_densities = np.stack(
        [model._compute_log_density(X_test, np.asarray(p), rng_replay) for p in perms]
    )
    scores = np.exp(log_densities.mean(axis=0))
    assert np.allclose(scores, model.score_samples(X_test, n_permutations=4), atol=1e-12)

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
        f"{ROOT}/verify/experiments/fixtures/density_scores.npz",
        x_train=X_train,
        x_test=X_test,
        perms=np.asarray(perms, dtype=np.int64),
        log_densities=log_densities,
        scores=scores,
        **noise,
    )
    print(f"density_scores.npz: perms {np.asarray(perms).shape}, scores {scores.shape}")
    print(f"log density range: [{log_densities.min():.3f}, {log_densities.max():.3f}]")


if __name__ == "__main__":
    main()
