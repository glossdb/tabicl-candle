"""Golden fixture for the classifier wrapper pinned to one member
(n_estimators=1, norm method "none" — the Shuffler short-circuits both
the feature and the class shuffle to identity for a single estimator).
Records the member's model inputs and the end-to-end probabilities so
the Rust classifier can be graded piecewise without Python present.

The synthetic data reuses the stage-2 adversarial recipe (skewed column,
constant column, injected outliers, NaNs in train and test); the labels
are non-contiguous values so the label encoder does real work, and the
class shuffle method stays at its default "shift" — the same shape the
density read's inner classifiers will have.

    uv run python verify/python/gen_classifier_fixtures.py
"""

import os

import numpy as np
from tabicl import TabICLClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    rng = np.random.default_rng(11)
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

    # Labels from a noisy score cut into three bins, mapped to
    # non-contiguous values so LabelEncoder does real work.
    score = 0.7 * f0 - 1.3 * np.log1p(f1) + 0.5 * f2 * f4 + rng.normal(0.0, 0.5, n)
    y = np.select(
        [score < np.quantile(score, 0.35), score < np.quantile(score, 0.7)],
        [42.0, 3.0],
        default=7.0,
    )

    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train]
    assert len(np.unique(y_train)) == 3

    est = TabICLClassifier(
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        use_amp=False,
        device="cpu",
        random_state=42,
    )
    est.fit(X_train, y_train)

    # Guard the pin: exactly one member, identity feature shuffle,
    # identity class shuffle (default "shift" short-circuited).
    assert list(est.ensemble_generator_.ensemble_configs_.keys()) == ["none"]
    ((shuffle, y_pattern),) = est.ensemble_generator_.ensemble_configs_["none"]
    assert list(shuffle) == list(range(len(shuffle)))
    assert list(y_pattern) == list(range(est.n_classes_))

    # The member's model inputs, straight from the wrapper's own plane.
    enc_test = est.X_encoder_.transform(X_test)
    ((Xs, ys),) = est.ensemble_generator_.transform(enc_test, mode="both").values()

    proba = est.predict_proba(X_test)
    labels = est.predict(X_test)

    np.savez_compressed(
        f"{ROOT}/crates/tabicl-inference/fixtures/wrapper_classifier.npz",
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        model_x=Xs.astype(np.float64),
        model_y=ys.astype(np.float64),
        classes=est.classes_.astype(np.float64),
        proba=proba.astype(np.float32),
        labels=labels.astype(np.float64),
    )
    print(
        f"wrapper_classifier.npz: model_x {Xs.shape}, classes {est.classes_}, "
        f"proba {proba.shape}, label counts {np.unique(labels, return_counts=True)}"
    )


if __name__ == "__main__":
    main()
