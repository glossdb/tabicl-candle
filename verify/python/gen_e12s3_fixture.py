"""Reduced-instance fixture for the stage-3 density half: the E1.2s3
shuffled-variant joined surface (the AUROC 0.9338 headline), cut down to
a size a standing CPU test can carry. The full protocol fits on the
whole 14,928-row reference — feasible for the torch oracle on MPS,
prohibitive for a cargo test — so the split is: the one-member verdict
runs Python-side on the full protocol (run_e12s3_pinned.py), and this
fixture grades the Rust port on a sampled real-data instance: pinned
inner estimators, recorded permutations and noise, per-perm log-density
parity, and the read-out contract (NLL, q95 threshold, batch score,
AUROC) computed on exactly this instance.

The instance mirrors the protocol's shape: the scored reference frame
is a subset of the conditioning context, as ref_sample is a subset of
the fitted reference in the engine.

Needs the tfmeval sibling for corpus + prep:

    uv run --project ../tfmeval python verify/python/gen_e12s3_fixture.py
"""

import os
import sys
from pathlib import Path

TFMEVAL = Path(__file__).resolve().parents[3] / "tfmeval"
sys.path.insert(0, str(TFMEVAL / "src"))
sys.path.insert(0, str(TFMEVAL / "experiments"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tabicl import TabICLUnsupervised  # noqa: E402
from tabicl._sklearn.preprocessing import Shuffler  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_CONTEXT = 400
N_SCORE = 200
N_PERMS = 2


def build_surface():
    os.chdir(TFMEVAL)  # corpora paths are relative to the tfmeval root
    from e1_2s3_fk import VARIANTS, joined_surface, load_pair, truth_rows
    from tfmeval.engines.classical import prep

    clean, corp = load_pair(VARIANTS["shuffled"])
    cols = ["amount", "amount_inv", "dd", "amount_gap", "terms_days",
            "method", "status", "category"]
    ref_j = joined_surface(clean.tables["payments"], clean.tables["invoices"])
    bat_j = joined_surface(corp.tables["payments"], corp.tables["invoices"])
    ref, bat = prep(ref_j.select(cols).to_pandas(), bat_j.select(cols).to_pandas())
    keep = [c for c in ref.columns if ref[c].nunique(dropna=True) > 1]
    ref, bat = ref[keep], bat[keep]
    med = ref.median(numeric_only=True)
    ref, bat = ref.fillna(med).fillna(0.0), bat.fillna(med).fillna(0.0)
    return ref, bat, truth_rows(corp)


def main() -> None:
    ref, bat, labels_set = build_surface()
    assert list(ref.columns) == ["amount", "amount_inv", "dd", "amount_gap",
                                 "terms_days", "method", "status", "category"]

    context = ref.sample(N_CONTEXT, random_state=1)
    ref_score = context.sample(N_SCORE, random_state=0)  # subset of the context
    bat_score = bat.sample(N_SCORE, random_state=2)
    labels = np.array([1 if i in labels_set else 0 for i in bat_score.index])
    assert 0 < labels.sum() < N_SCORE

    X_ctx = context.to_numpy().astype(np.float32)
    X_ref = ref_score.to_numpy().astype(np.float32)
    X_bat = bat_score.to_numpy().astype(np.float32)

    model = TabICLUnsupervised(
        n_estimators=1,
        batch_size=8,
        random_state=42,
        device="cpu",
        estimator_params={
            "norm_methods": "none",
            "feat_shuffle_method": "none",
            "use_amp": False,
        },
    )
    model.fit(X_ctx)
    print(f"categorical features (auto-detected): {model.categorical_features_}")

    d = X_ctx.shape[1]
    perms = Shuffler(d, random_state=42).shuffle(N_PERMS)[:N_PERMS]

    # Per frame, replay the first N_PERMS iterations of score_samples:
    # fresh rng per frame, permutations in order.
    frames = {"ref": X_ref, "bat": X_bat}
    log_densities = {}
    noise = {}
    for name, X in frames.items():
        rng = np.random.default_rng(42)
        log_densities[name] = np.stack(
            [model._compute_log_density(X, np.asarray(p), rng) for p in perms]
        )
        rng_noise = np.random.default_rng(42)
        for k, p in enumerate(perms):
            n_tr = int((~np.isnan(X_ctx[:, p[0]])).sum())
            noise[f"noise_{name}_train_{k}"] = rng_noise.standard_normal(
                (n_tr, 1)).astype(np.float32)
            noise[f"noise_{name}_test_{k}"] = rng_noise.standard_normal(
                (len(X), 1)).astype(np.float32)

    # The read-out contract on this instance: NLL from the composed
    # score, q95 threshold over the reference frame, batch score, AUROC.
    tiny = float(np.finfo(float).tiny)
    nll = {
        name: -np.log(np.exp(ld.mean(axis=0)) + tiny)
        for name, ld in log_densities.items()
    }
    threshold = float(np.quantile(nll["ref"], 0.95))
    batch_score = float((nll["bat"] > threshold).mean())
    auroc = float(roc_auc_score(labels, nll["bat"]))
    print(f"reduced instance: threshold={threshold:.4f} "
          f"batch_score={batch_score:.4f} auroc={auroc:.4f} "
          f"({labels.sum()} injected of {N_SCORE})")

    np.savez_compressed(
        f"{ROOT}/verify/experiments/fixtures/e12s3_reduced.npz",
        context=X_ctx,
        ref_score=X_ref,
        bat_score=X_bat,
        labels=labels.astype(np.int64),
        categorical=np.array(model.categorical_features_, dtype=np.int64),
        perms=np.asarray(perms, dtype=np.int64),
        log_densities_ref=log_densities["ref"],
        log_densities_bat=log_densities["bat"],
        nll_ref=nll["ref"],
        nll_bat=nll["bat"],
        threshold=np.array([threshold]),
        batch_score=np.array([batch_score]),
        auroc=np.array([auroc]),
        **noise,
    )
    print(f"e12s3_reduced.npz: context {X_ctx.shape}, perms {np.asarray(perms).shape}")


if __name__ == "__main__":
    main()
