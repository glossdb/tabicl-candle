"""The stage-3 density verdict experiment: does the pinned single
member carry the E1.2s3 joined-surface read? Reruns the recorded
protocol (tfmeval e1_2s3_fk, tabicl phase, relationship surface) with
the inner estimators pinned to the graded member — n_estimators=1,
norm "none", no shuffles, fp32 — and prints the pinned figures against
the recorded ones (4-member inner ensembles, latin + none/power, AMP).

Everything else matches the engine's density_score exactly: prep,
ref-constant column drop, median fill, fit on the full reference,
NLL = -log(score + tiny), q95 threshold from a 500-row reference
sample, n_permutations=2 (the latin Shuffler returns n_features
permutations regardless).

    uv run --project ../tfmeval python scripts/run_e12s3_pinned.py
"""

import sys
from pathlib import Path

TFMEVAL = Path(__file__).resolve().parents[2] / "tfmeval"
sys.path.insert(0, str(TFMEVAL / "src"))
sys.path.insert(0, str(TFMEVAL / "experiments"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tabicl import TabICLUnsupervised  # noqa: E402

RECORDED = {  # tfmeval output/results/e1_2s3_fk.jsonl, tabicl relationship
    "distinct": {"batch_score": 0.11254, "row_auroc": 0.991},
    "repeated": {"batch_score": 0.11254, "row_auroc": 0.991},
    "shuffled": {"batch_score": 0.144895, "row_auroc": 0.9338},
}


def main() -> None:
    import os

    os.chdir(TFMEVAL)  # corpora paths are relative to the tfmeval root
    from e1_2s3_fk import VARIANTS, joined_surface, load_pair, truth_rows
    from tfmeval.engines.classical import prep
    from tfmeval.engines.tabicl_engine import _device

    cols = ["amount", "amount_inv", "dd", "amount_gap", "terms_days",
            "method", "status", "category"]
    tiny = float(np.finfo(float).tiny)

    for name in ["shuffled", "distinct", "repeated"]:
        clean, corp = load_pair(VARIANTS[name])
        ref_j = joined_surface(clean.tables["payments"], clean.tables["invoices"])
        bat_j = joined_surface(corp.tables["payments"], corp.tables["invoices"])
        ref, bat = prep(ref_j.select(cols).to_pandas(), bat_j.select(cols).to_pandas())
        keep = [c for c in ref.columns if ref[c].nunique(dropna=True) > 1]
        ref, bat = ref[keep], bat[keep]
        med = ref.median(numeric_only=True)
        ref, bat = ref.fillna(med).fillna(0.0), bat.fillna(med).fillna(0.0)

        model = TabICLUnsupervised(
            n_estimators=1,
            device=_device(),
            estimator_params={
                "norm_methods": "none",
                "feat_shuffle_method": "none",
                "use_amp": False,
            },
        )
        model.fit(ref.to_numpy())

        def nll(frame):
            raw = np.asarray(
                model.score_samples(frame.to_numpy(), n_permutations=2), dtype=float
            )
            return -np.log(raw + tiny)

        ref_sample = ref if len(ref) <= 500 else ref.sample(500, random_state=0)
        ref_scores = nll(ref_sample)
        bat_scores = nll(bat)
        threshold = float(np.quantile(ref_scores, 0.95))
        batch_score = float((bat_scores > threshold).mean())

        labels_set = truth_rows(corp)
        y = np.array([1 if i in labels_set else 0 for i in range(len(bat))])
        auroc = float(roc_auc_score(y, bat_scores))

        rec = RECORDED[name]
        print(
            f"{name:<9} pinned batch_score={batch_score:.6f} auroc={auroc:.4f} | "
            f"recorded batch_score={rec['batch_score']} auroc={rec['row_auroc']} "
            f"(n={len(bat)}, injected={int(y.sum())})",
            flush=True,
        )


if __name__ == "__main__":
    main()
