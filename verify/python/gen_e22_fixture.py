"""E2.2 diagnosis fixture: the cause classifier over recorded episodes.

Reads the evidence archive's recorded episode rows (24 cause-labeled
episodes, 3 seeds x 8 cells), rebuilds the leave-one-seed-out folds for
both feature sets (FULL includes the glossary-supplied invariant rate;
BLIND is marginal statistics only), and runs the pinned single-member
TabICLClassifier per fold — the oracle the Rust classifier wrapper is
graded against. The recorded runs used ensemble defaults on MPS; the
comparison printed here is pinned-vs-recorded, the same shape every
other reproduction used.

    uv run python verify/python/gen_e22_fixture.py

Writes fixtures/e22_diagnosis.npz.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(
    os.environ.get(
        "TFMEVAL_RESULTS", ROOT.parent / "tfmeval" / "output" / "results"
    )
)

NUMERIC = ["pct_move", "surprise_z", "max_seg_z", "max_share_delta",
           "viol_rate", "psi_line_amount", "psi_unit_price", "volume_pct",
           "mix_frac", "rate_frac", "rpo_pct"]
BLIND = [f for f in NUMERIC if f != "viol_rate"]
CLASSES = ["artifact", "mix", "none", "rate"]


def episodes() -> list[dict]:
    rows = [json.loads(line) for line in
            open(RESULTS / "e2_2_diagnosis.jsonl")]
    ep = {r["corpus_id"]: r for r in rows if r.get("phase") == "episode"}
    out = sorted(ep.values(), key=lambda r: (r["seed"], r["corpus_id"]))
    assert len(out) == 24, f"expected 24 episodes, got {len(out)}"
    return out


def recorded_calls() -> dict[tuple[str, str], str]:
    rows = [json.loads(line) for line in
            open(RESULTS / "e2_2_diagnosis.jsonl")]
    return {
        (r["features"], r["corpus_id"]): r["pred"]
        for r in rows
        if r.get("phase") == "classify" and r.get("engine") == "tabicl"
    }


def pinned() -> TabICLClassifier:
    return TabICLClassifier(
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        use_amp=False,
        device="cpu",
        random_state=0,
    )


def main() -> None:
    ep = episodes()
    recorded = recorded_calls()
    seeds = sorted({r["seed"] for r in ep})
    arrays: dict[str, np.ndarray] = {}
    for label, feats in (("full", NUMERIC), ("blind", BLIND)):
        hits_pinned = hits_recorded = agree = n = 0
        for fold, held in enumerate(seeds):
            train = [r for r in ep if r["seed"] != held]
            test = [r for r in ep if r["seed"] == held]
            train_x = np.asarray([[r[f] for f in feats] for r in train], float)
            test_x = np.asarray([[r[f] for f in feats] for r in test], float)
            train_y = np.asarray([CLASSES.index(r["cause"]) for r in train])
            test_y = np.asarray([CLASSES.index(r["cause"]) for r in test])

            model = pinned()
            model.fit(train_x, train_y)
            proba = np.asarray(model.predict_proba(test_x), float)
            pred = proba.argmax(axis=1)

            tag = f"{label}_f{fold}"
            arrays[f"{tag}_train_x"] = train_x
            arrays[f"{tag}_train_y"] = train_y
            arrays[f"{tag}_test_x"] = test_x
            arrays[f"{tag}_test_y"] = test_y
            arrays[f"{tag}_proba"] = proba

            hits_pinned += int((pred == test_y).sum())
            n += len(test)
            for r, p in zip(test, pred):
                rec = recorded.get((label, r["corpus_id"]))
                if rec is not None:
                    hits_recorded += int(rec == r["cause"])
                    agree += int(rec == CLASSES[p])
        print(f"{label}: pinned {hits_pinned}/{n} = {hits_pinned / n:.3f}  "
              f"recorded {hits_recorded}/{n} = {hits_recorded / n:.3f}  "
              f"call agreement {agree}/{n}")

    out = ROOT / "verify" / "experiments" / "fixtures" / "e22_diagnosis.npz"
    np.savez_compressed(out, **arrays)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
