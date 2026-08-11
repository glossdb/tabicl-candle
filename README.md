# dataraum-tabicl

TabICL served from Rust: a hand-port of the TabICL forward pass to
[candle](https://github.com/huggingface/candle), with the Python side
that grades it. One repo, one unit — the port, the weight conversion,
the golden fixtures, and the fidelity suite live together so the whole
thing can be published as one piece.

Why a hand-port and not ONNX: a traced graph bakes `log(train_size)`
into the attention scales — silently wrong the moment the context size
differs from trace time, and for an in-context model the context IS the
data. Full evaluation with probes:
`../glossql/reports/2026-08-11-tabicl-export-evaluation.md`.

## Layout

```
src/            the candle port (crate tabicl-candle)
scripts/        Python side, torch is the oracle:
                  convert_weights.py   ckpt -> safetensors + digest
                  gen_fixtures.py      golden forwards at several (T, H, train)
fixtures/       committed golden fixtures — Rust tests run against these
                without any Python present
weights/        local only, gitignored (see weights policy)
```

## Environments

The Python side needs torch + tabicl; the repo carries its own env:

```bash
uv sync
uv run python scripts/convert_weights.py
uv run python scripts/gen_fixtures.py
```

The Rust side never imports Python. `cargo test` uses the committed
fixtures; tests that need converted weights skip with a message when
`weights/` is absent.

## Weights policy

- Checkpoints (jingang/TabICL v2, ~110 MB each) are never in git.
- `convert_weights.py` converts the torch checkpoints to safetensors
  under `weights/`, pins sha256 digests into `fixtures/DIGESTS`
  (committed), and the loader verifies them.
- Local runs: weights are cached under `weights/` once and reused.
- **Containers bake the weights in.** An image build runs the
  conversion (or copies a converted `weights/`) at build time — a
  container never fetches weights at start.

## Fidelity gate (in order, each blocks the next)

1. **Passed.** Golden fixtures: the Rust forward matches the torch
   train-mode forward on every fixture, both checkpoints, CPU and
   Metal (`cargo test --features metal`), max |diff| ~2e-5 fp32
   against the 1e-4 gate.
2. **Passed (pinned member).** Wrapper parity: the Rust regressor
   wrapper matches the sklearn wrapper end to end for one member
   (n_estimators=1, norm "none") — preprocessing at f64 machine
   epsilon, predictions ~3e-5 against the 5e-4 gate. Multi-member
   ensembling is still open.
3. **Bands half passed.** Statistical parity: the read-outs reproduce
   the evaluation-harness numbers on the oracle corpora. The E2.1
   walk-forward band calibration (374 fits over 44 metric series, four
   clean corpora) reproduces from the Rust side: the pinned member
   matches the recorded 8-member ensemble within one standard error on
   every figure, and the Rust wrapper matches the pinned sklearn oracle
   exactly (zero coverage flips):

   | grain | figure | recorded (8-member) | pinned sklearn | Rust |
   |---|---|---|---|---|
   | month (n=272) | coverage80 | 0.445 | 0.474 | 0.474 |
   | month | coverage90 | 0.563 | 0.544 | 0.544 |
   | month | median width80 | 0.139 | 0.141 | 0.141 |
   | segment (n=102) | coverage80 | 0.559 | 0.559 | 0.559 |
   | segment | coverage90 | 0.618 | 0.637 | 0.637 |
   | segment | median width80 | 0.189 | 0.187 | 0.187 |

   Verdict: multi-member ensembling is not needed for band calibration.
   The density-ranking half (E1.2s3, joined-surface AUROC) waits on the
   classifier wrapper — its surface includes categorical conditionals.
   The harness (local sibling `tfmeval`) stays behind as the evidence
   archive.

## Status

The full train-mode forward is ported and passes fidelity stage 1 for
both checkpoints: column SetTransformer (ISAB, QASSMax), row
transformer (non-interleaved RoPE, CLS tokens), ICL transformer
(train-prefix attention), heads. The checkpoints differ in one config
bit — the regressor's LayerNorms are bias-free, the classifier's are
not — the loader follows the tensors.

The quantile read-out (`quantile.rs`) is ported and graded against the
model's own `QuantileToDistribution`: monotone quantiles bit-identical,
bands (`icdf`) at ~6e-8, `mean` at ~1e-7, `log_prob` at ~6e-5.

One ordering discovery from reading the source: the unsupervised
density (`TabICLUnsupervised.score_samples`) is not a separate model —
it is chain-rule orchestration over the sklearn wrappers (per
conditional: `fit` + `predict(raw_quantiles)` + `log_prob`, or
`predict_proba` for categorical columns). So the density read comes
*after* wrapper parity, not beside it.

The wrapper (`regressor.rs`) is ported for the pinned member: mean
imputation, the unique-value filter, standard scaling with clipping,
the two-stage outlier soft clip, and y standardization — fit
statistics host-side in f64 exactly as numpy computes them, the model
seeing the f32 cast. The sklearn inference path was verified against
source to be numerically the train-mode forward for regression (the
InferenceManager only chunks batch dims), which is why the ported
forward slots in directly.

Stage 3 surfaced a real bug the earlier stages could not see: candle
0.9's CPU argsort ignores a view's start offset (candle-core
`sort.rs`, `asort` — the permutation comes from the wrong storage
window while gather reads the right one; Metal applies the offset
correctly). The narrow'd forward output is exactly such a view, and
`.contiguous()` is a no-op on it. On in-distribution data the damage
hid below the fp gates because quantile outputs are nearly sorted;
the extrapolating band fits exposed it at ~0.8 in scaled space.
`QuantileDist` now materializes a zero-offset copy before sorting,
and the suite pins that guarantee with a regression test.

The density read (`unsupervised.rs`) is ported for numerical columns:
chain-rule orchestration over the wrapper, graded per permutation
against the oracle's own `_compute_log_density` at ~3e-3 in log space
(the gate is set by the read's conditioning — log_prob differentiates
the quantile grid, amplifying the forward's ~2e-5 — not by porting
slack), with the score *ranking* asserted to match exactly.
Permutations and the empty-conditioning noise column are API inputs:
the sklearn source draws them from Python's Mersenne Twister and
numpy's Generator, and nothing semantic rides on those streams —
grading replays the recorded oracle streams. Still open: the
categorical conditional (needs the classifier wrapper's
`predict_proba`), and multi-member ensembling if the stage-3 numbers
need it.
