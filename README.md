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
2. **Passed (pinned member).** Wrapper parity: the Rust regressor and
   classifier wrappers match the sklearn wrappers end to end for one
   member (n_estimators=1, norm "none") — preprocessing at f64 machine
   epsilon, regressor predictions ~3e-5 and classifier probabilities
   ~2e-6 against the 5e-4 gate, predicted labels exactly equal.
   Multi-member ensembling is still open.
3. **Passed.** Statistical parity: the read-outs reproduce the
   evaluation-harness numbers on the oracle corpora.

   *Bands.* The E2.1 walk-forward band calibration (374 fits over 44
   metric series, four clean corpora) reproduces from the Rust side:
   the pinned member matches the recorded 8-member ensemble within one
   standard error on every figure, and the Rust wrapper matches the
   pinned sklearn oracle exactly (zero coverage flips):

   | grain | figure | recorded (8-member) | pinned sklearn | Rust |
   |---|---|---|---|---|
   | month (n=272) | coverage80 | 0.445 | 0.474 | 0.474 |
   | month | coverage90 | 0.563 | 0.544 | 0.544 |
   | month | median width80 | 0.139 | 0.141 | 0.141 |
   | segment (n=102) | coverage80 | 0.559 | 0.559 | 0.559 |
   | segment | coverage90 | 0.618 | 0.637 | 0.637 |
   | segment | median width80 | 0.189 | 0.187 | 0.187 |

   *Density.* The E1.2s3 joined-surface anomaly ranking (referential
   integrity broken three ways; 14,928 rows, mixed
   numerical/categorical, the strongest TabICL result in the
   evaluation) reproduces with pinned single-member inner estimators —
   the recorded runs used 4-member inner ensembles (latin shuffles,
   none+power norms, AMP):

   | variant | figure | recorded (4-member) | pinned (1-member) |
   |---|---|---|---|
   | shuffled | row AUROC | 0.9338 | 0.9343 |
   | shuffled | batch score | 0.1449 | 0.1488 |
   | distinct | row AUROC | 0.991 | 0.9932 |
   | distinct | batch score | 0.1125 | 0.1159 |
   | repeated | row AUROC | 0.991 | 0.9932 |
   | repeated | batch score | 0.1125 | 0.1159 |

   The Rust side is graded on a reduced instance of the shuffled
   surface (400-row sampled context, 200-row frames — the full
   protocol's 15k-row fits are a torch-on-MPS job, not a standing CPU
   test): recorded permutations and noise, per-permutation parity, and
   the read-out contract end to end, AUROC agreeing with the pinned
   oracle to 8e-4 (`tests/e12s3.rs`).

   *What-if (the point read).* The E4 counterfactual walk (fine grid,
   coarse grid, two-lever interaction; 21 fits against exact generated
   truth) reproduces from the Rust side at 5.7e-5 max relative against
   the pinned oracle, harness grades matching on every fit
   (`tests/e4.rs`). The pinned-vs-recorded comparison splits the
   ensemble verdict for the first time: on the dense grid (7 factors,
   42 train rows) the pinned member matches the recorded 8-member run
   within thousandths of median APE, but on the sparse grid (3
   factors, 18 rows) the ensemble buys 2-3x lower point error
   (revenue mape 0.0092 recorded vs 0.0294 pinned; gross_profit
   0.0681 vs 0.1305) and consistently better effect recovery. The
   interaction fits match the recorded figures — and both lose to
   additive composition where the true interaction is small, as the
   recorded run already found.

   Verdict: the pinned single member carries every calibration and
   ranking read (bands, density), and point reads on dense support.
   Point reads on sparse support are the one demonstrated case where
   multi-member ensembling earns its cost. The harness (local sibling
   `tfmeval`) stays behind as the evidence archive.

   *The ensemble (the sparse-support point read).* Ruled in
   (2026-08-11): most real metrics stand on few inputs, so sparse
   support is the normal what-if regime. The regressor ensemble is
   ported: the Yeo-Johnson power stage (`power.rs` — lambda MLE via
   the bounded-Brent `fminbound` port, matching sklearn's
   `PowerTransformer(standardize=True)` at 1e-6 on six fixture
   matrices, `tests/power.rs`), the "power" pipeline slot between
   scaling and the outlier stage, per-member feature permutations,
   and quantile averaging across members (`ensemble.rs`). Graded
   against a full-default sklearn rerun of all 21 E4 fits — which
   itself sits on the recorded tfmeval figures to four decimals, so
   the fixture *is* the recorded configuration (4 members on the
   2-feature fits, 6 on the 3-feature; the "8" default truncates at
   shuffles x norms) — member configs injected exactly, final bands
   matching at 2.1e-4 max relative (`tests/e4_ensemble.rs`).
   Production member generation uses this crate's own RNG (latin
   squares crossed with both norms, `EnsembleMember::generate`);
   which permutation a member draws deliberately differs from
   sklearn's Python-`random` selection — the diversity, not the
   identity, is what the ensemble buys. The classifier-side ensemble
   (class shuffles) stays out until a categorical read needs it.

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

The classifier wrapper (`classifier.rs`) shares that preprocessing
plane unchanged (the sklearn `PreprocessingPipeline` is constructed
identically for both tasks) and adds the classification delta: label
encoding (sorted unique classes), the logit slice to the classes
present, and the wrapper's temperature softmax with its final
renormalization, computed host-side in f32 as numpy computes it. With
one member the class shuffle short-circuits to identity exactly like
the feature shuffle, so the ensemble-average and shuffle-correction
steps vanish — the same shape the density read's inner classifiers
have.

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

The density read (`unsupervised.rs`) is ported for mixed surfaces:
chain-rule orchestration over both wrappers — the quantile
distribution's log_prob for numerical conditionals, the classifier's
`predict_proba` lookup for categorical ones (sorted classes, 1e-10
floor for unseen classes, 0.0 for missing observations). Graded per
permutation against the oracle's own `_compute_log_density`: the
numerical-only fixture at ~3e-3 in log space, the mixed fixture at
~5e-3 (gates set by the read's conditioning — log_prob differentiates
the quantile grid, amplifying the forward's ~2e-5 — not by porting
slack), with the score ordering asserted pairwise: oracle pairs
separated by more than the numeric gate never flip. Permutations and
the empty-conditioning noise column are API inputs: the sklearn source
draws them from Python's Mersenne Twister and numpy's Generator, and
nothing semantic rides on those streams — grading replays the recorded
oracle streams.

One finding from the real-surface grading (the E1.2s3 fixture): on
near-deterministic conditionals — amount given amount_inv, where the
conditional distribution is a spike — the row-level log density is
chaotic in any fp32 implementation. The spike's quantile gaps sit at
~1e-4, the same order as legitimate fp32 forward jitter (torch's own
fp32 forward is 1.6e-4 from its fp64 reference there; this port is
1.3e-4 from the same reference), so per-row NLL moves by ~0.3 log
units between equally valid implementations while the rank-based
reads stay put (AUROC shifted 8e-4). Downstream consumers should
treat row NLL on such columns as ordinal, not cardinal. The read-out
contract itself (`readout.rs`: NLL, the q95 reference threshold,
batch score, tie-averaged AUROC) is ported from the harness and
graded in `tests/e12s3.rs`.
