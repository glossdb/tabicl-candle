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
2. Wrapper parity: preprocessing + ensembling pinned to one member,
   Rust wrapper matches the sklearn wrapper's outputs.
3. Statistical parity: the read-outs (regressor quantile bands, density
   ranking) reproduce the evaluation-harness numbers on the oracle
   corpora. The read-out code moves into this repo at that stage; the
   harness (local sibling `tfmeval`) stays behind as the evidence
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
*after* wrapper parity, not beside it. Next: stage 2 — the
preprocessing + ensembling wrapper, pinned to one member first.
