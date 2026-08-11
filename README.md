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

The Python side needs torch + tabicl. Either materialize this repo's
own env (`uv sync`) or, while the sibling harness is around, run
through it:

```bash
uv run --project ../tfmeval python scripts/convert_weights.py
uv run --project ../tfmeval python scripts/gen_fixtures.py
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

1. Golden fixtures: Rust forward matches the torch train-mode forward
   at multiple (T, H, train_size), ~1e-4 fp32, CPU and Metal.
2. Wrapper parity: preprocessing + ensembling pinned to one member,
   Rust wrapper matches the sklearn wrapper's outputs.
3. Statistical parity: the tfmeval read-outs (regressor quantile bands,
   density ranking) reproduce the harness numbers on the oracle corpora.

## Status

Skeleton. Conversion and fixtures work; the port itself is not started.
Read-out priority from the evaluation: regressor (what-if bands) and
the unsupervised density (join-suspect ranking) first, classifier after.
