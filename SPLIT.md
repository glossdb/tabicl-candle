# Split & donate plan

Status: analysis agreed 2026-09-01 and corrected the same day (see the
determinism record and the "(corrected)" markers). Local execution done
2026-09-01 on branch `workspace-split` — steps 1 and 2, rayon/anyhow/
sha2-hex cleanups, README numerics scope — every suite green on CPU and
Metal after each commit. Open: the four upstream donations (candle-core
argsort fix, hub safetensors, candle-transformers model, candle-examples)
and the post-upstream deletion of the local forward.

Step-1 deviations from the plan below: `load.rs` lives in
`tabicl-model/tests` (it imports only the loader/forward API, despite
sitting in the wrapper-consumers audience list); workspace-root
`fixtures/` survives holding only `DIGESTS` — since the sha2/hex item
landed it is read by the load suite's fixture-time digest gate, never
by the runtime loader.

## Verdict

This repo is a port-verification workbench that shipped as the deliverable.
The oracle-next-to-port structure was right while porting and wrong the day
stage 1 of the fidelity gate passed; nothing was re-cut afterward. Five
separate concerns are interleaved in one crate, three test audiences share
one `tests/` directory, and ops concerns (digests) plus harness math
(readout) leaked into the library. Fix: split into a workspace now, donate
the model later with a clean cut.

## Inventory — five things in one crate

1. **The model port** (~665 lines: `attention`, `isab`, `row`, `icl`,
   `embedding`, `ssmax`, `rope`, `config`, `tabicl`). Internally clean but
   built on the wrong substrate: `nn.rs` hand-rolls Linear/LayerNorm and
   `TensorMap` hand-rolls loading, where candle-nn's `Linear`, `LayerNorm`,
   and `VarBuilder` have identical semantics (activations already use
   candle's `gelu_erf` op; SkippableLinear and OneHotAndLinear are genuinely
   custom and stay). `anyhow` in every public signature. ~185 lines
   (`nn.rs`, most of `weights.rs`) delete on rewrite to candle idiom.
2. **`quantile.rs`** — legitimately model-side (ports tabicl's own torch
   `QuantileToDistribution`), but carries a workaround for the candle CPU
   argsort bug (see donation targets).
3. **Inference preprocessing** (~900 lines: `regressor`, `classifier`,
   `power`, `ensemble`). Runtime code, not test code: the checkpoint only
   performs on inputs normalized exactly as the tabicl Python wrapper
   normalizes them, and TabICL is an in-context learner — "fit" happens at
   serving time on every request. Misnamed as "sklearn reimplementation";
   its real name is TabICL's required input preprocessing (provenance below).
4. **`unsupervised.rs` + `readout.rs`.** The rayon fan-out in
   `score_log_mean` is an antipattern for the DataFusion target: task-level
   rayon over candle's own kernel-level rayon under tokio, all on the global
   pool. The code already computes a `Vec<Task>` decomposition internally —
   the library should expose that and let the caller schedule. The noise
   pre-draw does not die with the rayon dispatch (corrected): any
   decomposition must draw the dummy-noise streams in sequential order to
   match the oracle, so it survives as the task constructor; only the rayon
   fan-out and the CPU/device branch (~15 lines) go. `readout.rs`
   (np.quantile, tie-averaged AUROC, batch
   score) is evaluation-harness math that migrated into the lib; it moves out.
5. **Verification apparatus.** 17 Python scripts, 23 fixture files, 11 test
   files; sha2/hex as *runtime* deps solely to check weight digests at load.
   Digest verification belongs at fixture-generation and container-build
   time; once safetensors live on the hub, hf-hub's content addressing
   covers provenance and the mechanism dies.

Three test audiences, currently mixed:

- **candle** — `fidelity.rs` + forward fixtures: "does the Rust forward
  match torch." Evidence for the upstream contribution.
- **wrapper-crate consumers** — `wrapper.rs`, `classifier.rs`, `power.rs`,
  `load.rs`, quantile fixtures: "does the Rust inference path match
  sklearn's." The inference crate's contract tests.
- **dataraum internal** — `e4.rs`, `e4_ensemble.rs`, `e22.rs`, `e12s3.rs`,
  `bands.rs`, `density.rs`: replay recorded tfmeval experiments on dataraum
  corpora. Private evidence; does not belong in a published crate.

## Target layout

```
tabicl-candle/                    # workspace root
├── crates/
│   ├── tabicl-model/             # donation candidate → candle-transformers
│   │   # pure forward: attention, isab, row, icl, embedding, ssmax,
│   │   # rope, config, quantile — rewritten onto VarBuilder /
│   │   # candle-nn / candle::Result (nn.rs, weights.rs die here)
│   │   └── tests/                # fidelity.rs + forward fixtures
│   └── tabicl-inference/         # stays ours; what DataFusion embeds
│       # preprocess, power, regressor, classifier, ensemble,
│       # unsupervised (task-decomposition API; no rayon, no anyhow,
│       # no sha2/hex, no filesystem-layout assumptions)
│       └── tests/                # wrapper/power/quantile parity fixtures
├── verify/
│   ├── python/                   # today's scripts/: oracle, fixture gen, conversion
│   └── experiments/              # unpublished crate: e4, e22, e12s3, bands,
│                                 # density tests + fixtures + readout.rs
└── weights/                      # gitignored, as now
```

## Sequencing — two steps, not one

1. **Mechanical split**: pure file moves into the workspace, code unchanged,
   one commit, everything green. This alone gives the clear cut and honest
   names.
2. **Candle-idiom rewrite inside `tabicl-model`**: VarBuilder, candle-nn
   primitives, `candle::Result`; delete the argsort workaround once the
   upstream fix lands. The fixture suite is what makes this rewrite safe —
   full regression coverage is the one genuinely good property of the
   current pile.

Donation moment: `tabicl-model` compiles alone on candle idiom with
fixtures green. Its `src/` collapses into
`candle-transformers/src/models/tabicl.rs` nearly verbatim; fixtures become
the PR evidence; `tabicl-inference` switches its dependency to upstream and
the local forward is deleted (not kept in parallel).

## Donation targets

- **candle-core `sort.rs` fix — overtaken by upstream (2026-09-01).**
  The 0.11.0 release carries the bug (`asort` chunks the full storage
  slice from index 0, ignoring `layout.start_offset()`; CUDA slices by
  `contiguous_offsets()`; Metal correct), but candle main fixed it on
  2026-08-14 — PR #3875, merged after the 0.11.0 release — with exactly
  the fix and edge-case test this doc prescribed (slice by
  `contiguous_offsets()`, bail on non-contiguous; `narrow`'d offset-view
  `sort_last_dim` test). Validated against our suite via
  `[patch.crates-io]` on main: offset-view regression test, quantile
  parity, both forwards, and the bands suite (the original exposure) all
  green with the workaround deleted. The `quantile.rs` workaround stays
  until a release ships the fix. Note: an indexing bug — the right
  lesson is adversarial edge-case unit tests, not tolerance tightness.
- **candle-transformers `models/tabicl.rs`.** Groups 1+2 as one file in
  their convention. No tabular model exists there today. Technical
  justification for a native port: QASSMax bakes `log(train_size)` into
  attention scales at runtime, so ONNX tracing is silently wrong whenever
  context size differs from trace time.
- **Hub safetensors.** PR `convert_weights.py`'s output (safetensors +
  config.json) to `jingang/TabICL` (checkpoints are BSD-3-Clause; tabicl
  source likewise permissive). The conversion script and digest pinning then
  die; a packaged server that wants digests computes them at build time.
- **candle-examples `examples/tabicl/`.** hf-hub fetch + forward on a small
  dataset; either inline a minimal impute/scale (the whisper-example
  pattern) or demo on clean data and skip preprocessing.

## Preprocessing provenance (the "sklearn in Rust" question)

Production role: one request = raw table (train rows with labels + query
rows) → fit normalization statistics on train rows (f64, numpy semantics) →
transform to the normalized f32 tensor → forward → read-out mapped back to
original units. Everything except the forward is this code. Only three
pieces are literally sklearn; one is scipy underneath; the rest is
tabicl-custom or numpy semantics.

Feature pipeline (`regressor.rs`):

| Python original | Algorithm | Rust |
|---|---|---|
| `sklearn.impute.SimpleImputer(strategy="mean")` | column mean over non-NaN | `Preprocessor::fit` |
| `UniqueFeatureFilter` (tabicl custom) | drop single-unique-value columns | `Preprocessor::fit` |
| `CustomStandardScaler` (tabicl custom) | z = (x−mean)/(σ_pop+1e-6), clip ±100 | `Preprocessor::fit` |
| `sklearn.preprocessing.PowerTransformer(yeo-johnson, standardize=True)` | see below | `power.rs` |
| `OutlierRemover` (tabicl custom) | two-stage nan-aware mean±4σ_sample, soft clip ±log1p | `Preprocessor::fit`, `soft_clip` |
| y standardization (tabicl wrapper) | f32 cast, σ_pop, small-std guard → 1.0 | `TabIclRegressor::fit` |

Inside PowerTransformer (mostly scipy; ~200 of power.rs's 298 lines are the
optimizer + likelihood):

| Python original | Rust |
|---|---|
| `scipy.stats.yeojohnson` (stable log1p/expm1 forms) | `yeojohnson` |
| `scipy.stats.yeojohnson_llf` + `_log_var` (log-space branches) | `yeojohnson_llf`, `log_var` |
| `scipy.stats.yeojohnson_normmax` (MLE λ, overflow-safe bounds) | `yeojohnson_normmax` |
| `scipy.optimize.fminbound` (`_minimize_scalar_bounded`) | `fminbound` |
| sklearn `_is_constant_feature`, `_handle_zeros_in_scale` | `is_constant_feature`, `PowerStage::fit` |

Classifier (`classifier.rs`): `sklearn.preprocessing.LabelEncoder` (sorted
unique + searchsorted), tabicl's temperature softmax (T=0.9) + renormalize,
`np.argmax` first-max tie-breaking. Ensemble (`ensemble.rs`): tabicl's
`Shuffler` latin-square construction, deliberately with our own RNG (the
diversity, not the identity, is what the ensemble buys).

## Rust library candidates surveyed (2026-09) — and why not, today

- **argmin `BrentOpt`** — the one real candidate; same algorithm family as
  `fminbound` (golden section + parabolic interpolation). Would replace ~90
  graded lines with a framework dependency whose stopping criteria differ
  from scipy's. Not worth it now; the strongest revisit candidate later.
- **statrs / ndarray-stats** — solid, but everything needed from them is
  ~30 lines of arithmetic already written.
- **linfa** (sklearn-alike) — has scalers, no PowerTransformer, no imputer
  parity, and no promise of numpy/scipy numeric equivalence — which is the
  actual requirement during porting.
- **scientificcomputing.rs monthly** (`https://scientificcomputing.rs/monthly/rss.xml`)
  — good radar (burn, CubeCL relevant to the accelerator story); the
  highlighted crates are young v0.x, nothing touching bounded scalar
  minimization, power transforms, or parity preprocessing.

Rationale for keeping the ports during porting: (a) reference parity —
"scipy's Yeo-Johnson to the last branch" is a requirement no general
library will ever promise, in any language; (b) error-budget attribution —
with host-side f64 preprocessing deterministic, any fixture diff localizes
to the device forward instead of smearing across five stages; (c) evidence
transfer — the recorded evaluation numbers went through the sklearn path,
and tight parity lets the port inherit them without rerunning the harness.
Post-split this is revisitable per component: any stage feeding only
tolerance-gated tests (never exact discrete assertions) can take a
dependency, with its variation measured once and folded into the gate.

## Determinism: scope correction (decision record)

Accepted in review 2026-09-01; corrected the same day after checking the
claims against the code and the actual floating-point model. To be
reflected in code, tests, and README:

- **The reference model.** IEEE-754 arithmetic is deterministic: same
  binary, same inputs, same execution order → bit-identical results, every
  run. There is no run-to-run noise; a cross-run flake would be genuine
  nondeterminism, which this code avoids (the rayon fan-out collects in
  task order, all summation orders fixed). Variation lives across
  *environments*, on three axes: backend (CPU/Metal/CUDA), SIMD dispatch
  (candle's gemm picks kernels by CPU features at runtime), and libm —
  only `+ − × ÷ √` are correctly rounded per the standard;
  `ln`/`exp`/`expm1`/`log1p` come from the platform libm and differ in
  last ULPs across libc implementations.
- **Exactness is a pinned-environment instrument, not port-time-only**
  (corrected). Exact-equality assertions are valid wherever fixtures,
  backend, weights, and machine are pinned — and there they are the most
  sensitive regression tripwire available (a tolerance gate hides a 1-ULP
  behavior change; exact asserts don't expire when porting ends). The cut
  is by where a suite runs, not by project phase: exact asserts for
  pinned-environment suites, margin asserts for anything claimed across
  environments. Near-threshold flips across environments cannot be
  prevented by any implementation, ported or off-the-shelf; the danger is
  the illusion of exactness, and the guard is the contract, not the
  library choice.
- **README needs scope wording, not rescoping** (corrected: the earlier
  "overclaims to fix" item overstated). "Zero coverage flips" and "labels
  exactly equal" are already stated against the *pinned* oracle in the
  README, which also already carries the fp32-jitter quantification
  (torch's own fp32 forward is 1.6e-4 from its fp64 reference) and the
  "row NLL is ordinal, not cardinal" note. Remaining work: one paragraph
  stating the environment scope (pinned machine + backend) of the exact
  figures.
- **Stability classes in the `tabicl-inference` API contract.** Continuous
  outputs (bands, probabilities, scores): backend-dependent at ~1e-4,
  compare only with tolerance. Discrete outputs (labels, rankings, coverage
  indicators): environment-dependent near thresholds — consumers must not
  persist, dedupe, or cross-compare expecting equality. Generalizes the
  existing "row NLL is ordinal, not cardinal" note.
- **Margin-based assertions: mostly already in place** (corrected).
  `tests/bands.rs` already tolerates ≤2 edge-flip rows per coverage
  figure; `tests/density.rs` already asserts the pairwise
  never-flip-outside-gap contract. The remaining exact discrete asserts
  are `tests/classifier.rs` (predicted labels) and `tests/e22.rs`
  (accuracy hit counts) — keep them exact as pinned-environment gates,
  documented as such, per the previous bullet.
- **Keep the deterministic host-side f64 layer, scoped to a pinned
  platform — not "across IEEE-754 CPUs"** (corrected). Sequential order
  and no-FMA-contraction hold, but `power.rs` (Yeo-Johnson llf;
  `fminbound` compares llf values, so a ULP flip can shift the chosen λ),
  `readout.rs`, and `quantile.rs` lean on libm transcendentals, which
  vary across libc implementations. Still cheap determinism, still taken
  for error-budget attribution; documented as pinned-toolchain+platform,
  not promised outward.
- **Fleet determinism is a deployment decision.** If a product feature ever
  needs same-input-same-answer across machines, pin one backend — and one
  platform/libm — and say so; numerics will not provide it.

## Open items

- [x] Step 1: mechanical workspace split (pure moves, one commit) —
      2026-09-01, branch `workspace-split`
- [x] Step 2: rewrite `tabicl-model` onto VarBuilder / candle-nn /
      `candle::Result` — 2026-09-01; `TabIcl::new(&config, vb)` is the
      donation-shaped constructor, `from_checkpoint` stays as convenience;
      `nn.rs` keeps only the custom pieces (skip protocol, OneHotAndLinear,
      shape-free loaders); every gate held, exact discrete asserts included
- [x] Drop rayon from `unsupervised.rs`; expose task decomposition instead
      — 2026-09-01; `tasks()`/`run()` public, convenience reads sequential
- [x] Drop anyhow from public APIs; drop sha2/hex from `[dependencies]`
      — 2026-09-01; digest verification moved to the load suite
      (fixture-time gate), sha2/hex are dev-dependencies of tabicl-model
- [x] Move `readout.rs` + E-experiment tests/fixtures to `verify/experiments`
      (part of step 1)
- [x] Add environment-scope wording to README (claims already pinned-scoped;
      see the corrected determinism record) — 2026-09-01: "Numerics: what is
      pinned and what travels" section; stability classes in the
      tabicl-inference crate docs
- [x] candle-core PR: argsort view-offset fix + edge-case test — overtaken
      by upstream #3875 (merged 2026-08-14, post-0.11.0; identical fix and
      test); nothing to submit
- [ ] Delete the `quantile.rs` argsort workaround when a candle release
      ships #3875 (validated 2026-09-01 against main: full suite green
      without it)
- [ ] Hub PR: safetensors + config.json to `jingang/TabICL` — prepared
      2026-09-01: `verify/python/upload_hub_safetensors.py` (dry run by
      default, verifies pinned digests, remote names mirror the source
      ckpt basenames); run with `--create-pr` after `hf auth login`
- [ ] candle-transformers PR: `models/tabicl.rs`; candle-examples: `examples/tabicl/`
- [ ] After upstream lands: delete local forward, depend on candle-transformers
