//! TabICL's forward pass in candle. The port follows the module map of
//! the 2026-08-11 export evaluation (glossql `reports/`): column
//! SetTransformer (ISAB, QASSMax query scaling) → row transformer
//! (RoPE, CLS tokens) → ICL transformer (train-prefix attention), with
//! the classifier and regressor differing only in config and heads.
//!
//! Grading: `fixtures/` carries torch train-mode forwards at several
//! (T, H, train_size); stage 1 of the fidelity gate is matching them at
//! ~1e-4 fp32. Weights load from `weights/*.safetensors`, verified
//! against `fixtures/DIGESTS`.

pub mod attention; // MHA + the pre-norm block; K/V slicing, no masks
pub mod config;
pub mod embedding; // feature grouping, SkippableLinear, y-encoders
pub mod icl; // ICL transformer, train-prefix attention
pub mod isab; // induced self-attention blocks
pub mod nn; // Linear / LayerNorm / one-hot, torch semantics
pub mod quantile; // regressor read-out: raw quantiles -> bands
pub mod rope;
pub mod row; // row transformer, non-interleaved RoPE, CLS
pub mod ssmax; // QASSMax query scaling — log(train_size) at runtime
pub mod tabicl; // the composed forward
pub mod weights;
