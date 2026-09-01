//! The checkpoint's non-standard pieces and its loading convention.
//! Forward math is candle-nn's (`Linear`, `LayerNorm` at torch
//! defaults: biased variance, eps 1e-5; GELU is the erf form via
//! `gelu_erf`). What stays custom: the skip protocol shared by the
//! embedding and ISAB, torch's OneHotAndLinear, and shape-free loading
//! — the port never hardcodes dimensions, the checkpoint says what it
//! is, so layers read their shapes off the tensors instead of a config.

use candle_core::{D, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

pub const SKIP_VALUE: f64 = -100.0;

/// Torch nn.Linear (weight stored (out, in), bias always present) read
/// off the tensors at `vb`'s prefix.
pub fn linear(vb: &VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get_unchecked("weight")?,
        Some(vb.get_unchecked("bias")?),
    ))
}

/// LayerNorm at torch defaults; the bias follows the checkpoint's
/// tensors (bias_free_ln differs between the shipped checkpoints).
pub fn layer_norm(vb: &VarBuilder) -> Result<LayerNorm> {
    let w = vb.get_unchecked("weight")?;
    Ok(if vb.contains_tensor("bias") {
        LayerNorm::new(w, vb.get_unchecked("bias")?, 1e-5)
    } else {
        LayerNorm::new_no_bias(w, 1e-5)
    })
}

/// SkippableLinear: rows whose inputs are all SKIP_VALUE come out as
/// SKIP_VALUE, everything else is a plain linear.
pub fn skippable_linear(lin: &Linear, x: &Tensor) -> Result<Tensor> {
    let y = lin.forward(x)?;
    // mask: 1.0 where the whole input row equals the skip value
    let dev = (x - SKIP_VALUE)?.abs()?.max_keepdim(D::Minus1)?;
    let mask = dev.eq(0f64)?.to_dtype(candle_core::DType::F32)?;
    let keep = (1.0 - &mask)?;
    y.broadcast_mul(&keep)?
        .broadcast_add(&(mask * SKIP_VALUE)?)
}

/// One-hot against `num_classes` then linear — torch's OneHotAndLinear.
pub fn one_hot_linear(lin: &Linear, y: &Tensor, num_classes: usize) -> Result<Tensor> {
    let classes = Tensor::arange(0f32, num_classes as f32, y.device())?;
    let one_hot = y
        .unsqueeze(D::Minus1)?
        .broadcast_eq(&classes)?
        .to_dtype(candle_core::DType::F32)?;
    lin.forward(&one_hot)
}
