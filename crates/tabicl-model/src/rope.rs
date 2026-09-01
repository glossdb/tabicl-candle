//! Non-interleaved rotary embedding, as the row transformer uses it
//! (`row_rope_interleaved: false`): the head dim splits into contiguous
//! halves, freqs come from the checkpoint (`rope.freqs`, length
//! head_dim/2) and cos/sin are tiled `[f, f]` across the full head dim.

use candle_core::{Result, Tensor};

pub struct Rope {
    /// Inverse frequencies from the checkpoint, shape (head_dim / 2).
    pub freqs: Tensor,
}

impl Rope {
    /// t: (batch, heads, len, head_dim) — rotates along the len axis
    /// with positions 0..len, exactly `rotate_queries_or_keys`.
    pub fn apply(&self, t: &Tensor) -> Result<Tensor> {
        let (_, _, len, hd) = t.dims4()?;
        let half = hd / 2;
        let pos = Tensor::arange(0f32, len as f32, t.device())?;
        // outer product (len, half)
        let ang = pos.unsqueeze(1)?.broadcast_mul(&self.freqs.unsqueeze(0)?)?;
        let cos = ang.cos()?;
        let sin = ang.sin()?;
        let cos2 = Tensor::cat(&[&cos, &cos], 1)?.reshape((1, 1, len, hd))?;
        let sin2 = Tensor::cat(&[&sin, &sin], 1)?.reshape((1, 1, len, hd))?;
        let x1 = t.narrow(3, 0, half)?;
        let x2 = t.narrow(3, half, half)?;
        let rot = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
        Ok(t.broadcast_mul(&cos2)?
            .broadcast_add(&rot.broadcast_mul(&sin2)?)?)
    }
}
