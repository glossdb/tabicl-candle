//! Row interaction: the first `num_cls` token slots are replaced by the
//! learned CLS tokens (whatever the column stage produced there is
//! discarded), all-but-last blocks are full self-attention over a row's
//! tokens with RoPE, the last block queries only the CLS tokens, and
//! the normalized CLS outputs concatenate into the row representation.

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder};

use crate::attention::{Block, Kv};
use crate::nn::layer_norm;
use crate::rope::Rope;

pub struct RowInteractor {
    blocks: Vec<Block>,
    cls_tokens: Tensor, // (num_cls, E)
    out_ln: LayerNorm,
    rope: Rope,
    pub num_cls: usize,
}

impl RowInteractor {
    pub fn new(num_blocks: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let blocks = (0..num_blocks)
            .map(|i| Block::new(num_heads, false, vb.pp(format!("tf_row.blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let cls_tokens = vb.get_unchecked("cls_tokens")?;
        let num_cls = cls_tokens.dim(0)?;
        Ok(Self {
            blocks,
            cls_tokens,
            out_ln: layer_norm(&vb.pp("out_ln"))?,
            rope: Rope {
                freqs: vb.get_unchecked("tf_row.rope.freqs")?,
            },
            num_cls,
        })
    }

    /// emb: (B, T, G, E) -> (B, T, num_cls * E).
    pub fn forward(&self, emb: &Tensor) -> Result<Tensor> {
        let (b, t, g, e) = emb.dims4()?;
        let cls = self
            .cls_tokens
            .reshape((1, 1, self.num_cls, e))?
            .expand((b, t, self.num_cls, e))?
            .contiguous()?;
        let body = emb.narrow(2, self.num_cls, g - self.num_cls)?;
        let x = Tensor::cat(&[&cls, &body], 2)?;

        // Every row is a batch element; tokens are the sequence.
        let mut x = x.reshape((b * t, g, e))?;
        let (last, rest) = self.blocks.split_last().unwrap();
        for block in rest {
            x = block.forward(&x, Kv::SelfFull, Some(&self.rope))?;
        }
        let q = x.narrow(1, 0, self.num_cls)?.contiguous()?;
        let cls_out = last.forward(&q, Kv::Cross(&x), Some(&self.rope))?;
        let cls_out = self.out_ln.forward(&cls_out)?;
        cls_out.reshape((b, t, self.num_cls * e))
    }
}
