//! Column embedding, train path with feature grouping "same" and
//! affine=false as both shipped checkpoints configure it: features
//! group by circular permutation at offsets 2^i, CLS slots are padded
//! in as the skip value, targets are added to the train prefix of
//! every group, and the set transformer's output IS the embedding.

use candle_core::{D, Tensor};

use crate::isab::Isab;
use crate::nn::{Linear, SKIP_VALUE, TensorMap, one_hot_linear, skippable_linear};

pub struct ColEmbedder {
    in_linear: Linear, // SkippableLinear(group_size -> E)
    blocks: Vec<Isab>,
    y_encoder: Linear,
    pub max_classes: usize,
    pub group_size: usize,
    pub reserve_cls: usize,
    pub embed_dim: usize,
}

impl ColEmbedder {
    pub fn load(
        tm: &TensorMap,
        num_blocks: usize,
        num_heads: usize,
        max_classes: usize,
        group_size: usize,
        reserve_cls: usize,
    ) -> anyhow::Result<Self> {
        let blocks = (0..num_blocks)
            .map(|i| {
                Isab::load(
                    tm,
                    &format!("col_embedder.tf_col.blocks.{i}"),
                    num_heads,
                    true,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let in_linear = tm.linear("col_embedder.in_linear")?;
        let embed_dim = in_linear.w.dim(0)?;
        Ok(Self {
            in_linear,
            blocks,
            y_encoder: tm.linear("col_embedder.y_encoder")?,
            max_classes,
            group_size,
            reserve_cls,
            embed_dim,
        })
    }

    /// x: (B, T, H), y: (B, train_size) -> (B, T, H + reserve_cls, E).
    pub fn forward(&self, x: &Tensor, y: &Tensor) -> anyhow::Result<Tensor> {
        let (b, t, h) = x.dims3()?;
        let train_size = y.dim(1)?;

        // Feature grouping by circular permutation: group g holds
        // features (g + 2^i) % H for i in 0..group_size.
        let mut parts = Vec::with_capacity(self.group_size);
        for i in 0..self.group_size {
            let off = 1usize << i;
            let idx: Vec<u32> = (0..h).map(|g| ((g + off) % h) as u32).collect();
            let idx = Tensor::from_vec(idx, h, x.device())?;
            parts.push(x.index_select(&idx, 2)?.unsqueeze(D::Minus1)?);
        }
        let grouped = Tensor::cat(&parts.iter().collect::<Vec<_>>(), D::Minus1)?; // (B, T, H, gs)

        // CLS slots padded in front of the group axis at the skip value.
        let cls_pad = Tensor::full(
            SKIP_VALUE as f32,
            (b, t, self.reserve_cls, self.group_size),
            x.device(),
        )?;
        let grouped = Tensor::cat(&[&cls_pad, &grouped], 2)?; // (B, T, G, gs)
        let g = h + self.reserve_cls;

        // (B, G, T, gs) flattened: every group is a batch element.
        let features =
            grouped
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b * g, t, self.group_size))?;

        let src = skippable_linear(&self.in_linear, &features)?; // (B*G, T, E)

        // Target embedding added to the train prefix of every group.
        let y_g = y
            .unsqueeze(1)?
            .expand((b, g, train_size))?
            .contiguous()?
            .reshape((b * g, train_size))?;
        let y_emb = if self.max_classes > 0 {
            one_hot_linear(&self.y_encoder, &y_g, self.max_classes)?
        } else {
            self.y_encoder.forward(&y_g.unsqueeze(D::Minus1)?)?
        };
        let src_train = (src.narrow(1, 0, train_size)? + y_emb)?;
        let src_rest = src.narrow(1, train_size, t - train_size)?;
        let mut src = Tensor::cat(&[&src_train, &src_rest], 1)?;

        for block in &self.blocks {
            src = block.forward(&src, train_size)?;
        }

        // affine=false: the transformer output is the embedding.
        Ok(src
            .reshape((b, g, t, self.embed_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }
}
