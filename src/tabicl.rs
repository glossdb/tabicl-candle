//! The composed train-mode forward: column embedding -> row interaction
//! -> in-context learning, the exact path `gen_fixtures.py` pins. The
//! constructor checks the checkpoint config against everything this
//! port hardcodes — a checkpoint configured differently must refuse to
//! load, never silently miscompute.

use candle_core::Tensor;

use crate::embedding::ColEmbedder;
use crate::icl::IclPredictor;
use crate::nn::TensorMap;
use crate::row::RowInteractor;
use crate::weights::Checkpoint;

pub struct TabIcl {
    col: ColEmbedder,
    row: RowInteractor,
    icl: IclPredictor,
}

impl TabIcl {
    pub fn from_checkpoint(ckpt: Checkpoint) -> anyhow::Result<Self> {
        let cfg = &ckpt.config;
        for (key, expected) in [
            ("col_feature_group", "same"),
            ("col_ssmax", "qassmax-mlp-elementwise"),
            ("icl_ssmax", "qassmax-mlp-elementwise"),
            ("activation", "gelu"),
        ] {
            let got = cfg.str(key);
            anyhow::ensure!(
                got.as_deref() == Some(expected),
                "unsupported checkpoint: {key} = {got:?}, this port implements {expected:?}"
            );
        }
        // bias_free_ln is deliberately not checked: LayerNorm bias
        // presence follows the checkpoint's tensors (the regressor is
        // bias-free, the classifier is not).
        for (key, expected) in [
            ("col_affine", false),
            ("col_target_aware", true),
            ("row_rope_interleaved", false),
            ("norm_first", true),
        ] {
            let got = cfg.bool(key);
            anyhow::ensure!(
                got == Some(expected),
                "unsupported checkpoint: {key} = {got:?}, this port implements {expected:?}"
            );
        }
        let int = |key: &str| -> anyhow::Result<usize> {
            cfg.int(key)
                .map(|v| v as usize)
                .ok_or_else(|| anyhow::anyhow!("checkpoint config missing {key}"))
        };

        let max_classes = int("max_classes")?;
        let reserve_cls = int("row_num_cls")?;
        let tm = TensorMap(ckpt.tensors);
        Ok(Self {
            col: ColEmbedder::load(
                &tm,
                int("col_num_blocks")?,
                int("col_nhead")?,
                max_classes,
                int("col_feature_group_size")?,
                reserve_cls,
            )?,
            row: RowInteractor::load(&tm, int("row_num_blocks")?, int("row_nhead")?)?,
            icl: IclPredictor::load(&tm, int("icl_num_blocks")?, int("icl_nhead")?, max_classes)?,
        })
    }

    /// x: (B, T, H) with train rows first, y: (B, train_size).
    /// Returns (B, T - train_size, out_dim): raw quantiles for the
    /// regressor, all max_classes logits for the classifier.
    pub fn forward(&self, x: &Tensor, y: &Tensor) -> anyhow::Result<Tensor> {
        let emb = self.col.forward(x, y)?;
        let repr = self.row.forward(&emb)?;
        self.icl.forward(&repr, y)
    }
}
