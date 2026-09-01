//! Weight loading: safetensors under the workspace's `weights/`.
//! Weights are never in git and never fetched at container start — a
//! container image bakes them in at build time (repo README, weights
//! policy). Provenance is verified where the bytes enter, not on every
//! load: `convert_weights.py` pins sha256 digests into
//! `fixtures/DIGESTS`, the load suite checks the local weights against
//! them at test time, and a packaged build verifies before baking
//! (`load_bytes` carries bytes the build already checked).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Result, Tensor};

pub struct Checkpoint {
    pub tensors: HashMap<String, Tensor>,
    pub config: crate::config::TabIclConfig,
}

/// `which` is "classifier" or "regressor". `root` is this repo's
/// layout: weights under `weights/`.
pub fn load(root: &Path, which: &str, device: &Device) -> Result<Checkpoint> {
    load_dir(&root.join("weights"), which, device)
}

/// Deployment layout: one flat directory holding the safetensors and
/// the config json — what a consuming server ships (and a container
/// bakes) without carrying this repo's shape.
pub fn load_dir(dir: &Path, which: &str, device: &Device) -> Result<Checkpoint> {
    let st = dir.join(format!("tabicl-{which}.safetensors"));
    let tensors = candle_core::safetensors::load(&st, device)?;
    let config =
        crate::config::TabIclConfig::load(&dir.join(format!("tabicl-{which}.config.json")))?;
    Ok(Checkpoint { tensors, config })
}

/// Baked-in layout: the safetensors and config ride the consuming
/// binary itself (`include_bytes!`), verified against the pinned
/// digests by that binary's build.
pub fn load_bytes(safetensors: &[u8], config_json: &str, device: &Device) -> Result<Checkpoint> {
    let tensors = candle_core::safetensors::load_buffer(safetensors, device)?;
    let config = crate::config::TabIclConfig::from_json(config_json)?;
    Ok(Checkpoint { tensors, config })
}
