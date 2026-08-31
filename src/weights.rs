//! Weight loading: safetensors under `weights/`, digest-verified
//! against the committed `fixtures/DIGESTS`. Weights are never in git
//! and never fetched at container start — a container image bakes them
//! in at build time (repo README, weights policy).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

pub struct Checkpoint {
    pub tensors: HashMap<String, Tensor>,
    pub config: crate::config::TabIclConfig,
}

/// `which` is "classifier" or "regressor". `root` is this repo's
/// layout: weights under `weights/`, digests at `fixtures/DIGESTS`.
pub fn load(root: &Path, which: &str, device: &Device) -> anyhow::Result<Checkpoint> {
    load_from(
        &root.join("weights"),
        &root.join("fixtures/DIGESTS"),
        which,
        device,
    )
}

/// Deployment layout: one flat directory holding the safetensors, the
/// config json, and a `DIGESTS` file — what a consuming server ships
/// (and a container bakes) without carrying this repo's shape.
pub fn load_dir(dir: &Path, which: &str, device: &Device) -> anyhow::Result<Checkpoint> {
    load_from(dir, &dir.join("DIGESTS"), which, device)
}

fn load_from(
    dir: &Path,
    digests: &Path,
    which: &str,
    device: &Device,
) -> anyhow::Result<Checkpoint> {
    let st = dir.join(format!("tabicl-{which}.safetensors"));
    verify_digest(digests, which, &st)?;
    let tensors = candle_core::safetensors::load(&st, device)?;
    let config =
        crate::config::TabIclConfig::load(&dir.join(format!("tabicl-{which}.config.json")))?;
    Ok(Checkpoint { tensors, config })
}

/// Baked-in layout: the safetensors and config ride the consuming
/// binary itself (`include_bytes!`), verified against the pinned
/// digests by that binary's build — there is no file left to verify
/// at load time.
pub fn load_bytes(
    safetensors: &[u8],
    config_json: &str,
    device: &Device,
) -> anyhow::Result<Checkpoint> {
    let tensors = candle_core::safetensors::load_buffer(safetensors, device)?;
    let config = crate::config::TabIclConfig::from_json(config_json)?;
    Ok(Checkpoint { tensors, config })
}

fn verify_digest(digests: &Path, which: &str, st: &PathBuf) -> anyhow::Result<()> {
    let digests: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(digests)?)?;
    let expected = digests[which]["sha256"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("no pinned digest for {which}"))?;
    let bytes = std::fs::read(st)?;
    let actual = sha256_hex(&bytes);
    anyhow::ensure!(
        actual == expected,
        "weights digest mismatch for {which}: expected {expected}, got {actual} — \
         re-run scripts/convert_weights.py or restore the pinned weights"
    );
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    // Tiny local SHA-256 to keep the dependency surface flat is not
    // worth it — candle already pulls enough. This uses the same crate
    // the dev-dependencies use.
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}
