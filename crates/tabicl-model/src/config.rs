//! The checkpoint's own config, written beside the safetensors by
//! `verify/python/convert_weights.py`. The port never hardcodes dimensions —
//! the checkpoint says what it is.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TabIclConfig {
    /// Raw config map as the checkpoint carries it; typed fields are
    /// promoted as the port needs them, not before.
    #[serde(flatten)]
    pub raw: serde_json::Map<String, serde_json::Value>,
}

impl TabIclConfig {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn int(&self, key: &str) -> Option<i64> {
        self.raw.get(key).and_then(|v| v.as_i64())
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.raw.get(key).and_then(|v| v.as_bool())
    }

    pub fn str(&self, key: &str) -> Option<String> {
        self.raw
            .get(key)
            .and_then(|v| v.as_str().map(str::to_owned))
    }
}
