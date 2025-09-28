use config::{Config, ConfigError, File};
use serde::Deserialize;
use std::path::PathBuf;

use crate::settings::{APP_CFG, Configuration, Network, initialize_app_config_with_network};

#[derive(Debug, Deserialize, Clone)]
pub struct Ci0Settings {
    pub output_embedding: Vec<usize>,
    pub layer_sizes: Vec<usize>,
    pub lut_bank_size: usize,
    pub layer_span_details: Vec<((usize, usize, usize), (usize, usize, usize))>,
}

impl Ci0Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let mut settings_path = PathBuf::from(manifest_dir);
        settings_path.push("src/architectures/cnn_iv0/Settings.toml");

        let s = Config::builder()
            .add_source(File::from(settings_path))
            .build()?;
        s.get::<Ci0Settings>("network")
    }
}

pub fn get_aligned_cfg(ci_cfg: &Ci0Settings) -> Configuration {
    // let ci_cfg = initialize_ci_cfg();
    let network = Network {
        output_embedding: ci_cfg.output_embedding.clone(),
        layer_sizes: ci_cfg.layer_sizes.clone(),
        lut_bank_size: ci_cfg.lut_bank_size,
    };
    initialize_app_config_with_network(Some(network))
}

pub fn get_cfg(ci_cfg: &Ci0Settings) -> &'static Configuration {
    APP_CFG.get_or_init(|| get_aligned_cfg(ci_cfg))
}
