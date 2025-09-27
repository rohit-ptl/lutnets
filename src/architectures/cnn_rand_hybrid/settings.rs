use config::{Config, ConfigError, File};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct CiSettings {
    pub layer_span_details: Vec<((usize, usize, usize), (usize, usize, usize))>,
    pub layer_sizes: Vec<usize>,
}

impl CiSettings {
    pub fn new() -> Result<Self, ConfigError> {
        let s = Config::builder()
            .add_source(File::with_name("src/architectures/cnn_rand_hybrid/ciconf"))
            .build()?;
        s.get::<CiSettings>("network")
    }
}

pub fn initialize_ci_cfg() -> CiSettings {
    CiSettings::new().expect("Failed to load cnn_inspired/ciconf.toml")
}
