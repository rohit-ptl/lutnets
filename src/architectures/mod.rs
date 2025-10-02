pub mod cnn_iv0 {
    pub mod netimpl;
    pub mod settings;
}
pub mod cnn_iv1 {
    pub mod netimpl;
    pub mod settings;
}
pub mod cnn_iv2 {
    pub mod netimpl;
    pub mod settings;
}
pub mod cnn_iv3 {
    pub mod netimpl;
    pub mod settings;
}

use crate::{lut_bank_creators::*, netcore::LUTNet, settings::Configuration};
use std::str::FromStr;

pub trait LUTNetBuilder {
    fn build_net(&self) -> (&'static Configuration, LUTNet);
}

// In src/architectures/mod.rs

// ... existing code ...

pub enum Architecture {
    CnnIv0,
    CnnIv1,
    CnnIv2,
    CnnIv3,
    Random,
}

impl FromStr for Architecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cnn_iv0" => Ok(Architecture::CnnIv0),
            "cnn_iv1" => Ok(Architecture::CnnIv1),
            "cnn_iv2" => Ok(Architecture::CnnIv2),
            "cnn_iv3" => Ok(Architecture::CnnIv3),
            "random" => Ok(Architecture::Random),
            _ => Err(format!("'{}' is not a valid architecture.", s)),
        }
    }
}

impl Architecture {
    pub fn build(&self) -> (&'static Configuration, LUTNet) {
        match self {
            Architecture::CnnIv0 => {
                let arch_settings = crate::architectures::cnn_iv0::settings::Ci0Settings::new()
                    .expect("Failed to load cnn_iv0/Settings.toml");
                arch_settings.build_net()
            }
            Architecture::CnnIv1 => {
                let arch_settings = crate::architectures::cnn_iv1::settings::Ci1Settings::new()
                    .expect("Failed to load cnn_iv1/Settings.toml");
                arch_settings.build_net()
            }
            Architecture::CnnIv2 => {
                let arch_settings = crate::architectures::cnn_iv2::settings::Ci2Settings::new()
                    .expect("Failed to load cnn_iv2/Settings.toml");
                arch_settings.build_net()
            }
            Architecture::CnnIv3 => {
                let arch_settings = crate::architectures::cnn_iv3::settings::Ci3Settings::new()
                    .expect("Failed to load cnn_iv3/Settings.toml");
                arch_settings.build_net()
            }
            Architecture::Random => {
                let cfg = crate::settings::get_cfg(None);

                let lut_bank = if cfg.network.lut_bank_size > 0 {
                    generate_luts(cfg)
                } else {
                    None
                };
                let ltnet = LUTNet::init_random(
                    cfg.derived.img_bitcount,
                    &cfg.derived.layer_edges,
                    lut_bank,
                    &cfg.network.output_embedding,
                );
                (cfg, ltnet)
            }
        }
    }
}
