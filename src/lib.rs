pub mod architectures;
pub mod dataloader;
pub mod iterators;
pub mod lut_bank_creators;
pub mod modelloader;
pub mod netcore;
pub mod processing;
pub mod settings;
pub mod training_algos;
pub mod utils;

pub const HOME_DIR: &'static str = env!("CARGO_MANIFEST_DIR");
