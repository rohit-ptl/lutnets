use config::{Config, ConfigError, File};
use serde::Deserialize;
use std::sync::OnceLock;

pub static APP_CFG: OnceLock<Configuration> = OnceLock::new();

#[derive(Debug, Deserialize, Clone)]
pub enum DataSplit {
    Train(u32),
    Val(u32),
    Test(u32),
}

#[derive(Debug, Deserialize, Clone)]
pub struct Data {
    pub dim1: usize, // 8 bits to represent each pixel
    pub dim2: usize, // 28 pixels wide
    pub dim3: usize, // 28 pixels tall
    pub train_filepath: String,
    pub test_filepath: String,
    pub datasplit: DataSplit, // Data split and number of rows to be loaded
    pub batch_size: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Network {
    pub output_embedding: Vec<u8>,
    pub layer_sizes: Vec<usize>,
    pub lut_bank_size: usize,
}

#[derive(Debug)]
pub struct DerivedValues {
    pub rows: usize,
    pub cols: usize,
    pub img_bitcount: usize,
    pub data_bitcount: usize,
    pub num_layers: usize,
    pub num_edges: usize,
    pub layer_edges: Vec<usize>,
    pub network_size: usize,
    pub bitvec_size: usize,
    pub bitvec_edges: Vec<usize>,
    pub batch_bitcount: usize,
    pub num_batches: usize,
}

impl DerivedValues {
    pub fn new(data: &Data, network: &Network) -> Self {
        let rows = match data.datasplit {
            DataSplit::Train(r) => r as usize,
            DataSplit::Val(r) => r as usize,
            DataSplit::Test(r) => r as usize,
        };
        let cols = data.dim2 * data.dim3;
        let img_bitcount = data.dim1 * data.dim2 * data.dim3;
        let data_bitcount = rows * img_bitcount;
        let num_layers = network.layer_sizes.len();
        let num_edges = num_layers + 1;

        let mut layer_edges = vec![0];
        let mut cumulative_sum = 0;
        for &size in &network.layer_sizes {
            cumulative_sum += size;
            layer_edges.push(cumulative_sum);
        }

        let network_size: usize = network.layer_sizes.iter().sum();
        let bitvec_size = data.batch_size * (img_bitcount + network_size);
        let bitvec_edges: Vec<usize> = layer_edges.iter().map(|&x| x * data.batch_size).collect();
        let batch_bitcount = data.batch_size * img_bitcount;
        let num_batches = rows / data.batch_size; // If batch size does not divide rows, some data will be ignored
        Self {
            rows,
            cols,
            img_bitcount,
            data_bitcount,
            num_layers,
            num_edges,
            layer_edges,
            network_size,
            bitvec_size,
            bitvec_edges,
            batch_bitcount,
            num_batches,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub data: Data,
    pub network: Network,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let s = Config::builder()
            .add_source(File::with_name("Settings.toml"))
            .build()?;
        s.try_deserialize()
    }
}

#[derive(Debug)]
pub struct Configuration {
    pub data: Data,
    pub network: Network,
    pub derived: DerivedValues,
}

pub fn initialize_app_config() -> Configuration {
    let settings = Settings::new().expect("Failed to load configuration");
    let derived = DerivedValues::new(&settings.data, &settings.network);
    Configuration {
        data: settings.data,
        network: settings.network,
        derived,
    }
}

pub fn get_cfg() -> &'static Configuration {
    APP_CFG.get_or_init(|| initialize_app_config())
}
