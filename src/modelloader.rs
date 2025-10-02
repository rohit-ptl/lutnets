use crate::{netcore::*, settings::*};
use std::{fs::File, io::Read};

pub fn load_model_from_file(model_filename: &str) -> (&'static Configuration, LUTNet) {
    if !std::path::Path::new(&model_filename).exists() {
        panic!(
            "Model loader failed: No file with name {} exists in the current folder!",
            &model_filename
        );
    }
    let mut file = File::open(model_filename).expect("Model file exists, but unable to open it.");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Model file opened, but unable to read it.");
    let (ltnet, _): (LUTNet, usize) =
        bincode::serde::decode_from_slice(&buffer, bincode::config::standard())
            .expect("Failed to deserialize model.");
    let output_embedding = ltnet.output_embedding.clone();
    let layer_sizes = ltnet
        .layer_edges
        .clone()
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();
    let lut_bank_size = ltnet.lut_bank.as_ref().map(|v| v.len()).unwrap_or(0);
    println!("Loaded model from file: {}", &model_filename);

    let cfg = get_cfg(Some(Network {
        output_embedding,
        layer_sizes,
        lut_bank_size,
    }));
    (cfg, ltnet)
}
