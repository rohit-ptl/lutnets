use lutnets::{
    lut_bank_creators::*, netcore::*, settings::*, training_algos::naive_evolutionary::trainer::*,
};
use std::{error::Error, fs::File, io::Read, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    println!("Running Naive Evolutionary Algorithm Example");
    let cfg = get_cfg();
    println!("Data split and size: {:?}", cfg.data.datasplit);

    let lut_bank;
    let model_filename;
    if cfg.network.lut_bank_size > 0 {
        lut_bank = generate_luts(&cfg);
        model_filename = format!(
            "naive_evo_model_lutbank_{}.ltnet",
            cfg.network.lut_bank_size
        );
    } else {
        lut_bank = None;
        model_filename = format!("naive_evo_model_{}.ltnet", cfg.network.lut_bank_size);
    }

    let (databits, labels) = lutnets::dataloader::csv_to_bitvec(cfg)?;
    println!(
        "Read CSV rows: {}, Wall time: {:?}",
        labels.len(),
        start_time.elapsed()
    );
    let mut ltnet;

    if std::path::Path::new(&model_filename).exists() {
        let mut file = File::open(&model_filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        (ltnet, _) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard())?;
        println!("Loaded model from file: {}", &model_filename);
    } else {
        ltnet = LUTNet::init_spanned(lut_bank);
    }
    ltnet.verify_lut_bank_integrity();
    train(
        &mut ltnet,
        &cfg,
        &databits,
        &labels,
        0.001,
        200,
        100000,
        2,
        &model_filename,
    );
    println!("Total time: {:?}", start_time.elapsed());
    Ok(())
}
