use lutnets::{netcore::*, settings::*, training_algos::piv_node_flipper::trainer::*};
use std::{error::Error, fs::File, io::Read, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    println!("Running Naive Evolutionary Algorithm Example");
    let cfg = get_cfg();
    println!("Data split and size: {:?}", cfg.data.datasplit);
    let model_filename = "piv_node_desc_model.ltnet";
    let (databits, labels) = lutnets::dataloader::csv_to_bitvec(cfg)?;
    println!(
        "Read CSV rows: {}, Wall time: {:?}",
        labels.len(),
        start_time.elapsed()
    );
    let mut ltnet;

    if std::path::Path::new(model_filename).exists() {
        let mut file = File::open(model_filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        (ltnet, _) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard())?;
    } else {
        ltnet = LUTNet::init_spanned(None);
    }

    train(&mut ltnet, &cfg, &databits, &labels, 100, 1, model_filename);
    println!("Total time: {:?}", start_time.elapsed());
    Ok(())
}
