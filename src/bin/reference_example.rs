use bitvec::prelude::*;
use lutnets::{netcore::*, processing::*, settings::*};
use std::{error::Error, fs::File, io::Read, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    println!("LUTNets Reference Example");
    let cfg = get_cfg();
    let model_filename = "naive_evo_model.ltnet";
    let (databits, labels) = lutnets::dataloader::csv_to_bitvec(cfg)?;
    println!(
        "Read {} rows, Wall time: {:?}",
        labels.len(),
        start_time.elapsed()
    );

    // Let's create a bit container to hold calculations then process a single batch
    let batch_num = 0;
    let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
    let y = &labels[batch_num * cfg.data.batch_size..(batch_num + 1) * cfg.data.batch_size];
    dbv[..cfg.derived.batch_bitcount].copy_from_bitslice(
        &databits
            [batch_num * cfg.derived.batch_bitcount..(batch_num + 1) * cfg.derived.batch_bitcount],
    );
    println!(
        "BitVec created. Capacity: {}, Length: {}, Wall time: {:?}",
        dbv.capacity(),
        dbv.len(),
        start_time.elapsed()
    );

    let ltnet;
    if std::path::Path::new(model_filename).exists() {
        let mut file = File::open(model_filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        (ltnet, _) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard())?;
        println!("Loaded model from file: {}", model_filename);
    } else {
        ltnet = LUTNet::init_random(cfg.derived.img_bitcount, &cfg.derived.layer_edges, None);
    }

    println!(
        "Network initialized. Nodes: {}, Wall time: {:?}",
        cfg.derived.network_size,
        start_time.elapsed()
    );

    ltnet.apply_gates(cfg, &mut dbv);
    println!(
        "Forward pass completed. Wall time: {:?}",
        start_time.elapsed()
    );

    let (_predicted_labels, loss) = (get_labels(cfg, &dbv), get_loss(cfg, &dbv, y));
    // println!("Predicted labels: {:?}, Loss: {:?}", predicted_labels, loss);
    println!(
        "Initial Loss: {:?}, Total time: {:?}",
        loss,
        start_time.elapsed()
    );
    Ok(())
}
