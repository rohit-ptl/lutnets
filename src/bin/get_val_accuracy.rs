use bitvec::prelude::*;
use lutnets::{netcore::*, processing::*, settings::*, utils::*};
use std::{env, error::Error, fs::File, io::Read, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    println!("Getting Validating accuracy");

    // Initialize and overwrite the default config to read validation data
    let mut cfg = initialize_app_config();
    let val_size: u32 = 10000;
    // cfg.data.datasplit = DataSplit::Train(val_size);
    cfg.data.datasplit = DataSplit::Val(val_size);
    cfg.data.batch_size = val_size as usize;
    cfg.derived = DerivedValues::new(&cfg.data, &cfg.network);
    let model_filename = &args[1][..];
    let (databits, labels) = lutnets::dataloader::csv_to_bitvec(&cfg)?;

    let batch_num = 0;
    let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
    let y = &labels[batch_num * cfg.data.batch_size..(batch_num + 1) * cfg.data.batch_size];
    dbv[..cfg.derived.batch_bitcount].copy_from_bitslice(
        &databits
            [batch_num * cfg.derived.batch_bitcount..(batch_num + 1) * cfg.derived.batch_bitcount],
    );

    let ltnet: LUTNet;
    if std::path::Path::new(model_filename).exists() {
        let mut file = File::open(model_filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        (ltnet, _) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard())?;
        println!("Loaded model from file: {}", model_filename);
    } else {
        panic!("Cannot find model file {}", model_filename);
    }

    println!(
        "Network initialized. Nodes: {}, Wall time: {:?}",
        cfg.derived.network_size,
        start_time.elapsed()
    );

    ltnet.apply_gates(&cfg, &mut dbv);
    // println!("Forward pass completed. Wall time: {:?}", start_time.elapsed());

    let (predicted_labels, loss) = (get_labels(&cfg, &dbv), get_loss(&cfg, &dbv, y));
    let accuracy = calculate_accuracy(y, &predicted_labels);
    // let predicted_embeddings = get_predicted_embedding(&cfg, &dbv);
    // let true_embeddings = y
    //     .iter()
    //     .map(|&val| cfg.network.output_embedding[val as usize])
    //     .collect::<Vec<u8>>();

    // let pred_lab_embed = predicted_labels
    //     .iter()
    //     .map(|&val| cfg.network.output_embedding[val as usize])
    //     .collect::<Vec<u8>>();

    // for i in 0..50 {
    //     let (pe, te, ple) = (
    //         predicted_embeddings[i],
    //         true_embeddings[i],
    //         pred_lab_embed[i],
    //     );
    //     let diff = pe ^ te;
    //     if te != ple {
    //         println!(
    //             "{:3}: {:8b}\n{:3}: {:8b}\n{:3}: {:8b}\n{:3}: {:8b}\n",
    //             te,
    //             te,
    //             pe,
    //             pe,
    //             ple,
    //             ple,
    //             diff.count_ones(),
    //             diff
    //         );
    //     }
    // }
    // println!("Predicted labels: {:?}, Loss: {:?}", predicted_labels, loss);
    println!(
        "Validation Loss: {:?}, Validation Accuracy: {}, Total time: {:?}",
        loss,
        accuracy,
        start_time.elapsed()
    );
    Ok(())
}
