use bitvec::prelude::*;
use lutnets::{architectures::*, processing::*, utils::calculate_accuracy};
use std::{error::Error, str::FromStr, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    // This file simply shows creating a new network, and manually running a forward pass on a single batch
    let start_time = Instant::now();
    println!("LUTNets Reference Example");
    // let cfg = get_cfg(None);
    let arch_name = "random";
    let architecture = Architecture::from_str(arch_name)?;
    let (cfg, ltnet) = architecture.build();

    println!(
        "Network initialized. Nodes: {}, Wall time: {:?}",
        cfg.derived.network_size,
        start_time.elapsed()
    );

    let (databits, labels) = lutnets::dataloader::csv_to_bitvec(cfg)?;
    println!(
        "Read {} rows, Wall time: {:?}",
        labels.len(),
        start_time.elapsed()
    );

    // Let's create a bit container to hold calculations then process a single batch
    let batch_num = 0; // Le't handle just one batch in this file.
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

    ltnet.apply_gates(cfg, &mut dbv);
    println!(
        "Forward pass completed. Wall time: {:?}",
        start_time.elapsed()
    );

    let (predicted_labels, loss) = (get_labels(cfg, &dbv), get_loss(cfg, &dbv, y));
    let accuracy = calculate_accuracy(y, &predicted_labels);
    // println!("Predicted labels: {:?}, Loss: {:?}", predicted_labels, loss);
    println!(
        "\n\nStatisitcs on the first batch of data:\n Batch size: {}\n Loss: {:?}\n Accuracy: {:.1}%\n Total time: {:?}",
        cfg.data.batch_size,
        loss,
        accuracy * 100.0,
        start_time.elapsed()
    );
    Ok(())
}
