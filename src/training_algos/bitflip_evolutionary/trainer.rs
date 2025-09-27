use crate::{netcore::*, processing::*, settings::*, utils::*};
use bitvec::prelude::*;
use rand::{prelude::*, seq::index};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{fs::File, io::Write, time::Instant};

pub fn train(
    ltnet: &mut LUTNet,
    cfg: &Configuration,
    databits: &BitVec<u8, Msb0>,
    labels: &[u8],
    corruption_ratio: f32,
    lut_sampling_depth: usize,
    epochs: usize,
    write_freq: usize,
    model_filename: &str,
) {
    let start_time = Instant::now();

    let mut loss_per_batch: Vec<usize> = Vec::with_capacity(cfg.derived.num_batches);
    let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
    let mut steps = 0;
    for batch_num in 0..cfg.derived.num_batches {
        let y = &labels[batch_num * cfg.data.batch_size..(batch_num + 1) * cfg.data.batch_size];
        dbv[..cfg.derived.batch_bitcount].copy_from_bitslice(
            &databits[batch_num * cfg.derived.batch_bitcount
                ..(batch_num + 1) * cfg.derived.batch_bitcount],
        );
        ltnet.apply_gates(&cfg, &mut dbv);
        loss_per_batch.push(get_loss(cfg, &dbv, y));
    }
    print!(
        "Initial losses per batch: {:?}\nCorrupting {} nodes at a time.\n",
        loss_per_batch,
        (cfg.derived.network_size as f32 * corruption_ratio).round() as usize
    );
    for epoch in 0..epochs {
        for batch_num in 0..cfg.derived.num_batches {
            let y = &labels[batch_num * cfg.data.batch_size..(batch_num + 1) * cfg.data.batch_size];
            dbv[..cfg.derived.batch_bitcount].copy_from_bitslice(
                &databits[batch_num * cfg.derived.batch_bitcount
                    ..(batch_num + 1) * cfg.derived.batch_bitcount],
            );
            // loss = loss_per_batch[batch_num];
            let (c_loss, mutated_nodes) = iterate_corruptions(
                ltnet,
                &cfg,
                &mut dbv,
                &y,
                corruption_ratio,
                lut_sampling_depth,
            )
            .unwrap();
            // println!("Min loss from corruptions: {}", c_loss);
            if c_loss < loss_per_batch[batch_num] {
                ltnet.nodes = mutated_nodes;
                println!(
                    "Epoch {}, Batch {}, Previous loss: {}, Improved Loss: {}, Time: {:?}",
                    epoch,
                    batch_num,
                    loss_per_batch[batch_num],
                    c_loss,
                    start_time.elapsed()
                );
                loss_per_batch[batch_num] = c_loss;
                // ltnet.apply_gates(&mut dbv);
                // let predicted_labels = get_labels(&dbv);
                // let accuracy = utils::calculate_accuracy(y, &predicted_labels);
                // println!("Accuracy: {:.2}% (of {})", accuracy * 100.0, cfg.data.batch_size);
                let encoded_bytes: Vec<u8> =
                    bincode::encode_to_vec(&*ltnet, bincode::config::standard()).unwrap();
                let mut file = File::create(model_filename).unwrap();
                file.write_all(&encoded_bytes).unwrap();
                println!("Model written to {}", model_filename);
            }
            steps += 1;
            if steps % 2 == 0 {
                println!(
                    "Epoch {}, Batch {}, Previous loss: {}, Current Loss: {}, Time: {:?}",
                    epoch,
                    batch_num,
                    loss_per_batch[batch_num],
                    c_loss,
                    start_time.elapsed()
                );
            }
        }
    }
}

pub fn iterate_corruptions(
    ltnet: &mut LUTNet,
    cfg: &Configuration,
    dbv: &mut BitVec<u8, Msb0>,
    y: &[u8],
    corruption_ratio: f32,
    iterations: usize,
) -> Option<(usize, Vec<Node>)> {
    assert!(cfg.network.lut_bank_size <=0, "Cannot run this algorithm on network with lut bank.");
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    let num_to_corrupt = (cfg.derived.network_size as f32 * corruption_ratio).round() as usize;
    // println!("Num of nodes to corrupt: {}", num_to_corrupt);
    // let node_idxs_to_corrupt: Vec<usize> = std::iter::repeat_with(|| (rng.next_u64() as usize) % cfg.derived.network_size)
    //     .take(num_to_corrupt)
    //     .collect();
    let node_idxs_to_corrupt: Vec<usize> =
        index::sample(&mut rng, cfg.derived.network_size, num_to_corrupt).into_vec();
    // let mut oloss   = initial_loss;
    let best_result = (0..iterations)
        .into_par_iter()
        .map_init(
            || {
                let pseudo_6bit_generator = pseudo_6bit_generator();
                (pseudo_6bit_generator, dbv.clone())
            },
            |(pseudo_6bit_generator, local_dbv), _| {
                let mutated_nodes = ltnet.apply_gates_with_bitflips(
                    local_dbv,
                    &node_idxs_to_corrupt,
                    pseudo_6bit_generator,
                );
                let loss = get_loss(&cfg, &local_dbv, y);
                // println!("Loss for an iteration: {}", loss);
                (loss, mutated_nodes)
            },
        )
        .min_by_key(|(loss, _)| *loss);
    best_result
}
