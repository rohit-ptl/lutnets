#![allow(warnings)]
use crate::{
    architectures::cnn_iv0::netimpl::*, netcore::*, processing::*, settings::*,
    training_algos::piv_node_flipper::utils::*, utils::*,
};
use bitvec::prelude::*;
use bitvec::prelude::*;
use rand::prelude::*;
use rand::seq::index;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Read, Write},
};
use std::{error::Error, sync::OnceLock, time::Instant};

pub fn train(
    ltnet: &mut LUTNet,
    cfg: &Configuration,
    databits: &BitVec<u8, Msb0>,
    labels: &[usize],
    epochs: usize,
    write_freq: usize,
    model_filename: &str,
) {
    let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
    let mut loss = usize::MAX;
    let mut steps: usize = 0;
    for epoch in 0..epochs {
        for batch_num in 0..cfg.derived.num_batches {
            let y = &labels[batch_num * cfg.data.batch_size..(batch_num + 1) * cfg.data.batch_size];
            dbv[..cfg.derived.batch_bitcount].copy_from_bitslice(
                &databits[batch_num * cfg.derived.batch_bitcount
                    ..(batch_num + 1) * cfg.derived.batch_bitcount],
            );
            let pivotal_nodes = ltnet.apply_gates_while_tracking_pivotal_nodes(cfg, &mut dbv);
            let mut last_layer_location_for_batch_in_bitvec = cfg.derived.bitvec_size
                - cfg.data.batch_size * cfg.network.layer_sizes[cfg.derived.num_layers - 1];
            let loss_vec = get_loss_vec(cfg, &dbv, y);
            let loss = &loss_vec.count_ones();
            let node_to_flip = find_global_pivotal_node(cfg, &pivotal_nodes, &loss_vec).unwrap();
            // let: usize = selected_bits_to_flip[1].unwrap();
            // ltnet.nodes[node_to_flip].lut = !ltnet.nodes[node_to_flip].lut;
            let mut lut_inputs_for_node_to_flip: Vec<u8> = Vec::with_capacity(cfg.data.batch_size);
            for img_num_in_batch in 0..cfg.data.batch_size {
                let node_indices_in_bitvec =
                    get_node_indices_in_bitvec(node_to_flip, &ltnet, cfg, img_num_in_batch);
                let idx_in_lut = (*dbv.get(node_indices_in_bitvec[0]).unwrap() as u8)
                    | ((*dbv.get(node_indices_in_bitvec[1]).unwrap() as u8) << 1)
                    | ((*dbv.get(node_indices_in_bitvec[2]).unwrap() as u8) << 2)
                    | ((*dbv.get(node_indices_in_bitvec[3]).unwrap() as u8) << 3)
                    | ((*dbv.get(node_indices_in_bitvec[4]).unwrap() as u8) << 4)
                    | ((*dbv.get(node_indices_in_bitvec[5]).unwrap() as u8) << 5);
                // // Slower way of getting idx_in_lut
                // let mut bitstr = String::new();
                // for index in node_indices_in_bitvec {bitstr.insert_str(0, if *(dbv.get(index)).clone().unwrap() {"1"} else {"0"},);}
                // let idx_in_lut = u8::from_str_radix(&bitstr, 2).unwrap();
                lut_inputs_for_node_to_flip.push(idx_in_lut);
            }
            let lut_scores_for_node_to_flip = score_lut_by_node_presence(
                cfg,
                &pivotal_nodes,
                &loss_vec,
                &lut_inputs_for_node_to_flip,
                node_to_flip,
            );
            println!(
                "LUT scores for node {}: {:?}",
                node_to_flip, lut_scores_for_node_to_flip
            );
            for (lut, score) in lut_scores_for_node_to_flip {
                if score > 0 {
                    ltnet.nodes[node_to_flip].lut ^= 1 << lut;
                    // break;
                }
            }
            ltnet.apply_gates(cfg, &mut dbv);
            let new_loss = get_loss(cfg, &dbv, y);
            println!(
                "Batch {} Loss before flipping: {:?}, New loss: {}, Improvement: {}",
                batch_num,
                loss,
                new_loss,
                *loss as i32 - new_loss as i32
            );
            steps += 1;
            if steps % write_freq == 0 {
                let encoded_bytes: Vec<u8> =
                    bincode::encode_to_vec(&*ltnet, bincode::config::standard()).unwrap();
                let mut file = File::create(model_filename).unwrap();
                file.write_all(&encoded_bytes).unwrap();
                println!("Model written to {}", model_filename);
            }
        }
    }
}
