use crate::{iterators::*, netcore::*, settings::*};
use bitvec::prelude::*;
use std::collections::{HashMap, HashSet};

impl LUTNet {
    pub fn apply_gates_while_tracking_pivotal_nodes<T, O>(
        &self,
        cfg: &Configuration,
        bv: &mut BitVec<T, O>,
    ) -> Vec<HashMap<usize, HashSet<usize>>>
    where
        T: BitStore,
        O: BitOrder,
    {
        // fewer checks, a little more rodeo version of apply_gates but tracks which nodes affect which output bits
        let mut pivotal_nodes_sets: Vec<HashMap<usize, HashSet<usize>>> =
            vec![
                (0..cfg.network.layer_sizes[0])
                    .map(|i| (i, HashSet::new()))
                    .collect();
                cfg.data.batch_size
            ];
        let mut next_pivotal_sets: Vec<HashMap<usize, HashSet<usize>>> =
            vec![HashMap::new(); cfg.data.batch_size];

        for layer in 0..cfg.derived.num_layers {
            let mut results = Vec::new();
            let gate_iterator = LayerGateIterator::new(cfg, layer);
            gate_iterator.for_each(
                |(node_idx, img_num_in_batch, _, bitvec_index, readbit_offset)| {
                    let node = self
                        .nodes
                        .get(node_idx)
                        .expect("Error: Invalid node index {}");
                    let indices = node.indices;
                    let lut_input = unsafe {
                        (*bv.get_unchecked(indices[0] as usize + readbit_offset) as u8)
                            | ((*bv.get_unchecked(indices[1] as usize + readbit_offset) as u8) << 1)
                            | ((*bv.get_unchecked(indices[2] as usize + readbit_offset) as u8) << 2)
                            | ((*bv.get_unchecked(indices[3] as usize + readbit_offset) as u8) << 3)
                            | ((*bv.get_unchecked(indices[4] as usize + readbit_offset) as u8) << 4)
                            | ((*bv.get_unchecked(indices[5] as usize + readbit_offset) as u8) << 5)
                    };
                    let output_bit = ((node.lut >> lut_input) & 1) != 0;
                    results.push((bitvec_index, output_bit));
                    if layer > 0 {
                        let mut current_node_deps = HashSet::new();
                        let mut pivotal_node_occurences_local: HashMap<usize, HashSet<usize>> =
                            HashMap::new(); // this set tracks, for all the inputs to the current node, a reverse lookup of which pivotal nodes affect which input bits in cases where nodes affect multiple input bits
                        for i in 0..6 {
                            let curr_input_node_idx =
                                (indices[i] - cfg.derived.img_bitcount) as usize;
                            let mutated_input = lut_input ^ (1 << i);
                            let mutated_output_bit = ((node.lut >> mutated_input) & 1) != 0;
                            let curr_input_node_piv_set = pivotal_nodes_sets[img_num_in_batch]
                                .get(&curr_input_node_idx)
                                .expect("ERROR: Missing pivotal set for node");
                            if mutated_output_bit != output_bit {
                                // If the input is pivotal...
                                current_node_deps.insert(curr_input_node_idx);
                                current_node_deps.extend(curr_input_node_piv_set);
                            }
                            for &item in curr_input_node_piv_set {
                                pivotal_node_occurences_local
                                    .entry(item)
                                    .or_default()
                                    .insert(i);
                            }
                        }
                        pivotal_node_occurences_local.retain(|_number, key_set| key_set.len() > 1);
                        for (piv_node, piv_set) in pivotal_node_occurences_local {
                            let mut mutated_input = lut_input;
                            for i in piv_set {
                                mutated_input ^= 1 << i;
                            }
                            let mutated_output_bit = ((node.lut >> mutated_input) & 1) != 0;
                            if mutated_output_bit == output_bit {
                                current_node_deps.remove(&piv_node);
                            } else {
                                current_node_deps.insert(piv_node);
                            }
                        }
                        next_pivotal_sets[img_num_in_batch].insert(node_idx, current_node_deps);
                    }
                },
            );
            if layer > 0 {
                std::mem::swap(&mut pivotal_nodes_sets, &mut next_pivotal_sets);
                for hm in &mut next_pivotal_sets {
                    hm.clear();
                }
            }
            for (index, new_value) in results {
                bv.set(index, new_value);
            }
        }
        // println!("Pivotal nodes sets: {:?}", pivotal_nodes_sets);
        pivotal_nodes_sets
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dataloader::*, netcore::LUTNet, processing::*, settings::*,
        training_algos::piv_node_flipper::utils::*, utils::*,
    };
    use bitvec::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn pivotal_nodes_tracking_works_correctly() {
        let mut cfg = initialize_app_config();
        cfg.data.batch_size = 20;
        cfg.derived = DerivedValues::new(&cfg.data, &cfg.network);
        let (databits, _labels) = csv_to_bitvec(&cfg).unwrap();
        let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
        dbv[..cfg.derived.batch_bitcount]
            .copy_from_bitslice(&databits[..cfg.derived.batch_bitcount]);
        let ltnet = LUTNet::init_random(cfg.derived.img_bitcount, &cfg.derived.layer_edges, None);
        let pivotal_nodes = ltnet.apply_gates_while_tracking_pivotal_nodes(&cfg, &mut dbv);
        let selected_bits_to_flip = find_most_frequent_pivotal_node(&pivotal_nodes);
        let mut last_layer_location_for_batch_in_bitvec = cfg.derived.bitvec_size
            - cfg.data.batch_size * cfg.network.layer_sizes[cfg.derived.num_layers - 1];
        // println!("selected bits to flip: {:?}", selected_bits_to_flip);
        for (img_num_in_batch, node_to_flip_hashset) in selected_bits_to_flip.iter().enumerate() {
            let mut all_nodes_affected: HashSet<usize> = HashSet::new();
            let mut ltnetclone = ltnet.clone();
            let mut dbvclone = dbv.clone();
            let predicted_label_bitslice = dbv[last_layer_location_for_batch_in_bitvec
                ..last_layer_location_for_batch_in_bitvec
                    + cfg.network.layer_sizes.last().unwrap()]
                .to_owned();
            // println!("Predicted label bitslice for batch {}: {:?}", img_num_in_batch, predicted_label_bitslice);
            for (&node_to_flip, nodes_affected) in node_to_flip_hashset {
                let indices =
                    get_node_indices_in_bitvec(node_to_flip, &ltnet, &cfg, img_num_in_batch);
                let mut bitstr = String::new();
                for index in indices {
                    bitstr.insert_str(
                        0,
                        if *(dbv.get(index)).clone().unwrap() {
                            "1"
                        } else {
                            "0"
                        },
                    );
                }
                // println!("Indices for node {}, in batch {}: {:?}", node_to_flip,  img_num_in_batch, indices);
                // println!("Input for node {} in batch {}: {:?}", node_to_flip, img_num_in_batch, bitstr);
                // println!("Node LUT: {:b}", ltnet.nodes[node_to_flip].lut);
                // dbg!(ltnet.nodes[node_to_flip].lut);
                let idx_in_lut = u8::from_str_radix(&bitstr, 2).unwrap();
                ltnetclone.nodes[node_to_flip].lut ^= 1 << idx_in_lut;
                all_nodes_affected.extend(nodes_affected);
                // println!("Node LUT: {:b}", ltnet.nodes[node_to_flip].lut);
            }

            let _ = ltnetclone.apply_gates(&cfg, &mut dbvclone);
            let bitdiffs = predicted_label_bitslice.clone()
                ^ (&dbvclone[last_layer_location_for_batch_in_bitvec
                    ..last_layer_location_for_batch_in_bitvec
                        + cfg.network.layer_sizes.last().unwrap()]);
            let diff_indices: HashSet<usize> = bitdiffs
                .iter_ones()
                .map(|index| index + cfg.derived.layer_edges[cfg.derived.num_layers - 1])
                .collect();
            assert_eq!(all_nodes_affected, diff_indices);
            last_layer_location_for_batch_in_bitvec += cfg.network.layer_sizes.last().unwrap();
            // println!("non matching bits {:?}",diff_indices);
        }
    }

    #[test]
    fn flipping_pivotal_reduces_loss() {
        let mut cfg = initialize_app_config();
        cfg.data.batch_size = 1;
        cfg.derived = DerivedValues::new(&cfg.data, &cfg.network);
        let (databits, labels) = csv_to_bitvec(&cfg).unwrap();
        let batch_num = 3; //arbitrary batch number
        let y = &labels[batch_num * cfg.data.batch_size..(batch_num + 1) * cfg.data.batch_size];
        let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
        dbv[..cfg.derived.batch_bitcount].copy_from_bitslice(
            &databits[batch_num * cfg.derived.batch_bitcount
                ..(batch_num + 1) * cfg.derived.batch_bitcount],
        );
        let mut ltnet =
            LUTNet::init_random(cfg.derived.img_bitcount, &cfg.derived.layer_edges, None);
        ltnet.apply_gates(&cfg, &mut dbv);
        let pivotal_nodes = ltnet.apply_gates_while_tracking_pivotal_nodes(&cfg, &mut dbv);
        // let last_layer_location_for_batch_in_bitvec = cfg.derived.bitvec_size - cfg.data.batch_size * cfg.network.layer_sizes[cfg.derived.num_layers - 1];
        let loss_vec = get_loss_vec(&cfg, &dbv, y);
        let loss = &loss_vec.count_ones();
        let selected_bits_to_flip = find_pivotal_node_with_scoring(&cfg, &pivotal_nodes, &loss_vec);
        for (img_num_in_batch, &node_to_flip) in selected_bits_to_flip.iter().enumerate() {
            let indices = get_node_indices_in_bitvec(node_to_flip, &ltnet, &cfg, img_num_in_batch);
            let mut bitstr = String::new();
            for index in indices {
                bitstr.insert_str(
                    0,
                    if *(dbv.get(index)).clone().unwrap() {
                        "1"
                    } else {
                        "0"
                    },
                );
            }
            let idx_in_lut = u8::from_str_radix(&bitstr, 2).unwrap();
            ltnet.nodes[node_to_flip].lut ^= 1 << idx_in_lut;
            // break; //important to break. We only want to flip one bit for testing otherwise bit interference can lead to test failure. No longer important to break since we changed cfg to batch size 1.
        }
        ltnet.apply_gates(&cfg, &mut dbv);
        let new_loss = get_loss(&cfg, &dbv, y);
        if *loss > 0usize {
            assert!(*loss >= new_loss);
        }
    }
}
