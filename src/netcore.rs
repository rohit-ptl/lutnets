use crate::{iterators::*, settings::*};
use bincode::Encode;
use bitvec::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Serialize, Deserialize, Debug, Copy, Clone, Encode)]
pub struct Node {
    // a node will have a LUT and 6 input indices
    pub lut: u64,
    pub indices: [usize; 6],
}
#[derive(Serialize, Deserialize, Debug, Clone, Encode)]
pub struct LUTNet {
    pub nodes: Vec<Node>,
    pub input_size_in_bits: usize, // This is the size, in bits, of each input item to the network. 784 for MNIST
    pub layer_edges: Vec<usize>,   // indices in nodes vector where each layer starts
    pub lut_bank: Option<Vec<u64>>, // If we want a network with finite lut bank for each node to pick from
}

impl LUTNet {
    pub fn init_random(
        input_size_in_bits: usize,
        layer_edges: &Vec<usize>,
        lut_bank: Option<Vec<u64>>,
    ) -> Self {
        // This is a highly random and pretty impractical initialization. A slightly better (but still pretty bad) one is init_spanned in architectures/cnn_inspired/netimpl.rs

        let &network_size = layer_edges.last().unwrap();
        let mut nodes = Vec::with_capacity(network_size);
        (0..network_size)
            .into_par_iter()
            .map_with(
                || Xoshiro256PlusPlus::from_rng(&mut rand::rng()),
                |rng, node_index| {
                    let layer_index = layer_edges[1..].partition_point(|&edge| edge <= node_index);
                    // println!("layer_index for node {:?}: {}", layer_edges, layer_index);
                    let (range_start, range_end) = if layer_index == 0 {
                        (0, input_size_in_bits)
                    } else {
                        let start = input_size_in_bits + layer_edges[layer_index - 1];
                        let end = input_size_in_bits + layer_edges[layer_index];
                        (start, end)
                    };
                    // println!("range for node {}: {} to {}", node_index, range_start, range_end);
                    let mut indices = [0usize; 6];
                    for i in 0..6 {
                        indices[i] = rng().random_range(range_start..range_end);
                    }
                    Node {
                        lut: if let Some(lut_bank) = &lut_bank {
                            lut_bank[rng().random_range(0..lut_bank.len())]
                        } else {
                            rng().next_u64()
                        },
                        indices,
                    }
                },
            )
            .collect_into_vec(&mut nodes);

        LUTNet {
            nodes: nodes,
            input_size_in_bits,
            layer_edges: layer_edges.clone(),
            lut_bank: None,
        }
    }

    pub fn apply_gates<T, O>(&self, cfg: &Configuration, bv: &mut BitVec<T, O>)
    where
        T: BitStore,
        O: BitOrder,
    {
        // This is sort of like the forward pass of the network. It applies all the gates in sequence to the provided BitVec.
        let bv_len = bv.len();
        for layer in 0..cfg.derived.num_layers {
            // We need to apply gates to layers in sequence.
            let gate_iterator = LayerGateIterator::new(cfg, layer);
            let results: Vec<(usize, bool)> = gate_iterator
                // .par_bridge()  // Tha par_bridge here works, but slows things down since work in each thread is too little.
                .filter_map(|(node_idx, _, _, bitvec_index, readbit_offset)| {
                    if let Some(node) = self.nodes.get(node_idx) {
                        let indices = &node.indices;
                        if indices.iter().any(|&idx| idx as usize >= bv_len) {
                            println!("indices: {:?}, bvlen: {}", indices, bv_len);
                            panic! {"Invalid read index in node at index {}", node_idx};
                        }
                        // // Safe version of the code below
                        // let lut_input =
                        //     (*bv.get(indices[0] + readbit_offset).unwrap() as u8) |
                        //     ((*bv.get(indices[1] + readbit_offset).unwrap() as u8) << 1) |
                        //     ((*bv.get(indices[2] + readbit_offset).unwrap() as u8) << 2) |
                        //     ((*bv.get(indices[3] + readbit_offset).unwrap() as u8) << 3) |
                        //     ((*bv.get(indices[4] + readbit_offset).unwrap() as u8) << 4) |
                        //     ((*bv.get(indices[5] + readbit_offset).unwrap() as u8) << 5);

                        let lut_input = unsafe {
                            (*bv.get_unchecked(indices[0] as usize + readbit_offset) as u8)
                                | ((*bv.get_unchecked(indices[1] as usize + readbit_offset) as u8)
                                    << 1)
                                | ((*bv.get_unchecked(indices[2] as usize + readbit_offset) as u8)
                                    << 2)
                                | ((*bv.get_unchecked(indices[3] as usize + readbit_offset) as u8)
                                    << 3)
                                | ((*bv.get_unchecked(indices[4] as usize + readbit_offset) as u8)
                                    << 4)
                                | ((*bv.get_unchecked(indices[5] as usize + readbit_offset) as u8)
                                    << 5)
                        };

                        // println!("LUT input: {:?}", lut_input);
                        let output_bit = ((node.lut >> lut_input) & 1) != 0;
                        if bitvec_index < bv_len {
                            Some((bitvec_index, output_bit))
                        } else {
                            panic!(
                                "Write index {} out of bounds for BitVec of length {}",
                                bitvec_index, bv_len
                            );
                        }
                    } else {
                        panic!("Node fetching failure at index {}", node_idx);
                    }
                })
                .collect();
            // println!("Processed {} nodes in parallel.", results.len());

            for (index, new_value) in results {
                bv.set(index, new_value);
            }
        }
    }

    pub fn apply_gates_with_new_luts<T, O>(
        &self,
        cfg: &Configuration,
        bv: &mut BitVec<T, O>,
        node_idxs_to_mutate: &Vec<usize>,
        new_luts: &Vec<u64>,
    ) -> Vec<Node>
    where
        T: BitStore,
        O: BitOrder,
    {
        // fewer checks, a little more rodeo version of apply_gates. This one will overwrite LUTs for node_ids with provided LUTs and return the mutated nodes. Can be handy in seeing impact of mutations.
        let mut nodes_to_iterate = self.nodes.clone();
        for (node_idx, lut) in node_idxs_to_mutate.iter().zip(new_luts.iter()) {
            nodes_to_iterate[*node_idx].lut = *lut;
        }
        for layer in 0..cfg.derived.num_layers {
            let gate_iterator = LayerGateIterator::new(cfg, layer);
            let results: Vec<(usize, bool)> = gate_iterator
                // .par_bridge()  // Tha par_bridge here works, but slows things down since work in each thread is too little.
                .filter_map(|(node_idx, _, _, bitvec_index, readbit_offset)| {
                    let node = &nodes_to_iterate[node_idx];
                    let indices = node.indices;
                    let lut_input = unsafe {
                        (*bv.get_unchecked(indices[0] as usize + readbit_offset) as u8)
                            | ((*bv.get_unchecked(indices[1] as usize + readbit_offset) as u8) << 1)
                            | ((*bv.get_unchecked(indices[2] as usize + readbit_offset) as u8) << 2)
                            | ((*bv.get_unchecked(indices[3] as usize + readbit_offset) as u8) << 3)
                            | ((*bv.get_unchecked(indices[4] as usize + readbit_offset) as u8) << 4)
                            | ((*bv.get_unchecked(indices[5] as usize + readbit_offset) as u8) << 5)
                    };

                    // println!("LUT input: {:?}", lut_input);
                    let output_bit = ((node.lut >> lut_input) & 1) != 0;
                    Some((bitvec_index, output_bit))
                })
                .collect();
            for (index, new_value) in results {
                bv.set(index, new_value);
            }
        }
        nodes_to_iterate
    }

    pub fn verify_lut_bank_integrity(&self) {
        if let Some(lut_bank) = &self.lut_bank {
            let lut_set: HashSet<_> = lut_bank.iter().collect(); //faster lookups
            for (idx, node) in self.nodes.iter().enumerate() {
                assert!(
                    lut_set.contains(&node.lut),
                    "Failed at node {}, LUT value '{}' was not found in the LUT bank.",
                    idx,
                    node.lut
                );
            }
            println!("LUTNet integrity verified.");
        } else {
            println!("LUTNet doesn't have lut_bank.");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{dataloader::*, lut_bank_creators::*, netcore::LUTNet, settings::*};
    use bitvec::prelude::*;
    use rand::Rng;

    #[test]
    fn random_lutnet_created_correctly() {
        let cfg = get_cfg();
        let lut_bank = generate_luts(&cfg);
        let ltnet = LUTNet::init_random(
            cfg.derived.img_bitcount,
            &cfg.derived.layer_edges,
            lut_bank.clone(),
        );

        for layer in 0..cfg.derived.num_layers {
            let layer_nodes =
                &ltnet.nodes[cfg.derived.layer_edges[layer]..cfg.derived.layer_edges[layer + 1]];
            let min_index = layer_nodes.iter().flat_map(|node| node.indices).min();
            let max_index = layer_nodes.iter().flat_map(|node| node.indices).max();
            assert!(
                min_index.unwrap()
                    >= (cfg.derived.layer_edges[layer.saturating_sub(1)]
                        + cfg.derived.img_bitcount)
                        * std::cmp::min(layer, 1)
            );
            assert!(max_index.unwrap() < cfg.derived.layer_edges[layer] + cfg.derived.img_bitcount);
            // println!("Layer {}: min_index={}, max_index={}", layer, min_index2.unwrap(), max_index2.unwrap());
        }
        if let Some(lut_bank) = lut_bank {
            for node in ltnet.nodes {
                assert!(lut_bank.contains(&node.lut));
            }
        }
    }

    #[test]
    fn gate_application_works_correctly() {
        let cfg = get_cfg();
        let (databits, _) = csv_to_bitvec(cfg).unwrap();
        // println!("DATA_BITS: {}, cfg.derived.batch_bitcount: {}, cfg.derived.bitvec_size: {}", cfg.derived.data_bitcount, cfg.derived.batch_bitcount, cfg.derived.bitvec_size);
        let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
        dbv[..cfg.derived.batch_bitcount]
            .copy_from_bitslice(&databits[..cfg.derived.batch_bitcount]);
        let ltnet = LUTNet::init_random(cfg.derived.img_bitcount, &cfg.derived.layer_edges, None);
        ltnet.apply_gates(cfg, &mut dbv);
        let mut rng = rand::rng();
        let some_node_ids: Vec<(usize, usize, usize)> = (0..cfg.derived.num_layers)
            .map(|i| {
                (
                    i,
                    rng.random_range(cfg.derived.layer_edges[i]..cfg.derived.layer_edges[i + 1]),
                    rng.random_range(0..cfg.data.batch_size),
                )
            })
            .collect();
        // println!("Some random node ids to test: {:?}", some_node_ids);
        for (layer, node, img_num) in &some_node_ids {
            let test_node = ltnet.nodes[*node].clone();
            let bv_offset = if *layer == 0 {
                img_num * cfg.derived.img_bitcount
            } else {
                (cfg.data.batch_size - 1)
                    * (cfg.derived.img_bitcount + cfg.derived.layer_edges[*layer - 1])
                    + img_num * cfg.network.layer_sizes[*layer - 1]
            };
            let bv_index = cfg.data.batch_size
                * (cfg.derived.img_bitcount + cfg.derived.layer_edges[*layer])
                + img_num * cfg.network.layer_sizes[*layer]
                + (node - cfg.derived.layer_edges[*layer]);
            // println!("Layer: {}, Node: {}, Batch: {}, Offset: {}, BV index: {}", layer, node, img_num, bv_offset, bv_index);
            let mut bitstr = String::new();
            for val in test_node.indices {
                bitstr.insert_str(
                    0,
                    if *(dbv.get(val as usize + bv_offset)).clone().unwrap() {
                        "1"
                    } else {
                        "0"
                    },
                );
                // println!("  Input bit: {:?}", *(dbv.get(val as usize + bv_offset)).clone().unwrap());
            }
            let lut1 = test_node.lut;
            let idx_in_lut = u8::from_str_radix(&bitstr, 2).unwrap();
            let bit_at_location = *dbv.get(bv_index).unwrap();
            // println!("  LUT: {:b}",  lut1);
            // println!("  LUT: {:b}",  lut1 >>idx_in_lut);
            // println!("  Output bit: {:?}, bitstr: {}, num {}", bit_at_location, bitstr, idx_in_lut);
            assert_eq!(bit_at_location, ((lut1 >> idx_in_lut) & 1 != 0));
        }
    }
}
