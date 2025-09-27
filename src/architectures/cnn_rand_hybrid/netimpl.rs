use crate::{architectures::cnn_rand_hybrid::settings::*, iterators::*, netcore::*, settings::*};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

impl LUTNet {
    pub fn init_spanew(lut_bank: Option<Vec<u64>>) -> Self {
        let cfg = initialize_app_config();
        let ci_cfg = initialize_ci_cfg();
        assert_eq!(cfg.network.layer_sizes, ci_cfg.layer_sizes);
        if let Some(lut_bank) = &lut_bank {
            assert_eq!(cfg.network.lut_bank_size, lut_bank.len());
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(cfg.derived.network_size);
        let mut offset: usize = 0;
        let (mut dim1, mut dim2, mut dim3) = (cfg.data.dim1, cfg.data.dim2, cfg.data.dim3);
        let mut spanbitcount = cfg.derived.img_bitcount;
        let mut rng_luts = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        // println!("{}",spanbitcount);
        for i in 0..2 {
            let (len1, len2, len3, hop1, hop2, hop3) = (
                ci_cfg.layer_span_details[i].0.0,
                ci_cfg.layer_span_details[i].0.1,
                ci_cfg.layer_span_details[i].0.2,
                ci_cfg.layer_span_details[i].1.0,
                ci_cfg.layer_span_details[i].1.1,
                ci_cfg.layer_span_details[i].1.2,
            );
            // println!("offset:{}, dim1:{},dim2:{},dim3:{}",offset,dim1,dim2,dim3);
            let spiterator =
                SpanGenerator::new(offset, dim1, dim2, dim3, len1, len2, len3, hop1, hop2, hop3);
            offset += spanbitcount;
            spanbitcount = spiterator.len();
            nodes.extend(
                spiterator
                    .map(|indices| Node {
                        lut: if let Some(lut_bank) = &lut_bank {
                            lut_bank[rng_luts.random_range(0..lut_bank.len())]
                        } else {
                            rng_luts.next_u64()
                        },
                        indices,
                    })
                    .collect::<Vec<Node>>(),
            );
            (dim1, dim2, dim3) = (
                (dim1 - len1 + 2 * hop1 - 1) / hop1,
                (dim2 - len2 + 2 * hop2 - 1) / hop2,
                (dim3 - len3 + 2 * hop3 - 1) / hop3,
            );
            // println!("nd1: {}, nd2: {}, nd3: {}", dim1,dim2,dim3);
            // println!("Total spans generated: {}", spanbitcount);
        }
        // println!("offset before last:{}, dim1:{},dim2:{},dim3:{}",offset,dim1,dim2,dim3);
        // let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::random()).unwrap();
        let mut rng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::from_rng(&mut rand::rng());

        for node_index in cfg.derived.layer_edges[2]..cfg.derived.network_size {
            let layer_index =
                cfg.derived.layer_edges[1..].partition_point(|&edge| edge <= node_index);

            let (range_start, range_end) = if layer_index == 0 {
                (0, cfg.derived.img_bitcount)
            } else {
                let start = cfg.derived.img_bitcount + cfg.derived.layer_edges[layer_index - 1];
                let end = cfg.derived.img_bitcount + cfg.derived.layer_edges[layer_index];
                (start, end)
            };

            let mut indices = [0usize; 6];
            for i in 0..6 {
                indices[i] = rng.random_range(range_start..range_end);
            }

            let node = Node {
                lut: if let Some(lut_bank) = &lut_bank {
                    // lut_bank[rng.gen_range(0..lut_bank.len())]
                    lut_bank[rng.random_range(0..lut_bank.len())]
                } else {
                    rng.next_u64()
                },
                indices,
            };

            nodes.push(node);
        }
        println!("{:?}, {}", nodes.len(), cfg.derived.network_size);
        // let luts = generate_LUTs();
        LUTNet {
            nodes: nodes,
            input_size_in_bits: cfg.derived.img_bitcount,
            layer_edges: cfg.derived.layer_edges.clone(),
            lut_bank,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{lut_bank_creators::*, netcore::LUTNet, settings::*};

    #[test]
    fn spanew_lutnet_created_correctly() {
        let cfg = get_cfg();
        let lut_bank = generate_luts(&cfg);
        let ltnet = LUTNet::init_spanew(lut_bank.clone());
        for layer in 0..cfg.derived.num_layers {
            let layer_nodes =
                &ltnet.nodes[cfg.derived.layer_edges[layer]..cfg.derived.layer_edges[layer + 1]];
            let min_index = layer_nodes.iter().flat_map(|node| node.indices).min();
            let max_index = layer_nodes.iter().flat_map(|node| node.indices).max();
            assert_eq!(
                min_index.unwrap(),
                (cfg.derived.layer_edges[layer.saturating_sub(1)] + cfg.derived.img_bitcount)
                    * std::cmp::min(layer, 1)
            );
            assert_eq!(
                max_index.unwrap(),
                cfg.derived.layer_edges[layer] + cfg.derived.img_bitcount - 1
            );
            // println!("Layer {}: min_index={}, max_index={}", layer, min_index2.unwrap(), max_index2.unwrap());
        }

        if let Some(lut_bank) = lut_bank {
            for node in ltnet.nodes {
                assert!(lut_bank.contains(&node.lut));
            }
        }
    }
}
