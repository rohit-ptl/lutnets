use crate::{
    architectures::{LUTNetBuilder, cnn_iv1::settings::*},
    iterators::*,
    lut_bank_creators::*,
    netcore::*,
    settings::Configuration,
};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

impl LUTNetBuilder for Ci1Settings {
    fn build_net(&self) -> (&'static Configuration, LUTNet) {
        let cfg = get_cfg(self);
        let mut nodes: Vec<Node> = Vec::with_capacity(cfg.derived.network_size);
        let mut offset: usize = 0;
        let (mut dim1, mut dim2, mut dim3) = (cfg.data.dim1, cfg.data.dim2, cfg.data.dim3);
        let mut spanbitcount = cfg.derived.img_bitcount;
        let mut rng_luts = Xoshiro256PlusPlus::from_rng(&mut rand::rng());

        let lut_bank;
        if cfg.network.lut_bank_size > 0 {
            lut_bank = generate_luts(&cfg);
        } else {
            lut_bank = None;
        }

        for i in 0..2 {
            let (len1, len2, len3, hop1, hop2, hop3) = (
                self.layer_span_details[i].0.0,
                self.layer_span_details[i].0.1,
                self.layer_span_details[i].0.2,
                self.layer_span_details[i].1.0,
                self.layer_span_details[i].1.1,
                self.layer_span_details[i].1.2,
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
        (
            cfg,
            LUTNet {
                nodes: nodes,
                input_size_in_bits: cfg.derived.img_bitcount,
                layer_edges: cfg.derived.layer_edges.clone(),
                lut_bank,
                output_embedding: cfg.network.output_embedding.clone(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::architectures::*;

    #[test]
    fn cnn_iv1_created_correctly() {
        let arch_name = "cnn_iv1";
        let architecture = Architecture::from_str(&arch_name)
            .expect("Could not create architecture in span creation test");
        let (cfg, ltnet) = architecture.build();
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
        }

        if let Some(lut_bank) = ltnet.lut_bank {
            for node in ltnet.nodes {
                assert!(lut_bank.contains(&node.lut));
            }
        }
    }
}
