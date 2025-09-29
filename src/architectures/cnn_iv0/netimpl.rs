use crate::{
    architectures::{LUTNetBuilder, cnn_iv0::settings::*},
    iterators::*,
    lut_bank_creators::*,
    netcore::*,
    settings::Configuration,
};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

impl LUTNetBuilder for Ci0Settings {
    fn build_net(&self) -> (&'static Configuration, LUTNet) {
        let cfg = get_cfg(self);
        let mut nodes: Vec<Node> = Vec::with_capacity(cfg.derived.network_size);
        let mut offset: usize = 0;
        let (mut dim1, mut dim2, mut dim3) = (cfg.data.dim1, cfg.data.dim2, cfg.data.dim3);
        let mut spanbitcount = cfg.derived.img_bitcount;
        let mut rng_luts = Xoshiro256PlusPlus::from_rng(&mut rand::rng());

        let lut_bank = if cfg.network.lut_bank_size > 0 {
            generate_luts(cfg)
        } else {
            None
        };

        for i in 0..self.layer_span_details.len() {
            let (len1, len2, len3, hop1, hop2, hop3) = (
                self.layer_span_details[i].0.0,
                self.layer_span_details[i].0.1,
                self.layer_span_details[i].0.2,
                self.layer_span_details[i].1.0,
                self.layer_span_details[i].1.1,
                self.layer_span_details[i].1.2,
            );
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
        let last_layer_iterator = SpanGenerator::new(0, 27, 1, 1, 6, 1, 1, 3, 1, 1); // bunch of hardcoded values, FIX later!
        nodes.extend(
            last_layer_iterator
                .map(|span| Node {
                    lut: if let Some(lut_bank) = &lut_bank {
                        lut_bank[rng_luts.random_range(0..lut_bank.len())]
                    } else {
                        rng_luts.next_u64()
                    },
                    indices: span.map(|x| (x % spanbitcount) + offset),
                })
                .collect::<Vec<Node>>(),
        );
        // println!("{:?}, {}",nodes.len(), cfg.derived.network_size);
        // let luts = generate_LUTs();
        (
            cfg,
            LUTNet::new(
                nodes,
                cfg.derived.img_bitcount,
                cfg.derived.layer_edges.clone(),
                lut_bank,
                cfg.network.output_embedding.clone(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::architectures::Architecture;
    use std::str::FromStr;
    #[test]
    fn cnn_iv0_created_correctly() {
        let arch_name = "cnn_iv0";
        let architecture = Architecture::from_str(arch_name)
            .expect("Could not create architecture in span creation test");
        let (cfg, ltnet) = architecture.build();
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

        if let Some(lut_bank) = ltnet.lut_bank {
            for node in ltnet.nodes {
                assert!(lut_bank.contains(&node.lut));
            }
        }
    }
}
