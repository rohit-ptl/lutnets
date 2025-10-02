use crate::{
    architectures::{LUTNetBuilder, cnn_iv3::settings::*},
    iterators::*,
    lut_bank_creators::*,
    netcore::*,
    settings::Configuration,
};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Extends a vector of nodes from an iterator that yields node indices.
fn extend_nodes_from_iterator<I>(
    nodes: &mut Vec<Node>,
    iterator: I,
    lut_bank: &Option<Vec<u64>>,
    rng_luts: &mut Xoshiro256PlusPlus,
) where
    I: Iterator<Item = [usize; 6]>,
{
    nodes.extend(iterator.map(|indices| Node {
        lut: if let Some(lut_bank) = lut_bank {
            lut_bank[rng_luts.random_range(0..lut_bank.len())]
        } else {
            rng_luts.next_u64()
        },
        indices,
    }));
}

impl LUTNetBuilder for Ci3Settings {
    fn build_net(&self) -> (&'static Configuration, LUTNet) {
        let cfg = get_cfg(self);
        let mut nodes: Vec<Node> = Vec::with_capacity(cfg.derived.network_size);
        let mut rng_luts = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
        let lut_bank = if cfg.network.lut_bank_size > 0 {
            generate_luts(cfg)
        } else {
            None
        };
        for &[offset, dim1, dim2, dim3, len1, len2, len3, hop1, hop2, hop3] in
            &self.layer_span_details
        {
            let spiterator =
                SpanGenerator::new(offset, dim1, dim2, dim3, len1, len2, len3, hop1, hop2, hop3);
            // println!("Nodes len{:?}, spiterator len: {}",nodes.len(), spiterator.len());
            extend_nodes_from_iterator(&mut nodes, spiterator, &lut_bank, &mut rng_luts);
        }

        let offset = cfg.derived.img_bitcount + cfg.derived.layer_edges[4];
        // println!("Offset: {}", offset);
        for _ in 0..cfg.derived.output_bitsize {
            let headgen2 = HeadHelperGenerator::new(offset, 36, self.layer_sizes[4]);
                        // println!("Nodes len {:?}, spiterator len: {}",nodes.len(), headgen2.len());
            extend_nodes_from_iterator(&mut nodes, headgen2, &lut_bank, &mut rng_luts);

        }
        let offset = cfg.derived.img_bitcount + cfg.derived.layer_edges[5];
                // println!("Offset: {}", offset);

        let headgen1 = SlidingGenerator::new(offset, self.layer_sizes[6], self.layer_sizes[5], 6);
                        // println!("Nodes leeen {:?}, spiterator len: {}",nodes.len(), headgen1.len());

        extend_nodes_from_iterator(&mut nodes, headgen1, &lut_bank, &mut rng_luts);

           let offset = cfg.derived.img_bitcount + cfg.derived.layer_edges[6];
                //    println!("Offset: {}", offset);

        let headgen0 = SlidingGenerator::new(offset, self.layer_sizes[7], self.layer_sizes[6], 6);
                        // println!("Nodes leeen {:?}, spiterator len: {}",nodes.len(), headgen0.len());

        extend_nodes_from_iterator(&mut nodes, headgen0, &lut_bank, &mut rng_luts);
        // println!("Offset {}, Num nodes: {:?}, Network len: {}", offset, nodes.len(), cfg.derived.network_size);
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
    fn cnn_iv3_created_correctly() {
        let arch_name = "cnn_iv3";
        let architecture = Architecture::from_str(arch_name)
            .expect("Could not create architecture in span creation test");
        let (cfg, ltnet) = architecture.build();
        assert_eq!(cfg.derived.network_size, ltnet.nodes.len());
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
