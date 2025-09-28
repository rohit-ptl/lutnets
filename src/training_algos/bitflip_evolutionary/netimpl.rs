use crate::{iterators::*, netcore::*, settings::*};
use bitvec::prelude::*;

impl LUTNet {
    pub fn apply_gates_with_bitflips<T, O>(
        &self,
        cfg: &Configuration,
        bv: &mut BitVec<T, O>,
        node_idxs_to_corrupt: &Vec<usize>,
        mut pseudo_6bit_generator: impl Iterator<Item = u8>,
    ) -> Vec<Node>
    where
        T: BitStore,
        O: BitOrder,
    {
        // fewer checks, a little more rodeo version of apply_gates. Adds corruption and returns the mutated nodes.
        // let cfg = initialize_app_config();
        let mut nodes_to_iterate = self.nodes.clone();
        for node_idx in node_idxs_to_corrupt {
            nodes_to_iterate[*node_idx].lut ^= 1 << pseudo_6bit_generator.next().unwrap();
        }
        for layer in 0..cfg.derived.num_layers {
            let gate_iterator = LayerGateIterator::new(&cfg, layer);
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
}
