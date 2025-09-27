use crate::{netcore::*, settings::*};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn calculate_accuracy(y: &[u8], predicted_labels: &[u8]) -> f32 {
    assert_eq!(
        y.len(),
        predicted_labels.len(),
        "Input slices must have the same length."
    );
    if y.is_empty() {
        return 0.0;
    }
    let correct_predictions = y
        .iter()
        .zip(predicted_labels.iter())
        .filter(|&(true_label, predicted_label)| true_label == predicted_label)
        .count();
    correct_predictions as f32 / y.len() as f32
}

pub fn get_ix(cfg: &Configuration, target: u8) -> Option<u8> {
    // Given target embedding, return the corresponding index (which is the label)
    cfg.network
        .output_embedding
        .iter()
        .position(|&x| x == target)
        .map(|index| index as u8)
}

pub fn get_node_indices_in_bitvec(
    node_idx: usize,
    ltnet: &LUTNet,
    cfg: &Configuration,
    img_num_in_batch: usize,
) -> [usize; 6] {
    // Given a node_id in LUTNet and and which image in batch, return the location of its 6 inputs in bitvec
    let node = ltnet.nodes[node_idx];
    let layer = cfg
        .derived
        .layer_edges
        .partition_point(|&val| val < node_idx)
        .saturating_sub(1);
    // println!("Layer: {}, Node idx: {}", layer, node_idx);
    let bv_offset = if layer == 0 {
        img_num_in_batch * cfg.derived.img_bitcount
    } else {
        (cfg.data.batch_size - 1) * (cfg.derived.img_bitcount + cfg.derived.layer_edges[layer - 1])
            + img_num_in_batch * cfg.network.layer_sizes[layer - 1]
    };
    // let bv_index = cfg.data.batch_size
    //     * (cfg.derived.img_bitcount + cfg.derived.layer_edges[layer])
    //     + img_num_in_batch * cfg.network.layer_sizes[layer]
    //     + (node_idx - cfg.derived.layer_edges[layer]);
    node.indices.map(|i| bv_offset + i)
}

pub fn create_pseudorandom_lut_generator(
    lut_bank_opt: &Option<Vec<u64>>,
) -> Box<dyn Iterator<Item = u64> + Send> {
    let mut rng_luts = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    if let Some(lut_bank) = lut_bank_opt {
        let lut_bank_size = lut_bank.len();
        let ltbank = lut_bank.clone();
        if lut_bank_size == 16 {
            Box::new(std::iter::repeat(()).flat_map(move |_| {
                let chunk = rng_luts.next_u64();
                let items: [u64; 16] =
                    std::array::from_fn(|i| ltbank[((chunk >> (i * 4)) & 0xF) as usize].clone());
                items.into_iter()
            }))
        } else if lut_bank_size == 256 {
            Box::new(std::iter::repeat(()).flat_map(move |_| {
                let chunk = rng_luts.next_u64();
                let items: [u64; 8] = std::array::from_fn(|i| {
                    let index = ((chunk >> (i * 8)) & 0xFF) as usize;
                    ltbank[index].clone()
                });
                items.into_iter()
            }))
        } else {
            Box::new(std::iter::from_fn(move || {
                Some(ltbank[rng_luts.random_range(0..lut_bank_size)])
            }))
        }
    } else {
        Box::new(std::iter::from_fn(move || Some(rng_luts.next_u64())))
    }
}

pub fn pseudo_4bit_generator() -> impl Iterator<Item = u8> {
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    std::iter::repeat(()).flat_map(move |_| {
        let chunk = rng.next_u64();
        (0..16).map(move |i| (chunk >> (i * 4)) as u8 & 0xF)
    })
}

pub fn pseudo_6bit_generator() -> impl Iterator<Item = u8> {
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    std::iter::repeat(()).flat_map(move |_| {
        let chunk = rng.next_u64();
        (0..10).map(move |i| (chunk >> (i * 6)) as u8 & 0x3F)
    })
}

pub fn pseudo_8bit_generator() -> impl Iterator<Item = u8> {
    let mut rng = Xoshiro256PlusPlus::from_rng(&mut rand::rng());
    std::iter::repeat(()).flat_map(move |_| {
        let chunk = rng.next_u64();
        (0..8).map(move |i| (chunk >> (i * 8)) as u8 & 0xFF)
    })
}


#[cfg(test)]
mod tests {
    // use crate::{dataloader::*, lut_bank_creators::*, netcore::LUTNet, settings::*};
    // use bitvec::prelude::*;
    // use rand::Rng;
    use super::*;

    #[test]
    fn accuracy_calculation_works_correctly() {
        let test_cases = vec![
            (
                vec![1, 2, 3, 4, 5],
                vec![1, 2, 3, 4, 5],
                1.0,
                "all predictions correct",
            ),
            (
                vec![1, 2, 3, 4, 5],
                vec![0, 0, 0, 0, 0],
                0.0,
                "no predictions correct",
            ),
            (
                vec![1, 2, 3, 4],
                vec![1, 0, 3, 0],
                0.5,
                "half predictions correct",
            ),
            (
                vec![1, 0, 1, 1, 0, 1],
                vec![1, 1, 0, 1, 0, 0],
                0.5,
                "some predictions correct",
            ),
            (vec![], vec![], 0.0, "empty input"),
        ];

        for (y, predicted, expected, description) in test_cases {
            assert_eq!(
                calculate_accuracy(&y, &predicted),
                expected,
                "Failed on test case: '{}'",
                description
            );
        }
    }
}
