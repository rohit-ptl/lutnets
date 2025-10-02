use crate::settings::*;
use bitvec::prelude::*;
use cached::proc_macro::cached;


pub fn get_labels(cfg: &Configuration, dbv: &BitVec<u8, Msb0>) -> Vec<usize> {
    // Extract predicted labels from the final part of the bitvec
    let predicted_label_bitslice = &dbv[cfg.derived.bitvec_size
        - cfg.data.batch_size * cfg.network.layer_sizes[cfg.derived.num_layers - 1]
        ..cfg.derived.bitvec_size];
    predicted_label_bitslice
        .chunks_exact(cfg.derived.output_bitsize)
        .take(cfg.data.batch_size)
        .map(|chunk| {
            cfg.network
                .output_embedding
                .iter()
                .enumerate()
                .min_by_key(|&(_index, &val)| (chunk.load_be::<usize>() ^ val).count_ones())
                .map(|(index, _val)| index)
                .unwrap()
        })
        .collect::<Vec<usize>>()
}

pub fn get_loss_vec(
    cfg: &Configuration,
    dbv: &BitVec<u8, Msb0>,
    labels: &[usize],
) -> BitVec<u8, Msb0> {
    // Calculate loss vector as differing bits between predicted and true labels
    let predicted_label_bitslice = &dbv[cfg.derived.bitvec_size
        - cfg.data.batch_size * cfg.derived.output_bitsize
        ..cfg.derived.bitvec_size];
    let mut true_label_bitslice = get_true_label_bitslice(cfg, labels);
    true_label_bitslice ^= predicted_label_bitslice; // This is XOR between true labels and predicted labels, despite the name
    true_label_bitslice
}

pub fn get_loss(cfg: &Configuration, dbv: &BitVec<u8, Msb0>, labels: &[usize]) -> usize {
    get_loss_vec(cfg, dbv, labels).count_ones()
}

pub fn get_predicted_embedding(cfg: &Configuration, dbv: &BitVec<u8, Msb0>) -> Vec<u8> {
    // Extract predicted labels from the final part of the bitvec
    let predicted_label_bitslice = &dbv[cfg.derived.bitvec_size
        - cfg.data.batch_size * cfg.network.layer_sizes[cfg.derived.num_layers - 1]
        ..cfg.derived.bitvec_size];
    predicted_label_bitslice
        .chunks_exact(8)
        .take(cfg.data.batch_size)
        .map(|chunk| chunk.load_be::<u8>())
        .collect::<Vec<u8>>()
}


#[cached(
    size = 50, // max 50 cached results
    key = "Vec<usize>", 
    convert = r#"{ labels.to_vec() }"#,
)]
fn get_true_label_bitslice(
    cfg: &Configuration,
    labels: &[usize],
) -> BitVec<u8, Msb0> {
    let chunk_size = cfg.derived.output_bitsize;
    labels
        .iter()
        .map(|label_val| cfg.network.output_embedding[*label_val])
        .fold(
            BitVec::<u8, Msb0>::with_capacity(labels.len() * chunk_size),
            |mut acc, embedding| {
                // Get the bits and extend the accumulator. The borrow is temporary and safe.
                let bits = embedding.view_bits::<Msb0>();
                acc.extend_from_bitslice(&bits[usize::BITS as usize - chunk_size..]);
                acc
            },
        )
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_fetched_correctly() {
        let mut cfg = initialize_app_config_with_network(None);
        cfg.network.output_embedding = vec![51, 15, 77, 85, 240, 170, 153, 204, 102, 210];
        cfg.data.batch_size = 3;
        let output_layer_size = 8;
        cfg.network.layer_sizes = vec![64, output_layer_size];
        cfg.derived = DerivedValues::new(&cfg.data, &cfg.network);
        let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];

        // The network's predicted raw output bytes for our batch of 3.
        // Item 1: `50` -> Closest to `51` (label 0). Hamming distance is 1.
        // Item 2: `77` -> Perfect match for label 2.
        // Item 3: `102`-> Perfect match for label 8.
        let predicted_outputs: [u8; 3] = [50, 77, 102];
        let output_start_index =
            cfg.derived.bitvec_size - (cfg.data.batch_size * output_layer_size);
        let output_slice = &mut dbv[output_start_index..];
        output_slice.store_be(
            (u32::from(predicted_outputs[0]) << 16)
                | (u32::from(predicted_outputs[1]) << 8)
                | (u32::from(predicted_outputs[2])),
        );
        let expected_labels = vec![0, 2, 8];
        let actual_labels = get_labels(&cfg, &dbv);
        assert_eq!(
            expected_labels, actual_labels,
            "The decoded labels did not match the expected labels."
        );
    }

    #[test]
    fn loss_calculation_test() {
        let mut cfg = initialize_app_config_with_network(None);
        cfg.network.output_embedding = vec![51, 15, 77, 85, 240, 170, 153, 204, 102, 210];
        cfg.data.batch_size = 3;
        let output_layer_size = 8;
        cfg.network.layer_sizes = vec![64, output_layer_size];
        cfg.derived = DerivedValues::new(&cfg.data, &cfg.network);
        let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];

        // The network's predicted raw output bytes for our batch of 3.
        // Item 1: `50` -> Closest to `51` (label 0). Hamming distance is 1.
        // Item 2: `77` -> Perfect match for label 2.
        // Item 3: `102`-> Perfect match for label 8.
        let predicted_outputs: [u8; 3] = [50, 77, 105];
        let true_labels: &[usize] = &[0, 2, 8];
        // println!("True labels: {:b}, {:b}", 102, 105);
        let output_start_index =
            cfg.derived.bitvec_size - (cfg.data.batch_size * output_layer_size);
        let output_slice = &mut dbv[output_start_index..];
        output_slice.store_be(
            (u32::from(predicted_outputs[0]) << 16)
                | (u32::from(predicted_outputs[1]) << 8)
                | (u32::from(predicted_outputs[2])),
        );

        let loss_byte_1: u8 = 51 ^ 50; // 0b00110011 ^ 0b00110010 = 0b00000001 (1)
        let loss_byte_2: u8 = 77 ^ 77; // 0b01001101 ^ 0b01001101 = 0b00000000 (0)
        let loss_byte_3: u8 = 102 ^ 105; // 0b01100110 ^ 0b01101001 = 0b00001111 (15)

        // Construct the expected BitVec from the loss bytes
        let mut expected_loss_vec = BitVec::<u8, Msb0>::new();
        expected_loss_vec.extend_from_bitslice(&loss_byte_1.view_bits::<Msb0>());
        expected_loss_vec.extend_from_bitslice(&loss_byte_2.view_bits::<Msb0>());
        expected_loss_vec.extend_from_bitslice(&loss_byte_3.view_bits::<Msb0>());

        // --- 4. Run the function and Assert ---
        let actual_loss_vec = get_loss_vec(&cfg, &dbv, true_labels);
        assert_eq!(actual_loss_vec.count_ones(), 5);
        assert_eq!(
            expected_loss_vec, actual_loss_vec,
            "The calculated loss vector did not match the expected XOR result."
        );
    }
}
