use crate::settings::*;
use bitvec::prelude::*;
use std::collections::{HashMap, HashSet};

// pivotal_nodes is a vector of hashmaps, one per input in the batch. The keys of the hashmap are the output nodes (eight bits, 0..8) and the values are sets of nodes in the bitvec that will flip the output node if flipped themselves.

//not all these functions are useful, and sorry about lack of comments.

pub fn find_most_frequent_pivotal_node(
    pivotal_nodes: &[HashMap<usize, HashSet<usize>>],
) -> Vec<HashMap<usize, HashSet<usize>>> {
    pivotal_nodes
        .iter()
        .map(|input_map| {
            let mut reverse_lookup: HashMap<usize, HashSet<usize>> = HashMap::new();
            for (&key, number_set) in input_map {
                for &number in number_set {
                    reverse_lookup.entry(number).or_default().insert(key);
                }
            }

            let winner_entry = reverse_lookup
                .into_iter()
                .max_by_key(|(_, keys)| keys.len());

            if let Some((winner, keys)) = winner_entry {
                // keys.sort(); // Not really needed, but makes testing easier to read when printing output
                HashMap::from([(winner, keys)])
            } else {
                HashMap::new()
            }
        })
        .collect()
}

pub fn find_randomized_selections(
    pivotal_nodes: &[HashMap<usize, HashSet<usize>>],
) -> Vec<HashMap<usize, usize>> {
    pivotal_nodes
        .iter()
        .map(|input_map| {
            let mut result_map = HashMap::new();
            let mut disqualified_numbers = HashSet::new();

            let mut coverage_map: HashMap<usize, HashSet<usize>> = HashMap::new(); //let's have a reverse lookup for efficiency
            for (&key, values) in input_map {
                for &value in values {
                    coverage_map.entry(value).or_default().insert(key);
                }
            }
            // println!("Coverage map: {:?}", coverage_map);
            // let mut map_keys = input_map.keys().collect::<Vec<&usize>>(); // Remove the sort after debugging and print iterate over input_map.keys() for indeterministic behavior
            // map_keys.sort();
            for &key in input_map.keys() {
                if result_map.contains_key(&key) {
                    continue;
                }
                // println!("Key: {}", key);
                // let choice = input_map[&key] // pick randomly from available options
                //     .iter()
                //     .find(|&num| !disqualified_numbers.contains(num));

                let choice = input_map[&key] // let's pick numbers deep in to see a lot of cascading effects
                    .iter()
                    .filter(|&num| !disqualified_numbers.contains(num))
                    .min();
                println!(
                    "Choice: {:?}, Coverage: {:?}",
                    choice,
                    coverage_map.get(choice.unwrap_or(&usize::MAX))
                );
                if let Some(&chosen_num) = choice {
                    // Assign this number to all keys that have it.
                    let keys_to_assign = coverage_map.get(&chosen_num).cloned().unwrap_or_default();
                    for &key_to_assign in &keys_to_assign {
                        result_map.insert(key_to_assign, chosen_num);
                    }
                    // Disqualify all numbers from the sets of the keys we just assigned.
                    for &key_to_assign in &keys_to_assign {
                        if let Some(original_set) = input_map.get(&key_to_assign) {
                            for &num_to_remove in original_set {
                                disqualified_numbers.insert(num_to_remove);
                            }
                        }
                    }
                }
            }
            result_map
        })
        .collect()
}

pub fn find_pivotal_node_with_scoring(
    cfg: &Configuration,
    pivotal_nodes: &[HashMap<usize, HashSet<usize>>],
    flags: &BitVec<u8, Msb0>,
) -> Vec<usize> {
    pivotal_nodes
        .iter()
        .zip(flags.chunks(8))
        .map(|(input_map, key_flags)| {
            let mut node_scores: HashMap<usize, isize> = HashMap::new();
            for (&key, number_set) in input_map {
                let key_index = key - cfg.derived.layer_edges[cfg.derived.num_layers - 1];
                let is_good = key_flags
                    .get(key_index)
                    .as_deref()
                    .copied()
                    .unwrap_or(false);
                let key_score = if is_good { 1 } else { -1 };
                for &number in number_set {
                    *node_scores.entry(number).or_default() += key_score;
                }
            }
            node_scores
                .into_iter()
                .max_by_key(|&(_, score)| score)
                .map(|(node, _score)| node)
                .unwrap_or(0)
        })
        .collect()
}

pub fn find_pivotal_nodes_by_key(
    cfg: &Configuration,
    pivotal_nodes: &[HashMap<usize, HashSet<usize>>],
    flags: &BitVec<u8, Msb0>,
) -> Vec<Option<usize>> {
    // Create a scoreboard for each key from 3706 to 3713. idx 0 is for key 3706, 1 for 3707, etc.
    let num_keys = cfg.network.layer_sizes[cfg.derived.num_layers - 1];
    let start_key = cfg.derived.layer_edges[cfg.derived.num_layers - 1];
    println!("nk {}, sk {}", num_keys, start_key);
    let mut key_specific_scores: Vec<HashMap<usize, isize>> = vec![HashMap::new(); num_keys];

    for (input_map, key_flags) in pivotal_nodes.iter().zip(flags.chunks(num_keys)) {
        for (&key, number_set) in input_map {
            let key_index = key - start_key;
            if key_index < num_keys {
                let is_good = key_flags
                    .get(key_index)
                    .as_deref()
                    .copied()
                    .unwrap_or(false);
                let score_change = if is_good { 1 } else { -1 };

                let node_scores = &mut key_specific_scores[key_index];

                for &number in number_set {
                    *node_scores.entry(number).or_default() += score_change;
                }
            }
        }
    }

    key_specific_scores
        .into_iter()
        .map(|node_scores| {
            node_scores
                .into_iter()
                .max_by_key(|&(_, score)| score)
                .map(|(node, _)| node)
        })
        .collect()
}

pub fn find_global_pivotal_node(
    cfg: &Configuration,
    pivotal_nodes: &[HashMap<usize, HashSet<usize>>],
    flags: &BitVec<u8, Msb0>,
) -> Option<usize> {
    let mut global_node_scores: HashMap<usize, isize> = HashMap::new();

    for (input_map, key_flags) in pivotal_nodes.iter().zip(flags.chunks(8)) {
        for (&key, number_set) in input_map {
            let key_index = key - cfg.derived.layer_edges[cfg.derived.num_layers - 1];
            let is_good = key_flags
                .get(key_index)
                .as_deref()
                .copied()
                .unwrap_or(false);
            let key_score = if is_good { 1 } else { -1 };

            for &number in number_set {
                *global_node_scores.entry(number).or_default() += key_score;
            }
        }
    }

    global_node_scores
        .into_iter()
        .max_by_key(|&(_, score)| score)
        .map(|(node, _score)| node)
}

pub fn score_lut_by_node_presence(
    cfg: &Configuration,
    pivotal_nodes: &[HashMap<usize, HashSet<usize>>],
    flags: &BitVec<u8, Msb0>,
    lut_inputs: &[u8],
    node_idx: usize,
) -> HashMap<u8, isize> {
    let mut lut_scores: HashMap<u8, isize> = HashMap::new();
    let zipped_iter = pivotal_nodes
        .iter()
        .zip(flags.chunks(8))
        .zip(lut_inputs.iter());

    for ((input_map, key_flags), &lut_input) in zipped_iter {
        for (&key, number_set) in input_map {
            if number_set.contains(&node_idx) {
                let key_index = key - cfg.derived.layer_edges[cfg.derived.num_layers - 1];
                let is_good = key_flags
                    .get(key_index)
                    .as_deref()
                    .copied()
                    .unwrap_or(false);
                let score_change = if is_good { 1 } else { -1 };
                *lut_scores.entry(lut_input).or_default() += score_change;
            }
        }
    }

    lut_scores
}
