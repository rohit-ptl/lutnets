use crate::settings::*;

// These iterators contain a lot of the hard logic needed for forward pass and generating spans for CNN style networks

// Iterator over all gates in all batches and layers ----------------------------------------------------
pub struct GateIterator<'a> {
    node_index: usize, // Number that gives you node_index in ltnet
    img_num_in_batch: usize,
    current_layer: usize,
    bitvec_index: usize,     // index in the bitvec for current node and batch
    max_bitvec_index: usize, // length of iterator + data bitcount of batch
    offset_increment_size: usize, // this is how much the current batch offset will be
    readbit_offset: usize,   // the amount of offset of bit locations due to packed batches
    cfg: &'a Configuration,
}

impl<'a> GateIterator<'a> {
    pub fn new(cfg: &'a Configuration) -> Self {
        let total_len = cfg.derived.layer_edges.last().unwrap() * cfg.data.batch_size;
        GateIterator {
            node_index: 0,
            img_num_in_batch: 0,
            current_layer: 0,
            bitvec_index: cfg.derived.batch_bitcount,
            max_bitvec_index: total_len + cfg.derived.batch_bitcount,
            offset_increment_size: cfg.derived.img_bitcount,
            readbit_offset: 0,
            cfg,
        }
    }
}

impl<'a> Iterator for GateIterator<'a> {
    type Item = (usize, usize, usize, usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bitvec_index >= self.max_bitvec_index {
            return None;
        }

        let current_tuple = (
            self.node_index,
            self.img_num_in_batch,
            self.current_layer,
            self.bitvec_index,
            self.readbit_offset,
        );

        // This is state for next iteration (so this is not being returned in this iteration, in-fact the last time this part is run it is never returned because iterator ends)
        self.node_index += 1;
        self.bitvec_index += 1;
        if self.node_index >= self.cfg.derived.layer_edges[self.current_layer + 1] {
            self.node_index = self.cfg.derived.layer_edges[self.current_layer];
            self.img_num_in_batch += 1;
            self.readbit_offset += self.offset_increment_size;
            if self.img_num_in_batch >= self.cfg.data.batch_size {
                self.readbit_offset -= self.offset_increment_size;
                self.node_index = self.cfg.derived.layer_edges[self.current_layer + 1];
                self.img_num_in_batch = 0;
                self.current_layer += 1;
                self.offset_increment_size = self.cfg.network.layer_sizes[self.current_layer - 1]
            }
        }
        Some(current_tuple)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.max_bitvec_index - self.bitvec_index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for GateIterator<'a> {
    fn len(&self) -> usize {
        self.max_bitvec_index - self.bitvec_index
    }
}

// Iterator over all gates in a specific layer and all batches ----------------------------------------------------
pub struct LayerGateIterator<'a> {
    node_index: usize, // Number that gives you node_index in ltnet
    img_num_in_batch: usize,
    layer: usize,
    bitvec_index: usize,     // index in the bitvec for current node and batch
    max_bitvec_index: usize, // length of iterator + data bitcount of batch
    offset_increment_size: usize, // this is how much the current batch offset will be
    readbit_offset: usize,   // the amount of offset of bit locations due to packed batches
    cfg: &'a Configuration,
}

impl<'a> LayerGateIterator<'a> {
    pub fn new(cfg: &'a Configuration, layer: usize) -> Self {
        assert!(
            layer < cfg.derived.num_edges - 1,
            "Layer index out of bounds"
        );
        // let total_len = cfg.derived.layer_edges.last().unwrap() * cfg.data.batch_size;
        LayerGateIterator {
            node_index: cfg.derived.layer_edges[layer],
            img_num_in_batch: 0,
            layer: layer,
            bitvec_index: cfg.derived.batch_bitcount
                + (cfg.data.batch_size * cfg.derived.layer_edges[layer]),
            max_bitvec_index: cfg.derived.batch_bitcount
                + (cfg.data.batch_size * cfg.derived.layer_edges[layer + 1]),
            offset_increment_size: match layer {
                0 => cfg.derived.img_bitcount,
                _ => cfg.network.layer_sizes[layer - 1],
            },
            readbit_offset: match layer {
                0 => 0,
                _ => {
                    (cfg.data.batch_size - 1)
                        * (cfg.derived.img_bitcount + cfg.derived.layer_edges[layer - 1])
                }
            },
            cfg,
        }
    }
}
impl<'a> Iterator for LayerGateIterator<'a> {
    type Item = (usize, usize, usize, usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bitvec_index >= self.max_bitvec_index {
            return None;
        }

        let current_tuple = (
            self.node_index,
            self.img_num_in_batch,
            self.layer,
            self.bitvec_index,
            self.readbit_offset,
        );

        // This is state for next iteration (so this is not being returned in this iteration, in-fact the last time this part is run it is never returned because iterator ends)
        self.node_index += 1;
        self.bitvec_index += 1;
        if self.node_index >= self.cfg.derived.layer_edges[self.layer + 1] {
            self.node_index = self.cfg.derived.layer_edges[self.layer];
            self.img_num_in_batch += 1;
            self.readbit_offset += self.offset_increment_size;
        }
        Some(current_tuple)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.max_bitvec_index - self.bitvec_index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for LayerGateIterator<'a> {
    fn len(&self) -> usize {
        self.max_bitvec_index - self.bitvec_index
    }
}

// Iterator to generate the filter spans for 6-input LUTs ----------------------------------------------------
pub struct SpanGenerator {
    // this is a good building block for generating spans to build architectures insipired by CNNs. One example is in netimpl.rs
    curr: usize,
    x: usize,
    y: usize,
    z: usize,
    dim1: usize, //these dims are not meant to be dims of input image and do not come from config
    dim2: usize,
    dim3: usize,
    len1: usize,
    len2: usize,
    len3: usize,
    hop1: usize,
    hop2: usize,
    hop3: usize,
    rpoa: [usize; 6], // This will hold the offsets for the 6 inputs
    finished: bool,
}

impl<'a> SpanGenerator {
    pub fn new(
        offset: usize,
        dim1: usize,
        dim2: usize,
        dim3: usize,
        len1: usize,
        len2: usize,
        len3: usize,
        hop1: usize,
        hop2: usize,
        hop3: usize,
    ) -> Self {
        let input_len = len1 * len2 * len3;
        assert!(input_len == 6, "We are working with 6-input LUTs");
        assert!(
            hop1 != 0 && hop2 != 0 && hop3 != 0,
            "hop1, hop2, and hop3 must all be non-zero"
        );
        let mut relative_position_offset_array = [0; 6];
        let mut index = 0;
        for k in 0..len3 {
            for j in 0..len2 {
                for i in 0..len1 {
                    relative_position_offset_array[index] = offset + dim1 * dim2 * k + dim1 * j + i;
                    index += 1;
                }
            }
        }
        // println!("RPOA: {:?}", relative_position_offset_array);
        SpanGenerator {
            curr: 0,
            x: 0,
            y: 0,
            z: 0,
            dim1,
            dim2,
            dim3,
            len1,
            len2,
            len3,
            hop1,
            hop2,
            hop3,
            rpoa: relative_position_offset_array,
            finished: false,
        }
    }
}

impl Iterator for SpanGenerator {
    type Item = [usize; 6];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        // println!("x: {}, y: {}, z: {}", self.x, self.y, self.z);
        let current_tuple: [usize; 6] = [
            self.curr + self.rpoa[0],
            self.curr + self.rpoa[1],
            self.curr + self.rpoa[2],
            self.curr + self.rpoa[3],
            self.curr + self.rpoa[4],
            self.curr + self.rpoa[5],
        ];
        // Values for next iteration
        if self.x + self.len1 < self.dim1 {
            if self.x + self.hop1 + self.len1 <= self.dim1 {
                self.x += self.hop1;
            } else {
                self.x = self.dim1 - self.len1;
            }
        } else if self.y + self.len2 < self.dim2 {
            if self.y + self.hop2 + self.len2 <= self.dim2 {
                self.y += self.hop2;
                self.x = 0;
            } else {
                self.y = self.dim2 - self.len2;
                self.x = 0;
            }
        } else if self.z + self.len3 < self.dim3 {
            if self.z + self.hop3 + self.len3 <= self.dim3 {
                self.z += self.hop3;
                self.y = 0;
                self.x = 0;
            } else {
                self.z = self.dim3 - self.len3;
                self.x = 0;
                self.y = 0;
            }
        } else {
            self.finished = true;
        }
        self.curr = self.dim1 * self.dim2 * self.z + self.dim1 * self.y + self.x; // Do not update current_tuple here, obviously
        Some(current_tuple)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        // This is exactly the number of iterations remaining
        let remaining = ((self.dim1 - self.len1 - self.x + 2 * self.hop1 - 1) / self.hop1) as usize
            * ((self.dim2 - self.len2 - self.y + 2 * self.hop2 - 1) / self.hop2) as usize
            * ((self.dim3 - self.len3 - self.z + 2 * self.hop3 - 1) / self.hop3) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SpanGenerator {
    fn len(&self) -> usize {
        ((self.dim1 - self.len1 - self.x + 2 * self.hop1 - 1) / self.hop1) as usize
            * ((self.dim2 - self.len2 - self.y + 2 * self.hop2 - 1) / self.hop2) as usize
            * ((self.dim3 - self.len3 - self.z + 2 * self.hop3 - 1) / self.hop3) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_iterator_test() {
        let cfg = get_cfg(None);

        let mut iterator = GateIterator::new(cfg);
        let mut i: usize = 0;
        // let (mut node_index, mut img_num_in_batch, mut current_layer, mut bitvec_index, mut readbit_offset) = iterator.next().unwrap();
        let mut val = iterator.next();
        assert_eq!(
            val,
            Some((0, 0, 0, cfg.derived.img_bitcount * cfg.data.batch_size, 0))
        );
        while val.is_some() {
            if i == cfg.derived.layer_edges[1] {
                // println!("i: {}, cfg.derived.layer_edges: {:?}, val: {:?}, trueval: {}", i, cfg.derived.layer_edges, val.unwrap(), cfg.derived.img_bitcount*(cfg.data.batch_size-1));
                assert_eq!(val.unwrap().2 + val.unwrap().1, 1);
                assert_eq!(val.unwrap().3, i + cfg.derived.batch_bitcount);
                assert_eq!(
                    val.unwrap().4,
                    cfg.derived.img_bitcount * std::cmp::min(1, cfg.data.batch_size - 1)
                );
            }
            if i == cfg.derived.layer_edges[2] * cfg.data.batch_size {
                // println!("i: {}, cfg.derived.layer_edges: {:?}, val: {:?}, trueval: {}", i, cfg.derived.layer_edges, val.unwrap(), cfg.derived.img_bitcount*(cfg.data.batch_size-1));
                assert_eq!(val.unwrap().3, i + cfg.derived.batch_bitcount);
                assert_eq!(
                    val.unwrap().4,
                    (cfg.derived.img_bitcount + cfg.network.layer_sizes[0])
                        * (cfg.data.batch_size - 1)
                );
            }
            if i == cfg.derived.layer_edges[3] * cfg.data.batch_size - 1 {
                // println!("i: {}, cfg.derived.layer_edges: {:?}, val: {:?}, trueval: {}", i, cfg.derived.layer_edges, val.unwrap(), cfg.derived.img_bitcount*(cfg.data.batch_size-1));
                assert_eq!(val.unwrap().3, i + cfg.derived.batch_bitcount);
                assert_eq!(
                    val.unwrap().4,
                    (cfg.derived.img_bitcount
                        + cfg.network.layer_sizes[0]
                        + cfg.network.layer_sizes[1])
                        * (cfg.data.batch_size - 1)
                );
            }
            i += 1;
            val = iterator.next();
        }
        assert_eq!(
            i,
            cfg.network.layer_sizes.iter().sum::<usize>() * cfg.data.batch_size
        );
    }
    #[test]
    fn layer_gate_iterator_test() {
        let cfg = get_cfg(None);
        let mut iterator = LayerGateIterator::new(cfg, 0);
        let mut i: usize = 0;
        let mut val = iterator.next();
        assert_eq!(
            val,
            Some((0, 0, 0, cfg.derived.img_bitcount * cfg.data.batch_size, 0))
        );
        while val.is_some() {
            if i == cfg.derived.layer_edges[1] {
                // println!("i: {}, cfg.derived.layer_edges: {:?}, val: {:?}, trueval: {}", i, cfg.derived.layer_edges, val.unwrap(), cfg.derived.img_bitcount*(cfg.data.batch_size-1));
                assert_eq!(val.unwrap().2 + val.unwrap().1, 1);
                assert_eq!(
                    val.unwrap().4,
                    cfg.derived.img_bitcount * std::cmp::min(1, cfg.data.batch_size - 1)
                );
            }
            i += 1;
            val = iterator.next();
        }
        assert_eq!(i, cfg.network.layer_sizes[0] * cfg.data.batch_size);

        let mut iterator = LayerGateIterator::new(cfg, 2);
        let mut i: usize = 0;
        let mut val = iterator.next();
        // println!("i: {}, cfg.network.layer_sizes: {:?}, val: {:?}, trueval: {}", i, cfg.network.layer_sizes, &val.unwrap(), cfg.derived.img_bitcount*(cfg.data.batch_size-1));
        assert_eq!(
            val,
            Some((
                cfg.network.layer_sizes[0] + cfg.network.layer_sizes[1],
                0,
                2,
                cfg.data.batch_size
                    * (cfg.derived.img_bitcount
                        + cfg.network.layer_sizes[0]
                        + cfg.network.layer_sizes[1]),
                (cfg.data.batch_size - 1) * (cfg.derived.img_bitcount + cfg.network.layer_sizes[0])
            ))
        );
        while val.is_some() {
            if i == cfg.data.batch_size * cfg.network.layer_sizes[2] - 1 {
                // println!("i: {}, cfg.derived.layer_edges: {:?}, val: {:?}, trueval: {}", i, cfg.derived.layer_edges, val.unwrap(), cfg.derived.img_bitcount*(cfg.data.batch_size-1));
                assert_eq!(val.unwrap().2 + val.unwrap().1, 1 + cfg.data.batch_size);
                assert_eq!(
                    val.unwrap().4,
                    (cfg.derived.img_bitcount
                        + cfg.network.layer_sizes[0]
                        + cfg.network.layer_sizes[1])
                        * (cfg.data.batch_size - 1)
                );
            }
            i += 1;
            val = iterator.next();
        }
        assert_eq!(i, cfg.network.layer_sizes[2] * cfg.data.batch_size);
    }

    #[test]
    fn gate_and_layergate_iterators_are_the_same() {
        let cfg = get_cfg(None);
        // let (mut node_index, mut img_num_in_batch, mut current_layer, mut bitvec_index, mut readbit_offset) = iterator.next().unwrap();
        let mut layer: usize = 0;
        let mut iterator1 = GateIterator::new(cfg);
        let mut iterator2 = LayerGateIterator::new(cfg, layer as usize);
        let mut val1 = iterator1.next();
        let mut val2 = iterator2.next();
        let mut count: usize = 0;
        while val1.is_some() {
            if layer != val1.unwrap().2 {
                layer = val1.unwrap().2;
                iterator2 = LayerGateIterator::new(cfg, layer as usize);
                val2 = iterator2.next();
            }
            while val2.is_some() {
                assert_eq!(val1, val2);
                val1 = iterator1.next();
                val2 = iterator2.next();
                count += 1;
            }
        }
        assert_eq!(
            count,
            cfg.network.layer_sizes.iter().sum::<usize>() * cfg.data.batch_size
        );
    }
}
