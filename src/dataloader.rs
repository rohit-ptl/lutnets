use crate::settings::*;
use bitvec::prelude::*;
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

pub fn csv_to_bitvec(cfg: &Configuration) -> Result<(BitVec<u8, Msb0>, Vec<u8>), Box<dyn Error>> {
    let (filepath, rows, start_row) = match cfg.data.datasplit {
        // We are going to treat last 10k train rows as validation set
        DataSplit::Train(r) => {
            if r == 0 || r > 50000 {
                return Err("Train rows must be between 1 and 50000".into());
            }
            (cfg.data.train_filepath.clone(), r as usize, 0)
        }
        DataSplit::Val(r) => {
            if r == 0 || r > 10000 {
                return Err("Validation rows must be between 1 and 10000".into());
            }
            (cfg.data.train_filepath.clone(), r as usize, 50000)
        }
        DataSplit::Test(r) => {
            if r == 0 || r > 10000 {
                return Err("Test rows must be between 1 and 10000".into());
            }
            (cfg.data.test_filepath.clone(), r as usize, 0)
        }
    };
    let cols = cfg.derived.cols;

    let file = File::open(filepath)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut bit_vector = BitVec::new();
    let mut labels = Vec::with_capacity(rows);
    let mut records_iter = rdr.records();

    // Skip rows if necessary
    for _ in 0..start_row {
        if records_iter.next().is_none() {
            return Err(format!(
                "Expected to skip {} rows, but the file ended before that.",
                start_row
            )
            .into());
        }
    }

    for i in 0..rows {
        match records_iter.next() {
            Some(Ok(record)) => {
                if record.len() != cols + 1 {
                    return Err(format!(
                        "Row {} has {} columns, but {} were expected.",
                        i + 1,
                        record.len(),
                        cols + 1
                    )
                    .into());
                }

                let mut fields = record.iter();
                let first_field = fields.next().unwrap();
                let label_value: u8 = first_field.trim().parse()?;
                labels.push(label_value);

                for field in fields {
                    let value: u8 = field.trim().parse()?;
                    bit_vector.extend_from_bitslice(value.view_bits::<Msb0>());
                }
            }
            Some(Err(e)) => return Err(e.into()),
            None => {
                return Err(format!(
                    "Expected to read {} rows, but the file ended after {} rows.",
                    rows, i
                )
                .into());
            }
        }
    }
    Ok((bit_vector, labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_to_bitvec_loads_data_correctly() {
        let mut cfg = initialize_app_config();
        cfg.data.datasplit = DataSplit::Train(1000); // The hardcoded values below are from first 1000 rows of train set

        let (databits, labels) = csv_to_bitvec(&cfg).unwrap();
        // Manually looked up values hardcoded below
        let test_rows: [usize; 10] = [5, 21, 78, 101, 225, 321, 453, 654, 777, 876];
        let test_cols: [usize; 10] = [215, 158, 157, 691, 518, 429, 442, 292, 296, 372];
        let true_pixel_values: [u8; 10] = [87, 112, 42, 209, 129, 168, 252, 41, 11, 3];
        let true_labels: [u8; 10] = [9, 4, 1, 5, 1, 0, 3, 5, 0, 3];
        for (i, &row_idx) in test_rows.iter().enumerate() {
            let idx_to_print = (row_idx - 1) * (cfg.data.dim1 * cfg.data.dim2 * cfg.data.dim3)
                + (test_cols[i] - 1) * (cfg.data.dim1);
            let bslice = &databits[idx_to_print..idx_to_print + 8];
            assert_eq!(bslice.load_be::<u8>(), true_pixel_values[i]);
            assert_eq!(labels[row_idx - 1], true_labels[i]);
        }
    }
}
