use bitvec::prelude::*;
use clap::Parser;
use lutnets::{ modelloader::load_model_from_file, netcore::*, settings::*};
use lutnets::{processing::*, utils::*};
use std::{error::Error, time::Instant};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, requires = "f")]
    s: Option<String>, // arg for passing split: Train, Val, Test
    #[arg(short, long)]
    f: Option<String>, // arg for passing model filename to test
}

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    let args = Args::parse();
    // let model_filename;
    let (cfg_default, ltnet): (&'static Configuration, LUTNet);
    let mut cfg: Configuration;
    let batch_size: u32;
    // Parse the arguments and load the mode. This code is exposed here because various algoritms may need additional arguments.

    match args.f {
        Some(model_file_str) => {
            if !std::path::Path::new(&model_file_str).exists() {
                panic!("File {} does not exist!", &model_file_str);
            }
            (cfg_default, ltnet) = load_model_from_file(&model_file_str);
            cfg = cfg_default.clone();
        }
        None => {
            panic!("Must specify a file with option -f");
        }
    }

    match args.s {
        Some(s_str) => match s_str.as_str() {
            "train" => {
                batch_size = 50000;
                cfg.data.datasplit = DataSplit::Train(batch_size);
            }
            "val" => {
                batch_size = 10000;
                cfg.data.datasplit = DataSplit::Val(batch_size);
            }
            "test" => {
                batch_size = 10000;
                cfg.data.datasplit = DataSplit::Test(batch_size);
            }
            _ => {
                panic!("Provide a valid split");
            }
        },
        None => {
            println!("No split provided, assuming Validation split.");
            batch_size = 10000;
            cfg.data.datasplit = DataSplit::Val(batch_size);
        }
    }
    println!("Getting Validating accuracy");

    cfg.data.batch_size = batch_size as usize;
    cfg.derived = DerivedValues::new(&cfg.data, &cfg.network);

    // let model_filename = &args[1][..];
    let (databits, labels) = lutnets::dataloader::csv_to_bitvec(&cfg)?;

    let batch_num = 0;
    let mut dbv = bitvec![u8, Msb0; 0; cfg.derived.bitvec_size];
    let y = &labels[batch_num * cfg.data.batch_size..(batch_num + 1) * cfg.data.batch_size];
    dbv[..cfg.derived.batch_bitcount].copy_from_bitslice(
        &databits
            [batch_num * cfg.derived.batch_bitcount..(batch_num + 1) * cfg.derived.batch_bitcount],
    );

    ltnet.apply_gates(&cfg, &mut dbv);
    // println!("Forward pass completed. Wall time: {:?}", start_time.elapsed());

    let (predicted_labels, loss) = (get_labels(&cfg, &dbv), get_loss(&cfg, &dbv, y));
    let accuracy = calculate_accuracy(y, &predicted_labels);
    println!(
        "Validation Loss: {:?}, Validation Accuracy: {}, Total time: {:?}",
        loss,
        accuracy,
        start_time.elapsed()
    );
    Ok(())
}
