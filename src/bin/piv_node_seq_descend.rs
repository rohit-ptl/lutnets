use clap::Parser;
use lutnets::{
    architectures::*, modelloader::load_model_from_file, netcore::*, settings::*,
    training_algos::piv_node_flipper::trainer::*,
};
use std::{error::Error, str::FromStr, time::Instant};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, requires = "f")]
    a: Option<String>, // arg for passing architecture, requires an output filename to save
    #[arg(short, long)]
    f: Option<String>, // arg for passing output filename along with architecture, or input model if passed alone
}

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();
    let args = Args::parse();
    let model_filename;
    let (cfg, mut ltnet): (&'static Configuration, LUTNet);

    // Parse the arguments and load the mode. This code is exposed here because various algoritms may need additional arguments.
    match (args.a, args.f) {
        (Some(arch_name), Some(model_file_base)) => {
            let arch = Architecture::from_str(&arch_name)?;
            (cfg, ltnet) = arch.build();
            model_filename = format!("{}.ltnet", model_file_base);
            if std::path::Path::new(&model_filename).exists() {
                panic!(
                    "File {} already exists! Omit the -a flag if you want to load an existing model file.",
                    &model_filename
                );
            }
        }
        (None, Some(model_file_base)) => {
            model_filename = format!("{}.ltnet", model_file_base);
            (cfg, ltnet) = load_model_from_file(&model_filename);
        }
        (Some(_), None) => unreachable!(),
        (None, None) => {
            panic!(
                "You need to either pass an existing model file with -m or specify architecture and output file name with -a and -f."
            );
        }
    }

    println!("Running Naive Evolutionary Algorithm Example");
    println!("Data split and size: {:?}", cfg.data.datasplit);
    let (databits, labels) = lutnets::dataloader::csv_to_bitvec(cfg)?;
    println!(
        "Read CSV rows: {}, Wall time: {:?}",
        labels.len(),
        start_time.elapsed()
    );
    ltnet.verify_lut_bank_integrity();
    train(
        &mut ltnet,
        cfg,
        &databits,
        &labels,
        100000,
        1,
        &model_filename,
    );
    println!("Total time: {:?}", start_time.elapsed());
    Ok(())
}
