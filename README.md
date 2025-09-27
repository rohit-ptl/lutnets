# LUTNets

**Gaol :** Find out if we can build a simple LUT based deep learning network that performs well on MNIST (>98% accuracy), and can be trained in a reasonable timeframe on an everyday CPU.

This repository contains built from scratch (pure Rust), tools to build, test, and train Look-Up Table (LUT) based deep learning networks. There networks are comprised solely of 6-input LUTs, operating directly on data bits. The idea is to provide an experimentation ground for reseach on model architecture, training methods and more. This repo provides:
- Tools to quickly build your own LUTNet and experiment with architectures
- Out of the box, you get forward pass/inference, loss calculations
- Many other utility functions e.g. mutations, and tracking which nodes deep within the network can affect final output with a single flip
- Tools to implement and experiment with your own training algorithms
- Everything implemented for MNIST ready for iteration and experimentation
- Examples of architectures and training algorithms
- All in a highly optimized for CPU, pure Rust code so that experiments can be done locally and painlessly

## Overview
The core idea behind this project is to create a deep learning network that operates on binary data (bits) and uses Look-Up Tables (LUTs) as its fundamental building blocks. LUTs can have fairly complex logic embedded in them with astonishingly little compute. Compute is the key point here, and networks built entirely from LUTs would be game-changing for the industry. A "forward pass" on a large network with several hundred thousand nodes takes seconds on modest hardware. For example with ~450k nodes, which is about the same number of neurons that a Llama 3 8B has, it takes about **10 milliseconds** to run a forward pass on my laptop with Intel Core Ultra 7 155H CPU. That's with this repo which is pure rust code, feel free to build your network and run `cargo run --release --bin reference_example` to try yourself.

LUT based learning networks have the potential to increase our computing capacity by several orders of magnitude - the sort of jump we need to unlock next level of intelligence. **But**, they're near impossible to train.

### The training challenge
Discrete space is not differentiable and there's no good way to train these models. Training obviously works with evolutionary algorithms but they are impractical. I tried some more tailored approaches to get rapid loss improvements, but they hit local minima fast (you can find them in the repo). I'll admit that figuring out how to successfully and efficiently train these network is a fairly low probability event.

Training challenges are the main reason there isn't much research in this area. But the reward for figuring this out is enormous, and I think more effort is warranted. This is sort of the main reason for putting together this repo.

### Architecture 

I haven't had the opportunity to explore architectures. Even the cnn_inspired example is somewhat lazy in the sense that it just applies spans all the way to the end but never flattens or distributes bits. Obvious ideas/improvements haven't been tested/implemented. Lot of room for experimentation here.

## Project Layout
The project is structured as follows:

```
├── Cargo.toml
├── ....
├── Settings.toml
├── mnist_data_csv (not pushed to github)
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── src
│   ├── architectures
│   │   ├── cnn_inspired
│   │   └── mod.rs
│   ├── bin
│   │   ├── naive_evolutionary.rs
│   │   ├── piv_node_seq_descend.rs
│   │   └── reference_example.rs
│   ├── training_algos
│   │   ├── naive_evolutionary
│   │   ├── piv_node_flipper
│   │   └── mod.rs
│   ├── dataloader.rs
│   ├── iterators.rs
│   ├── lib.rs
│   ├── lut_bank_creators.rs
│   ├── netcore.rs
│   ├── processing.rs
│   ├── settings.rs
│   └── utils.rs
└── ...
```

*   `mnist_data_csv`: Doesn't exist on github repo, but put your MNIST csv files here.
*   `src/lib.rs`: The main library file, which contains the core modules.
*   `src/netcore.rs`: The core of the neural network, defining the `LUTNet` struct and its associated methods.
*   `src/dataloader.rs`:  Handles loading and preprocessing of data. The current implementation reads data from CSV files.
*   `src/processing.rs`: Contains functions for processing the output of the network, such as calculating loss and accuracy.
*   `src/settings.rs`:  Defines the configuration structures for the network and training process.
*   `src/architectures`: Contains different network architectures.
    *   `cnn_inspired`: An example of a CNN-inspired architecture.
*   `src/training_algos`: Contains different training algorithms.
    *   `naive_evolutionary`: A simple evolutionary training algorithm.
    *   `piv_node_flipper`: A training algorithm that flips bits in the LUTs.
*   `src/bin`: Contains the binary targets for the project.
    *   `reference_example.rs`: A simple example of how to use the library.
    *   `naive_evolutionary.rs`: A binary for training a network using the naive evolutionary algorithm.
    *   `piv_node_seq_descend.rs`: A binary for training a network using the pivotal node sequential descend algorithm.

## Where to start

Start by going through `src/bin/reference_example.py` line my line and following and understanding the functions called. It has relatively few dependencies within the repo and covers the core concepts. 

## License

This project is licensed under the Mozilla Public License Version 2.0. See the `LICENSE` file for details.
