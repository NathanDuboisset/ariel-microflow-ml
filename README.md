## Tiny ML using Ariel OS and MicroFlow

This repo contains a few examples of how to use microflow to load tflite files and then run it on embedded devices using Ariel OS.

Using microflow and ariel os as the infrastructure, for a full Rust code

- MicroFlow repo: `https://github.com/matteocarnelos/microflow-rs`
- MicroFlow paper: `https://arxiv.org/pdf/2409.19432`

## Prerequisites

# Rust and toolchain

# Ariel OS

https://ariel-os.github.io/ariel-os/dev/docs/book/getting-started.html

# MicroFlow (fork)

This project uses a custom fork of MicroFlow with parallelization support.
Clone it **next to** this repo (the path `../microflow-rs` is expected by `Cargo.toml`):

```
git clone --branch feat/parallelization \
  https://github.com/NathanDuboisset/microflow-rs.git \
  ../microflow-rs
```

The dependency in `Cargo.toml` resolves to that local path:
```toml
microflow = { default-features = false, path = "../microflow-rs" }
```



# Training / saving models

First step is to train and save a model using python and TenserFlowas a tflite format
Now only supporting quantized models

For that, use uv from astral (https://docs.astral.sh/uv/ ) to manage packages and version ; then run the notebook corresponding to your desired model to generate / update the tflite file

# Rust link

Simple linking using the microflow macro mechanic gives access to these models, for use with Ariel OS

Then run using
```
laze build -b {your-board-ariel-id} run --features {lenet5q/mcunetq}
```

# Benchmarking
To see RAM / Flash usage

path for runtime is :
```
runtime_file_path = build/bin/{your-board-ariel-id}/cargo/thumbv8m.main-none-eabihf/release/ariel-microflow-ml
```

```
arm-none-eabi-size runtime_file_path
nm --print-size --size-sort --demangle=rust --radix=d runtime_file_path
```