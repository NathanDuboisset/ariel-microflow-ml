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
git clone --branch ariel-os-compatibility \
  https://github.com/NathanDuboisset/microflow-rs.git \
  ../microflow-rs
```

The dependency in `Cargo.toml` resolves to that local path:
```toml
microflow = { default-features = false, path = "../microflow-rs" }
```

Change linking in `Cargo.toml` if you clone it elsewhere

# Training / saving models

First step is to train and save a model using python and TenserFlowas a tflite format
Now only supporting quantized models.

For that, use uv from astral (https://docs.astral.sh/uv/ ) to manage packages and version ; then run the notebook corresponding to your desired model to generate / update the tflite file.

Since torch and tensorflow can be conflicting on versions, there is two folders to generate, each having its own env,
In a folder, run 
```
uv sync
```
then run notebooks with the .venv created in that folder

In addition, we provide the mobilenetv1 model file, coming from the microflow repository : models need to be made "microflow compatible", by using only implemented operations, so we need to bake some operations(like padding or batch normalisation) into the weights.

# Rust link

Simple linking using the microflow macro mechanic gives access to these models, for use with Ariel OS

Then run using
```
laze build -b {your-board-ariel-id} run --features {lenet5qtf/lenet5qtorch/...}
```

Measurement for RAM are noted under, and are impactful in `laze-project.yml`, `src/multicore_backend.rs`, `src/main.rs`

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

# Benchmarks

Here are the current results by model, depending on the use of dual core or not.
Measurement are made and tested using a rpi-pico2-w
For model that have a torch and tensorflow version, it should not make any difference so we run tests with the tensorflow version.
Timings are averaged over 10 runs


| model       | cores     | time (ms) | main thread stack (ko) | helper stack (ko) |
|-------------|-----------|-----------|-------------------------|-------------------|
| lenet5q     | mono core | 53        | 14                      | -                 |
| lenet5q     | dual core | 41        | 14                      | 1                 |
| mobilenetv1 | mono core | 3537      | 320                     | -                 |
| mobilenetv1 | dual core | 2267      | 320                     | 17                |


## Current state of work / TODO

- explore threading options :
  - one "helper" thread ?
  - optimizations on the use of ariel ?
  - to what size should microflow(see branch for parrallelization) delegate operations ? small operations mean too much overhead etc : per channel jobs, per row ? delegate activations, reshapes etc ?

- implement operators in microflow
  - other operators for more complex network : split, add, mean,
  - better quantization (per channel)
  - only supports quantized models