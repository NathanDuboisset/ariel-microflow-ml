## Tiny ML using Ariel OS and MicroFlow

This repo contains a few examples of how to use microflow to load tflite files and then run it on embedded devices using Ariel OS.

# Training / saving models

First step is to train and save a model using python and TenserFlow(version at most TODO) as a tflite format
Now only supporting quantized models

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