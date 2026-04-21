#![no_main]
#![no_std]

use ariel_os::debug::{ExitCode, exit, log::info};
use ariel_os::time::Instant;
use microflow::model;
use nalgebra::SMatrix;

mod multicore_backend;

// Exactly one model feature overall (reject none and any multi-select).
#[cfg(not(any(
    feature = "lenet5qtf",
    feature = "mcunetqtf",
    feature = "lenet5qtorch",
    feature = "mcunetqtorch"
)))]
compile_error!("Enable exactly one model feature: lenet5qtf | mcunetqtf | lenet5qtorch | mcunetqtorch");

#[cfg(feature = "lenet5qtf")]
#[model("models/lenet5_quantized.tflite", crate::multicore_backend::ArielBackend)]
struct MyModel;

#[cfg(feature = "mcunetqtf")]
#[model("models/mcunet_quantized.tflite", crate::multicore_backend::ArielBackend)]
struct MyModel;

#[cfg(feature = "lenet5qtorch")]
#[model("models/lenet5_quantized_torch.tflite", crate::multicore_backend::ArielBackend)]
struct MyModel;

#[cfg(feature = "mcunetqtorch")]
#[model("models/mcunet_quantized_torch.tflite", crate::multicore_backend::ArielBackend)]
struct MyModel;

#[cfg(any(feature = "lenet5qtf", feature = "lenet5qtorch"))]
fn make_input_sample() -> microflow::buffer::Buffer4D<f32, 1, 28, 28, 1> {
    type Img = SMatrix<[f32; 1], 28, 28>;
    let img: Img = SMatrix::from_fn(|r, c| {
        let v = if ((r + c) & 1) == 0 { 0.5 } else { 0.0 };
        [v]
    });
    [img]
}

#[cfg(any(feature = "mcunetqtf", feature = "mcunetqtorch"))]
fn make_input_sample() -> microflow::buffer::Buffer4D<f32, 1, 32, 32, 3> {
    type Img = SMatrix<[f32; 3], 32, 32>;
    let img: Img = SMatrix::from_fn(|r, c| {
        let base: f32 = if ((r + c) & 1) == 0 { 0.6_f32 } else { 0.1_f32 };
        let r = base;
        let g = (base * 0.7_f32).min(1.0_f32);
        let b = (base * 0.4_f32).min(1.0_f32);
        [r, g, b]
    });
    [img]
}

#[ariel_os::thread(autostart, priority = 2,stacksize = 128000)]
fn main() {
    info!("microflow on {} board and core {:?}", ariel_os::buildinfo::BOARD, ariel_os::thread::current_tid().unwrap());
    #[cfg(feature = "lenet5qtf")]
    info!("Model: lenet5_quantized (models/lenet5_quantized.tflite)");
    #[cfg(feature = "lenet5qtorch")]
    info!("Model: lenet5_quantized_torch (models/lenet5_quantized_torch.tflite)");
    #[cfg(feature = "mcunetqtf")]
    info!("Model: mcunet_quantized (models/mcunet_quantized.tflite)");
    #[cfg(feature = "mcunetqtorch")]
    info!("Model: mcunet_quantized_torch (models/mcunet_quantized_torch.tflite)");

    const RUNS: u64 = 10;
    let mut total_us: u64 = 0;
    let mut last_predicted_class: usize = 0;

    let input = make_input_sample();

    for _ in 0..RUNS {
        let start = Instant::now().as_micros();
        let prediction = MyModel::predict(input);
        let end = Instant::now().as_micros();

        total_us = total_us.wrapping_add(end.wrapping_sub(start));

        let predicted_class = if prediction.nrows() == 1 {
            let mut best_col: usize = 0;
            let mut best_val = prediction[(0, 0)];
            for c in 1..prediction.ncols() {
                let v = prediction[(0, c)];
                if v > best_val {
                    best_val = v;
                    best_col = c;
                }
            }
            best_col
        } else if prediction.ncols() == 1 {
            let mut best_row: usize = 0;
            let mut best_val = prediction[(0, 0)];
            for r in 1..prediction.nrows() {
                let v = prediction[(r, 0)];
                if v > best_val {
                    best_val = v;
                    best_row = r;
                }
            }
            best_row
        } else {
            prediction.iamax_full().1
        };

        last_predicted_class = predicted_class;
    }

    let avg_us = total_us / RUNS;
    info!(
        "Inference timing: runs={} total_us={} avg_us={}",
        RUNS, total_us, avg_us
    );
    info!("Last predicted class={}", last_predicted_class);

    exit(ExitCode::SUCCESS);
}
