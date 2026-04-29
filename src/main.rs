#![no_main]
#![no_std]

use ariel_os::debug::{ExitCode, exit, log::info};
use ariel_os::time::Instant;
use microflow::model;
use microflow::buffer::Buffer2D;
use nalgebra::SMatrix;

mod multicore_backend;

// Exactly one model feature overall (reject none and any multi-select).
#[cfg(not(any(
    feature = "lenet5qtf",
    feature = "lenet5qtorch",
    feature = "mobilenetv1",
    feature = "lenet5qtfdualcore",
    feature = "mobilenetv1dualcore"
)))]
compile_error!("Enable exactly one model feature: lenet5qtf | lenet5qtfdualcore | lenet5qtorch | mobilenetv1");

#[cfg(feature = "lenet5qtf")]
#[model("models/lenet5_quantized.tflite")]
struct MyModel;

#[cfg(feature = "lenet5qtfdualcore")]
#[model("models/lenet5_quantized.tflite", crate::multicore_backend::ArielBackend)]
struct MyModel;

#[cfg(feature = "lenet5qtorch")]
#[model("models/lenet5_quantized_torch.tflite")]
struct MyModel;

#[cfg(feature = "mobilenetv1")]
#[model("models_provided/mobilenetv1.tflite")]
struct MyModel;

#[cfg(feature = "mobilenetv1dualcore")]
#[model("models_provided/mobilenetv1.tflite", crate::multicore_backend::ArielBackend)]
struct MyModel;

#[cfg(any(feature = "lenet5qtf", feature = "lenet5qtfdualcore", feature = "lenet5qtorch"))]
fn make_input_sample() -> microflow::buffer::Buffer4D<f32, 1, 28, 28, 1> {
    type Img = SMatrix<[f32; 1], 28, 28>;
    let img: Img = SMatrix::from_fn(|r, c| {
        let v = if ((r + c) & 1) == 0 { 0.5 } else { 0.0 };
        [v]
    });
    [img]
}

#[cfg(any(feature = "mobilenetv1", feature = "mobilenetv1dualcore"))]
fn make_input_sample() -> microflow::buffer::Buffer4D<f32, 1, 96, 96, 1> {
    let input = Buffer2D::<[f32; 1], 96, 96>::from_element([0.5_f32]);
    [input]
}

#[ariel_os::thread(autostart, priority = 2,stacksize = 390000)]
fn main() {
    let my_id = ariel_os::thread::current_tid().unwrap();
    let core = ariel_os::thread::core_id();
    info!("microflow on {} board and thread [{:?}] core [{:?}]", ariel_os::buildinfo::BOARD, my_id, core);
    #[cfg(any(feature = "lenet5qtf", feature = "lenet5qtfdualcore"))]
    info!("Model: lenet5_quantized (models/lenet5_quantized.tflite)");
    #[cfg(feature = "lenet5qtorch")]
    info!("Model: lenet5_quantized_torch (models/lenet5_quantized_torch.tflite)");
    #[cfg(feature = "mobilenetv1")]
    info!("Model: mobilenetv1 (models/mobilenetv1.tflite)");

    const RUNS: u64 = 10;
    let mut total_us: u64 = 0;
    let input = make_input_sample();

    for i_run in 0..RUNS {
        info!("Running inference... run {}", i_run);
        let start = Instant::now().as_micros();
        let prediction = MyModel::predict(input);
        let end = Instant::now().as_micros();

        total_us = total_us.wrapping_add(end.wrapping_sub(start));
        #[cfg(any(feature = "mobilenetv1", feature = "mobilenetv1dualcore"))]
        {
            if prediction.nrows() > 0 && prediction.ncols() > 1 {
                info!(
                    "Prediction shape={}x{}, first_values=({}, {})",
                    prediction.nrows(),
                    prediction.ncols(),
                    prediction[(0, 0)],
                    prediction[(0, 1)]
                );
            } else {
                info!(
                    "Prediction shape={}x{}",
                    prediction.nrows(),
                    prediction.ncols()
                );
            }
        }

        #[cfg(any(feature = "lenet5qtf", feature = "lenet5qtfdualcore", feature = "lenet5qtorch"))]
        {
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
            info!("Predicted class={}", predicted_class);
        }
    }

    let avg_us = total_us / RUNS;
    info!(
        "Inference timing: runs={} total_us={} avg_us={}",
        RUNS, total_us, avg_us
    );

    exit(ExitCode::SUCCESS);
}
