#![no_main]
#![no_std]

use ariel_os::debug::{exit, log::info, ExitCode};
use ariel_os::time::Instant;
use microflow::model;
use nalgebra::SMatrix;

// Exactly one model feature overall (reject none and any multi-select).
#[cfg(not(any(feature = "lenet5", feature = "mcunet", feature = "lenet5q", feature = "mcunetq")))]
compile_error!("Enable exactly one model feature: lenet5 | mcunet | lenet5q | mcunetq");

#[cfg(any(
    all(feature = "lenet5", feature = "mcunet"),
    all(feature = "lenet5", feature = "lenet5q"),
    all(feature = "lenet5", feature = "mcunetq"),
    all(feature = "mcunet", feature = "lenet5q"),
    all(feature = "mcunet", feature = "mcunetq"),
    all(feature = "lenet5q", feature = "mcunetq"),
))]
compile_error!("Enable exactly one model feature: lenet5 | mcunet | lenet5q | mcunetq");


#[cfg(feature = "lenet5")]
#[model("models/lenet5.tflite")]
struct MyModel;

#[cfg(feature = "mcunet")]
#[model("models/mcunet.tflite")]
struct MyModel;

#[cfg(feature = "lenet5q")]
#[model("models/lenet5_quantized.tflite")]
struct MyModel;

#[cfg(feature = "mcunetq")]
#[model("models/mcunet_quantized.tflite")]
struct MyModel;

#[cfg(any(feature = "lenet5", feature = "lenet5q"))]
fn make_input_sample() -> microflow::buffer::Buffer4D<f32, 1, 28, 28, 1> {
    type Img = SMatrix<[f32; 1], 28, 28>;
    let img: Img = SMatrix::from_fn(|r, c| {
        let v = if ((r + c) & 1) == 0 { 0.5 } else { 0.0 };
        [v]
    });
    [img]
}

#[cfg(any(feature = "mcunet", feature = "mcunetq"))]
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


#[ariel_os::task(autostart)]
async fn main() {
    info!(
        "microflow on {} board.",
        ariel_os::buildinfo::BOARD
    );
    #[cfg(feature = "lenet5")]
    info!("Model: lenet5 (models/lenet5.tflite)");
    #[cfg(feature = "lenet5q")]
    info!("Model: lenet5_quantized (models/lenet5_quantized.tflite)");
    #[cfg(feature = "mcunet")]
    info!("Model: mcunet (models/mcunet.tflite)");
    #[cfg(feature = "mcunetq")]
    info!("Model: mcunet_quantized (models/mcunet_quantized.tflite)");

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
    info!("Inference timing: runs={} total_us={} avg_us={}", RUNS, total_us, avg_us);
    info!("Last predicted class={}", last_predicted_class);

    exit(ExitCode::SUCCESS);
}
