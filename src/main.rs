#![no_main]
#![no_std]

use ariel_os::debug::{exit, log::info, ExitCode};
use microflow::model;
use nalgebra::SMatrix;

#[model("models/lenet_int8.tflite")]
struct LeNet;

fn make_input_sample() -> microflow::buffer::Buffer4D<f32, 1, 28, 28, 1> {
    type Img = SMatrix<[f32; 1], 28, 28>;
    let img: Img = SMatrix::from_fn(|r, c| {
        // A simple deterministic pattern. The exact values don't matter for toolchain validation.
        let v = if ((r + c) & 1) == 0 { 0.5 } else { 0.0 };
        [v]
    });
    [img]
}

#[ariel_os::task(autostart)]
async fn main() {
    info!(
        "Hello from main()! Running on a {} board.",
        ariel_os::buildinfo::BOARD
    );

    let input = make_input_sample();
    let prediction = LeNet::predict(input);

    let predicted_class = if prediction.nrows() == 1 {
        // Expected for classifiers: [1, num_classes]
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
        // Fallback: [num_classes, 1]
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
        // Generic fallback: flatten across the whole matrix.
        prediction.iamax_full().1
    };

    info!("Predicted class: {}", predicted_class);

    exit(ExitCode::SUCCESS);
}
