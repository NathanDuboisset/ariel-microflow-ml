#![no_main]
#![no_std]

use ariel_os::debug::{exit, log::info, ExitCode};
use ariel_os::time::{Instant, TICK_HZ};
use microflow::model;
use nalgebra::SMatrix;

#[model("models/lenet_int8.tflite")]
struct LeNet;

const BENCH_ITERS: u64 = 500;

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

    // Timing: simple start/finish using Ariel OS time (embassy-time based).
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = LeNet::predict(core::hint::black_box(input));
    }
    let end = Instant::now();

    let dt_ticks: u64 = end.as_ticks().wrapping_sub(start.as_ticks());
    let mean_ticks = dt_ticks / BENCH_ITERS;
    let dt_us: u64 = dt_ticks.saturating_mul(1_000_000) / (TICK_HZ as u64);
    let mean_us: u64 = mean_ticks.saturating_mul(1_000_000) / (TICK_HZ as u64);

    info!(
        "Inference timing: iters={} total_ticks={} total_us={} mean_ticks={} mean_us={}",
        BENCH_ITERS,
        dt_ticks,
        dt_us,
        mean_ticks,
        mean_us
    );

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
