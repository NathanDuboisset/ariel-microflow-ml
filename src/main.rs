    #![no_main]
#![no_std]

use ariel_os::debug::{exit, log::info, ExitCode};
use ariel_os::time::Instant;
use microflow::model;
use nalgebra::SMatrix;

// Exactly one architecture feature overall.
#[cfg(not(any(feature = "lenet5", feature = "mcunet")))]
compile_error!("Enable exactly one model feature: lenet5 | mcunet");

#[cfg(all(feature = "lenet5", feature = "mcunet"))]
compile_error!("Enable exactly one model feature: lenet5 | mcunet");

#[cfg(feature = "lenet5")]
#[model("models/lenet5_quantized.tflite")]
struct MyModel;
#[cfg(feature = "mcunet")]
#[model("models/mcunet_quantized.tflite")]

struct MyModel;
#[cfg(feature = "lenet5")]
fn make_input_sample() -> microflow::buffer::Buffer4D<f32, 1, 28, 28, 1> {
    type Img = SMatrix<[f32; 1], 28, 28>;
    let img: Img = SMatrix::from_fn(|r, c| {
        let v = if ((r + c) & 1) == 0 { 0.5 } else { 0.0 };
        [v]
    });
    [img]
}

#[cfg(feature = "mcunet")]
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

#[cfg_attr(feature = "lenet5", ariel_os::thread(autostart, priority = 2, stacksize = 4000))]
#[cfg_attr(feature = "mcunet", ariel_os::thread(autostart, priority = 2, stacksize = 100000))]
fn main() {
    info!(
        "microflow on {} board.",
        ariel_os::buildinfo::BOARD
    );
    #[cfg(feature = "lenet5")]
    info!("Model: lenet5_quantized (models/lenet5_quantized.tflite)");
    #[cfg(feature = "mcunet")]
    info!("Model: mcunet_quantized (models/mcunet_quantized.tflite)");

    let start = Instant::now();
    let input = make_input_sample();
    let prediction = MyModel::predict(core::hint::black_box(input));
    let end = Instant::now();

    let dt_us: u64 = end.as_micros().wrapping_sub(start.as_micros());

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

    info!("Inference timing: total_us={}", dt_us);
    info!("Last predicted class={}", predicted_class);

    exit(ExitCode::SUCCESS);
}
