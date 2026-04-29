#![no_main]
#![no_std]

use ariel_os::debug::{ExitCode, exit, log::info};
use ariel_os::time::Instant;
use microflow::model;

mod multicore_backend;
#[cfg(any(feature = "lenet5qtf", feature = "lenet5qtfdualcore", feature = "lenet5qtorch"))]
#[path = "../samples/lenet/sample.rs"]
mod lenet_samples;
#[cfg(any(feature = "mobilenetv1", feature = "mobilenetv1dualcore"))]
#[path = "../samples/mobilenetv1/sample.rs"]
mod mobilenet_samples;

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

#[ariel_os::thread(autostart, priority = 2,stacksize = 320000)]
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
    #[cfg(any(feature = "lenet5qtf", feature = "lenet5qtfdualcore", feature = "lenet5qtorch"))]
    let samples = [lenet_samples::digit_0(), lenet_samples::digit_1()];
    #[cfg(any(feature = "mobilenetv1", feature = "mobilenetv1dualcore"))]
    let samples = [mobilenet_samples::PERSON, mobilenet_samples::NO_PERSON];

    for i_run in 0..RUNS {
        let sample_idx = (i_run % samples.len() as u64) as usize;
        let input = samples[sample_idx];
        info!("Running inference... run {} sample_{}", i_run, sample_idx);
        let start = Instant::now().as_micros();
        let prediction = MyModel::predict_quantized(input);
        let end = Instant::now().as_micros();

        total_us = total_us.wrapping_add(end.wrapping_sub(start));
        #[cfg(any(feature = "mobilenetv1", feature = "mobilenetv1dualcore"))]
        {
            let no_person_score = prediction[(0, 0)];
            let person_score = prediction[(0, 1)];
            let person_detected = person_score > no_person_score;
            info!(
                "sample_{} => person_detected={} (scores: no_person={}, person={})",
                sample_idx,
                person_detected,
                no_person_score,
                person_score
            );
        }

        #[cfg(any(feature = "lenet5qtf", feature = "lenet5qtfdualcore", feature = "lenet5qtorch"))]
        {
            let mut predicted_class: usize = 0;
            let mut best_val = prediction[(0, 0)];
            for c in 1..10 {
                let v = prediction[(0, c)];
                if v > best_val {
                    best_val = v;
                    predicted_class = c;
                }
            }
            info!("sample_{} => predicted_class={}", sample_idx, predicted_class);
        }
    }

    let avg_us = total_us / RUNS;
    info!(
        "Inference timing: runs={} total_us={} avg_us={}",
        RUNS, total_us, avg_us
    );

    exit(ExitCode::SUCCESS);
}
