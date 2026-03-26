import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Suppress oneDNN informational messages from TensorFlow builds.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

# Further reduce TF's Python-side logging.
try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass


def find_repo_root() -> Path:
    # Anchor on project files that should exist in the repo root.
    # This makes `--out models/...` deterministic even if you run from `./building`.
    import pyrootutils  # type: ignore[import-not-found]

    return Path(
        pyrootutils.find_root(search_from=__file__, indicator=["Cargo.toml", "laze-project.yml"])
    )


def build_lenet_keras() -> tf.keras.Model:
    # Classic LeNet-5 style model (Conv/AvgPool + Dense head).
    #
    # IMPORTANT: set `batch_size=1` and use fixed-size `Reshape` to avoid
    # dynamic shape ops (SHAPE / STRIDED_SLICE / PACK) that MicroFlow does not
    # support.
    inputs = tf.keras.Input(shape=(28, 28, 1), batch_size=1, name="input")

    x = tf.keras.layers.Conv2D(6, (5, 5), activation="relu", padding="valid")(inputs)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(16, (5, 5), activation="relu", padding="valid")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 4x4x16 -> 256
    x = tf.keras.layers.Reshape((256,), name="flatten_256")(x)
    x = tf.keras.layers.Dense(120, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(84, activation="relu", name="fc2")(x)
    logits = tf.keras.layers.Dense(10, name="fc3")(x)
    outputs = tf.keras.layers.Softmax()(logits)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lenet")


def representative_dataset(images: np.ndarray, samples: int):
    # Yields batches shaped exactly like the model input: [1, 28, 28, 1]
    for i in range(samples):
        img = images[i : i + 1].astype(np.float32)
        yield [img]


def export_iree_mlir(tflite_path: Path, mlir_path: Path) -> bool:
    # Try the IREE Python API first. Depending on your TF + iree versions, the
    # python API may hit a TensorFlow MLIR NameError. If that happens, fall
    # back to executing the `iree-import-tflite` module directly via `runpy`.
    # Docs: https://iree-python-api.readthedocs.io/en/latest/compiler/tools.html
    mlir_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import iree.compiler.tools.tflite as iree_tflite 

        # `import_only=True` writes textual MLIR if `output_file` is provided.
        iree_tflite.compile_file(
            str(tflite_path),
            import_only=True,
            output_file=str(mlir_path),
            output_format="mlir-text",
        )
        print("Wrote:", str(mlir_path))
        return True
    except Exception as e:
        print("IREE: Python API MLIR export failed, falling back:", repr(e))

    # Fallback: run the importer as a python module (no shelling out).
    try:
        import runpy
        import sys

        old_argv = sys.argv
        sys.argv = [
            "iree-import-tflite",
            str(tflite_path),
            "-o",
            str(mlir_path),
        ]
        runpy.run_module(
            "iree.tools.tflite.scripts.iree_import_tflite",
            run_name="__main__",
        )
        sys.argv = old_argv
        print("Wrote:", str(mlir_path))
        return True
    except Exception as e:
        print("IREE: MLIR export fallback also failed:", repr(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="Train + quantize LeNet to int8 TFLite.")
    parser.add_argument("--out", type=str, default="models/lenet_int8.tflite")
    parser.add_argument("--iree-mlir-out", type=str, default="models/lenet_int8.mlir")
    parser.add_argument("--no-iree-mlir", action="store_true")
    parser.add_argument(
        "--quantization",
        choices=["per-tensor", "per-channel"],
        default="per-tensor",
        help="Quantization granularity. MicroFlow FullyConnected requires per-tensor (QUANTS=1).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rep-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    repo_root = find_repo_root()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = np.expand_dims(x_train, axis=-1)  # [N, 28, 28, 1]
    x_test = np.expand_dims(x_test, axis=-1)

    model = build_lenet_keras()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=2,
    )

    # Quantization: PTQ with a representative dataset to calibrate int8 scales.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        x_train, min(args.rep_samples, x_train.shape[0])
    )

    # Full integer quantization so MicroFlow can consume INT8 tensors.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # MicroFlow's `fully_connected()` implementation expects Tensor2D<..., QUANTS=1>.
    # That means Dense weights must be per-tensor quantized (single scale/zero-point).
    #
    # Note: different TF versions/quantizers expose this knob under slightly
    # different attribute names. We set what we can, best-effort.
    disable_per_channel = args.quantization == "per-tensor"
    for attr in (
        "_experimental_disable_per_channel",
        "_experimental_disable_per_channel_quantization",
    ):
        if hasattr(converter, attr):
            setattr(converter, attr, disable_per_channel)

    # Some TF releases route through a “new quantizer”; ensure we don’t ignore
    # the per-channel disable setting by forcing the legacy path if available.
    for attr in ("experimental_new_quantizer", "_experimental_new_quantizer"):
        if hasattr(converter, attr):
            # Legacy quantizer tends to respect per-tensor better for some ops.
            setattr(converter, attr, False)

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print("Wrote:", str(out_path))

    if not args.no_iree_mlir:
        mlir_out = Path(args.iree_mlir_out)
        if not mlir_out.is_absolute():
            mlir_out = repo_root / mlir_out
        export_iree_mlir(out_path, mlir_out)

    # Quick sanity check: input/output dtypes should be int8.
    # We pass an explicit empty delegate list so that operator lists don't
    # include runtime-injected `DELEGATE` nodes during smoke-checking.
    interpreter = tf.lite.Interpreter(model_path=str(out_path), experimental_delegates=[])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("TFLite input dtype:", input_details["dtype"], "shape:", input_details["shape"])
    print(
        "TFLite output dtype:",
        output_details["dtype"],
        "shape:",
        output_details["shape"],
    )

    # Quantization sanity: detect per-channel quantized tensors (scales length > 1).
    # If these show up on FC weights, MicroFlow will not type-check.
    try:
        per_channel = []
        for td in interpreter.get_tensor_details():
            qp = td.get("quantization_parameters") or {}
            scales = qp.get("scales")
            if isinstance(scales, np.ndarray) and scales.size > 1:
                per_channel.append((td.get("name", "?"), int(scales.size)))
        if per_channel:
            print("Warning: per-channel quantization detected (name, scales_len):")
            for name, n in per_channel[:30]:
                print(" -", name, n)
    except Exception as e:  # pragma: no cover
        print("Warning: could not inspect quantization parameters:", repr(e))

    # Operator smoke-check: MicroFlow will only compile models containing a subset
    # of TFLite ops. We try to extract op names from the interpreter internals.
    try:
        ops = interpreter._get_ops_details()  # pylint: disable=protected-access
        op_names = [op.get("op_name") for op in ops]
        print("TFLite ops in order:", op_names)
    except Exception as e:  # pragma: no cover
        print("Warning: could not extract TFLite op names:", repr(e))

    # Smoke-run once with a representative sample.
    # Note: because we forced int8 input/output types, the interpreter expects
    # int8 tensors for the input.
    sample = x_test[0:1].astype(np.float32)
    in_scale, in_zero_point = input_details.get("quantization", (0.0, 0))
    if input_details["dtype"] == np.int8:
        # Quantize float32 -> int8 using the model's quantization parameters.
        in_scale = float(in_scale) if np.isscalar(in_scale) else np.array(in_scale, dtype=np.float32)
        in_zero_point = (
            int(in_zero_point)
            if np.isscalar(in_zero_point)
            else np.array(in_zero_point, dtype=np.int32)
        )
        # `in_scale` / `in_zero_point` are expected to broadcast over the input shape.
        q = np.round(sample / in_scale + in_zero_point).astype(np.int8)
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], q)
    else:
        interpreter.set_tensor(
            interpreter.get_input_details()[0]["index"],
            sample.astype(output_details["dtype"]),
        )

    interpreter.invoke()

    out = interpreter.get_tensor(output_details["index"])
    print("Output sample first value:", int(out.flatten()[0]))


if __name__ == "__main__":
    main()

