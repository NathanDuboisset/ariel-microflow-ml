import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def build_lenet_keras() -> tf.keras.Model:
    # LeNet-style model using ops that MicroFlow supports after conversion:
    # - Conv2D + fused activation (use activation inside the layer)
    # - AveragePooling2D
    # - Avoid FullyConnected (MicroFlow fully_connected has QUANTS=1 restriction)
    # - Use Conv2D as classifier head
    # - Softmax as a dedicated op (via Keras Softmax layer)
    #
    # IMPORTANT: set `batch_size=1` to avoid dynamic shape ops (SHAPE /
    # STRIDED_SLICE / PACK) that MicroFlow does not support.
    inputs = tf.keras.Input(shape=(28, 28, 1), batch_size=1, name="input")

    x = tf.keras.layers.Conv2D(6, (5, 5), activation="relu", padding="valid")(inputs)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(16, (5, 5), activation="relu", padding="valid")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Classifier head without Dense:
    # Current feature map is 4x4x16. A 4x4 valid conv to 10 channels yields 1x1x10.
    x = tf.keras.layers.Conv2D(10, (4, 4), activation=None, padding="valid", name="head")(x)
    x = tf.keras.layers.Reshape((10,), name="logits_10")(x)
    outputs = tf.keras.layers.Softmax()(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lenet")


def representative_dataset(images: np.ndarray, samples: int):
    # Yields batches shaped exactly like the model input: [1, 28, 28, 1]
    for i in range(samples):
        img = images[i : i + 1].astype(np.float32)
        yield [img]


def main():
    parser = argparse.ArgumentParser(description="Train + quantize LeNet to int8 TFLite.")
    parser.add_argument("--out", type=str, default="models/lenet_int8.tflite")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rep-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    out_path = Path(args.out)
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
    # MicroFlow's `fully_connected()` implementation only supports tensors with
    # a single quantization parameter (QUANTS=1). Many TFLite exports use
    # per-channel weight quantization for Dense layers (QUANTS > 1), which
    # breaks type-checking during the MicroFlow macro expansion.
    converter._experimental_disable_per_channel = True

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)

    # Quick sanity check: input/output dtypes should be int8.
    # Use an explicit empty delegate list so operator lists don't include
    # runtime-injected `DELEGATE` nodes during smoke-checking.
    interpreter = tf.lite.Interpreter(model_path=str(out_path), experimental_delegates=[])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("Wrote:", str(out_path))
    print("TFLite input dtype:", input_details["dtype"], "shape:", input_details["shape"])
    print("TFLite output dtype:", output_details["dtype"], "shape:", output_details["shape"])

    # Operator smoke-check (approximate): list operator names the interpreter sees.
    # If MicroFlow fails at Rust compile-time, we'll tighten this check to use
    # the raw flatbuffer operator table.
    try:
        ops = interpreter._get_ops_details()  # pylint: disable=protected-access
        op_names = [op.get("op_name") for op in ops]
        print("TFLite ops in order (interpreter view):", op_names)
    except Exception as e:  # pragma: no cover
        print("Warning: could not extract TFLite op names:", repr(e))

    # Smoke-run once with a representative sample.
    sample = x_test[0:1].astype(np.float32)
    if input_details["dtype"] == np.int8:
        in_scale, in_zero_point = input_details.get("quantization", (0.0, 0))
        in_scale = (
            float(in_scale) if np.isscalar(in_scale) else np.array(in_scale, dtype=np.float32)
        )
        in_zero_point = (
            int(in_zero_point)
            if np.isscalar(in_zero_point)
            else np.array(in_zero_point, dtype=np.int32)
        )
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

