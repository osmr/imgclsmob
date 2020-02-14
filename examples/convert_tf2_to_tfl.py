"""
    Script for converting trained model from TensorFlow 2.0 to TensorFlow Lite.
"""

import argparse
import numpy as np
import tensorflow as tf
from tf2cv.model_provider import get_model as tf2cv_get_model


def parse_args():
    """
    Create python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Converting a model from TensorFlow 2.0 to TensorFlow Lite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--input-shape",
        type=int,
        default=(1, 224, 224, 3),
        help="input tensor shape")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="path to dir for output TFL file")

    args = parser.parse_args()
    return args


def main():
    """
    Main body of script.
    """
    args = parse_args()

    model = tf2cv_get_model(args.model, pretrained=True)
    x = tf.zeros(shape=args.input_shape)
    _ = model.predict(x)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the TensorFlow Lite model on random input data.
    input_shape = input_details[0]["shape"]
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tflite_results = interpreter.get_tensor(output_details[0]["index"])

    # Test the TensorFlow model on random input data.
    tf_results = model(tf.constant(input_data))

    # Compare the result.
    for tf_result, tflite_result in zip(tf_results, tflite_results):
        np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)

    if args.output_dir is not None:
        open("{}/{}.tflite".format(args.output_dir, args.model), "wb").write(tflite_model)

    print("All OK.")


if __name__ == "__main__":
    main()
