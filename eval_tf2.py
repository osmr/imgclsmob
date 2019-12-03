"""
    Script for evaluating trained model on TensorFlow 2.0 (validate/test).
"""

import os
import logging
import argparse
import tensorflow as tf
from common.logger_utils import initialize_logging
from tensorflow2.utils import prepare_model


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification (TensorFlow 2.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../imgclsmob_data/imagenet",
        help="training and validation pictures to use")

    parser.add_argument(
        "--data-format",
        type=str,
        default="channels_last",
        help="ordering of the dimensions in tensors. options are channels_last and channels_first")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters if not None")
    parser.add_argument(
        "--calc-flops",
        dest="calc_flops",
        action="store_true",
        help="calculate FLOPs")

    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="size of the input for model")
    parser.add_argument(
        "--resize-inv-factor",
        type=float,
        default=0.875,
        help="inverted ratio for input image crop")

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of gpus to use")
    parser.add_argument(
        "-j",
        "--num-data-workers",
        dest="num_workers",
        default=4,
        type=int,
        help="number of preprocessing workers")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="training batch size per device (CPU/GPU)")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="directory of saved models and log-files")
    parser.add_argument(
        "--logging-file-name",
        type=str,
        default="train.log",
        help="filename of training log")

    parser.add_argument(
        "--log-packages",
        type=str,
        default="tensorflow-gpu",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="tensorflow-gpu, tensorpack",
        help="list of pip packages for logging")

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="show progress bar")

    args = parser.parse_args()
    return args


def main():
    """
    Main body of script.
    """
    args = parse_args()

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    data_format = "channels_last"
    tf.keras.backend.set_image_data_format(data_format)

    use_cuda = (args.num_gpus > 0)

    batch_size = args.batch_size
    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        batch_size=batch_size,
        use_cuda=use_cuda)
    assert (hasattr(net, "in_size"))
    # input_image_size = net.in_size

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def test_step(images, labels):
        predictions = net(images)
        test_accuracy(labels, predictions)

    data_dir = args.data_dir
    val_dir = os.path.join(data_dir, "val")

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1.0 / 255))
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False)

    if args.show_progress:
        from tqdm import tqdm
        val_generator = tqdm(val_generator)

    # total_img_count = val_generator.n
    total_img_count = 50000
    processed_img_count = 0
    for test_images, test_labels in val_generator:
        if processed_img_count >= total_img_count:
            break
        test_step(test_images, test_labels)
        processed_img_count += batch_size

    logging.info("Test Accuracy: {}".format(test_accuracy.result() * 100))


if __name__ == "__main__":
    main()
