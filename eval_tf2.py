"""
    Script for evaluating trained model on TensorFlow 2.0 (validate/test).
"""

import os
import logging
import argparse
import tensorflow as tf
from common.logger_utils import initialize_logging
from tensorflow2.utils import load_image_imagenet1k_val, img_normalization, prepare_model

import keras_preprocessing as keras_prep
keras_prep.image.iterator.load_img = load_image_imagenet1k_val


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification (TensorFlow 2.0 / ImageNet-1K)",
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
        default="tensorflow-gpu",
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

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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
    input_image_size = net.in_size
    resize_inv_factor = args.resize_inv_factor

    val_top1_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="val_top1_acc")
    val_top5_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="val_top6_acc")

    data_dir = args.data_dir
    val_dir = os.path.join(data_dir, "val")

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=img_normalization,
        data_format=data_format)
    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=input_image_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        interpolation="bilinear:" + str(resize_inv_factor))
    val_ds = tf.data.Dataset.from_generator(
        generator=lambda: val_generator,
        output_types=(tf.float32, tf.float32))

    # if args.show_progress:
    #     from tqdm import tqdm
    #     val_generator = tqdm(val_generator)

    total_img_count = val_generator.n
    processed_img_count = 0
    for test_images, test_labels in val_ds:

        # import cv2
        # import numpy as np
        # img = test_images[0].numpy().copy()
        # cv2.imshow(winname="img", mat=img.astype(np.uint8))
        # cv2.waitKey()

        predictions = net(test_images)
        val_top1_acc.update_state(test_labels, predictions)
        val_top5_acc.update_state(test_labels, predictions)
        processed_img_count += len(test_images)
        if processed_img_count >= total_img_count:
            break

    top1_err = 1.0 - val_top1_acc.result()
    top5_err = 1.0 - val_top5_acc.result()
    logging.info("Test: err-top1={top1:.4f} ({top1}), err-top5={top5:.4f} ({top5})".format(
        top1=top1_err,
        top5=top5_err))


if __name__ == "__main__":
    main()
