"""
    Script for training model on TensorFlow 2.0.
"""

import os
import logging
import argparse
import numpy as np
import random
import tensorflow as tf
from common.logger_utils import initialize_logging
from tensorflow2.tf2cv.model_provider import get_model
from tensorflow2.dataset_utils import get_dataset_metainfo, get_train_data_source, get_val_data_source


def add_train_cls_parser_arguments(parser):
    """
    Create python script parameters (for training/classification specific subpart).

    Parameters:
    ----------
    parser : ArgumentParser
        ArgumentParser instance.
    """
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model from github repo")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters if not None")
    parser.add_argument(
        "--resume-state",
        type=str,
        default="",
        help="resume from previously saved optimizer state if not None")

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
        "--num-epochs",
        type=int,
        default=120,
        help="number of training epochs.")
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="starting epoch for resuming, default is 1 for new training")
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="current attempt number for training")

    parser.add_argument(
        "--optimizer-name",
        type=str,
        default="nag",
        help="optimizer name")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate")
    parser.add_argument(
        "--lr-mode",
        type=str,
        default="cosine",
        help="learning rate scheduler mode. options are step, poly and cosine")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="decay rate of learning rate")
    parser.add_argument(
        "--lr-decay-period",
        type=int,
        default=0,
        help="interval for periodic learning rate decays. default is 0 to disable")
    parser.add_argument(
        "--lr-decay-epoch",
        type=str,
        default="40,60",
        help="epoches at which learning rate decays")
    parser.add_argument(
        "--target-lr",
        type=float,
        default=1e-8,
        help="ending learning rate")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum value for optimizer")
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        help="weight decay rate")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="number of batches to wait before logging")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=4,
        help="saving parameters epoch interval, best model will always be saved")
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
        "--seed",
        type=int,
        default=-1,
        help="Random seed to be fixed")
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


def parse_args():
    """
    Parse python script parameters (common part).

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for image classification/segmentation (TensorFlow 2.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K",
        help="dataset name. options are ImageNet1K, CUB200_2011, CIFAR10, CIFAR100, SVHN")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data"),
        help="path to working directory only for dataset root path preset")

    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir_path=args.work_dir)

    add_train_cls_parser_arguments(parser)

    args = parser.parse_args()
    return args


def init_rand(seed):
    if seed <= 0:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def main():
    """
    Main body of script.
    """
    args = parse_args()
    args.seed = init_rand(seed=args.seed)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    data_format = "channels_last"
    tf.keras.backend.set_image_data_format(data_format)

    model = args.model
    net = get_model(model, data_format=data_format)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = net(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = net(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)
    assert (ds_metainfo.ml_type != "imgseg") or (args.batch_size == 1)
    # assert (ds_metainfo.ml_type != "imgseg") or args.disable_cudnn_autotune

    batch_size = args.batch_size

    train_data, train_img_count = get_train_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        data_format=data_format)
    val_data, val_img_count = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        data_format=data_format)

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        for images, labels in train_data:
            train_step(images, labels)
            # break

        for test_images, test_labels in val_data:
            test_step(test_images, test_labels)
            # break

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        logging.info(template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            test_loss.result(),
            test_accuracy.result() * 100))

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == "__main__":
    main()
