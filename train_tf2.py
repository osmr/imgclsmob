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


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (TensorFlow 2.0)",
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
        help="enable using pretrained model from github repo")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters if not None")

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
        "--num-epochs",
        type=int,
        default=120,
        help="number of training epochs")
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="starting epoch for resuming, default is 1 for new training")
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="current number of training")

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
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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

    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    batch_size = args.batch_size
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=(1.0 / 255),
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1.0 / 255))
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="binary")

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        for images, labels in train_generator:
            train_step(images, labels)
            # break

        for test_images, test_labels in val_generator:
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
