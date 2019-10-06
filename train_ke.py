"""
    Script for training model on Keras.
"""

import argparse
import time
import logging
import os
import numpy as np
import random
import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import mxnet as mx
from common.logger_utils import initialize_logging
from keras_.utils import prepare_ke_context, prepare_model, get_data_rec, get_data_generator, backend_agnostic_compile


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (Keras)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--rec-train",
        type=str,
        default="../imgclsmob_data/imagenet_rec/train.rec",
        help="the training data")
    parser.add_argument(
        "--rec-train-idx",
        type=str,
        default="../imgclsmob_data/imagenet_rec/train.idx",
        help='the index of training data')
    parser.add_argument(
        "--rec-val",
        type=str,
        default="../imgclsmob_data/imagenet_rec/val.rec",
        help="the validation data")
    parser.add_argument(
        "--rec-val-idx",
        type=str,
        default="../imgclsmob_data/imagenet_rec/val.idx",
        help="the index of validation data")

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
        "--dtype",
        type=str,
        default="float32",
        help="data type for training")
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
        default="keras",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="keras, keras-mxnet, keras-applications, keras-preprocessing",
        help="list of pip packages for logging")
    args = parser.parse_args()
    return args


def init_rand(seed):
    if seed <= 0:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    return seed


def prepare_trainer(net,
                    optimizer_name,
                    momentum,
                    lr,
                    num_gpus,
                    state_file_path=None):

    optimizer_name = optimizer_name.lower()
    if (optimizer_name == "sgd") or (optimizer_name == "nag"):
        optimizer = keras.optimizers.SGD(
            lr=lr,
            momentum=momentum,
            nesterov=(optimizer_name == "nag"))
    else:
        raise ValueError("Usupported optimizer: {}".format(optimizer_name))

    backend_agnostic_compile(
        model=net,
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=[keras.metrics.categorical_accuracy, keras.metrics.top_k_categorical_accuracy],
        num_gpus=num_gpus)

    if (state_file_path is not None) and state_file_path and os.path.exists(state_file_path):
        net = load_model(filepath=state_file_path)
    return net


def train_net(net,
              train_gen,
              val_gen,
              train_num_examples,
              val_num_examples,
              num_epochs,
              checkpoint_filepath,
              start_epoch1):
    checkpointer = ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_best_only=True)

    tic = time.time()

    net.fit_generator(
        generator=train_gen,
        samples_per_epoch=train_num_examples,
        epochs=num_epochs,
        verbose=True,
        callbacks=[checkpointer],
        validation_data=val_gen,
        validation_steps=val_num_examples,
        class_weight=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=(start_epoch1 - 1))

    logging.info("Time cost: {:.4f} sec".format(
        time.time() - tic))


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

    batch_size = prepare_ke_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip())
    num_classes = net.classes if hasattr(net, "classes") else 1000
    input_image_size = net.in_size if hasattr(net, "in_size") else (args.input_size, args.input_size)

    train_data, val_data = get_data_rec(
        rec_train=args.rec_train,
        rec_train_idx=args.rec_train_idx,
        rec_val=args.rec_val,
        rec_val_idx=args.rec_val_idx,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_image_size=input_image_size,
        resize_inv_factor=args.resize_inv_factor)
    train_gen = get_data_generator(
        data_iterator=train_data,
        num_classes=num_classes)
    val_gen = get_data_generator(
        data_iterator=val_data,
        num_classes=num_classes)

    net = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        momentum=args.momentum,
        lr=args.lr,
        num_gpus=args.num_gpus,
        state_file_path=args.resume_state)

    train_net(
        net=net,
        train_gen=train_gen,
        val_gen=val_gen,
        train_num_examples=1281167,
        val_num_examples=50048,
        num_epochs=args.num_epochs,
        checkpoint_filepath=os.path.join(args.save_dir, "imagenet_{}.h5".format(args.model)),
        start_epoch1=args.start_epoch)


if __name__ == "__main__":
    main()
