"""
    Script for training model on TensorFlow.
"""

import argparse
import numpy as np
import random
from tensorpack.input_source import QueueInput
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
from tensorpack import ModelSaver, ScheduledHyperParamSetter, EstimatedTimeLeft, ClassificationError, InferenceRunner,\
    DataParallelInferenceRunner, TrainConfig, SyncMultiGPUTrainerParameterServer, launch_train_with_config
from common.logger_utils import initialize_logging
from tensorflow_.utils_tp import prepare_tf_context, prepare_model, get_data


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (TensorFlow/TensorPack)",
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
        default="tensorflow-gpu, tensorpack",
        help="list of pip packages for logging")
    args = parser.parse_args()
    return args


def init_rand(seed):
    if seed <= 0:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def train_net(net,
              session_init,
              batch_size,
              num_epochs,
              train_dataflow,
              val_dataflow):

    num_towers = max(get_num_gpu(), 1)
    batch_per_tower = batch_size // num_towers
    logger.info("Running on {} towers. Batch size per tower: {}".format(num_towers, batch_per_tower))

    num_training_samples = 1281167
    step_size = num_training_samples // batch_size
    max_iter = (num_epochs - 1) * step_size
    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter(
            "learning_rate",
            [(0, 0.5), (max_iter, 0)],
            interp="linear",
            step_based=True),
        EstimatedTimeLeft()]

    infs = [ClassificationError("wrong-top1", "val-error-top1"),
            ClassificationError("wrong-top5", "val-error-top5")]
    if num_towers == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(
            input=QueueInput(val_dataflow),
            infs=infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            input=val_dataflow,
            infs=infs,
            gpus=list(range(num_towers))))

    config = TrainConfig(
        dataflow=train_dataflow,
        model=net,
        callbacks=callbacks,
        session_init=session_init,
        steps_per_epoch=step_size,
        max_epoch=num_epochs)

    launch_train_with_config(
        config=config,
        trainer=SyncMultiGPUTrainerParameterServer(num_towers))


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
    logger.set_logger_dir(args.save_dir)

    batch_size = prepare_tf_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net, inputs_desc = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        data_format=args.data_format)

    train_dataflow = get_data(
        is_train=True,
        batch_size=batch_size,
        data_dir_path=args.data_dir,
        input_image_size=net.image_size,
        resize_inv_factor=args.resize_inv_factor)
    val_dataflow = get_data(
        is_train=False,
        batch_size=batch_size,
        data_dir_path=args.data_dir,
        input_image_size=net.image_size,
        resize_inv_factor=args.resize_inv_factor)

    train_net(
        net=net,
        session_init=inputs_desc,
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        train_dataflow=train_dataflow,
        val_dataflow=val_dataflow)


if __name__ == "__main__":
    main()
