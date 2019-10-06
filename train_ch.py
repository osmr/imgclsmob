"""
    Script for training model on Chainer.
"""

import os
import argparse
import numpy as np
import chainer
from chainer import training
from chainer.training import extensions
from chainer.serializers import save_npz
from common.logger_utils import initialize_logging
from chainer_.utils import prepare_ch_context, prepare_model
from chainer_.dataset_utils import get_dataset_metainfo
from chainer_.dataset_utils import get_train_data_source, get_val_data_source


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
        default="chainer, chainercv",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="cupy-cuda100, chainer, chainercv",
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
        description="Train a model for image classification/segmentation (Chainer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K",
        help="dataset name. options are ImageNet1K, CUB200_2011, CIFAR10, CIFAR100, SVHN, VOC2012, ADE20K, Cityscapes, "
             "COCO")
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
    return seed


def prepare_trainer(net,
                    optimizer_name,
                    lr,
                    momentum,
                    num_epochs,
                    train_data,
                    val_data,
                    logging_dir_path,
                    use_gpus):
    if optimizer_name == "sgd":
        optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
    elif optimizer_name == "nag":
        optimizer = chainer.optimizers.NesterovAG(lr=lr, momentum=momentum)
    else:
        raise Exception("Unsupported optimizer: {}".format(optimizer_name))
    optimizer.setup(net)

    # devices = tuple(range(num_gpus)) if num_gpus > 0 else (-1, )
    devices = (0,) if use_gpus else (-1,)

    updater = training.updaters.StandardUpdater(
        iterator=train_data["iterator"],
        optimizer=optimizer,
        device=devices[0])
    trainer = training.Trainer(
        updater=updater,
        stop_trigger=(num_epochs, "epoch"),
        out=logging_dir_path)

    val_interval = 100000, "iteration"
    log_interval = 1000, "iteration"

    trainer.extend(
        extension=extensions.Evaluator(
            iterator=val_data["iterator"],
            target=net,
            device=devices[0]),
        trigger=val_interval)
    trainer.extend(extensions.dump_graph("main/loss"))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(
        extensions.snapshot_object(
            net,
            "model_iter_{.updater.iteration}"),
        trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.PrintReport([
            "epoch", "iteration", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy",
            "lr"]),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    return trainer


def save_params(file_stem,
                net,
                trainer):
    save_npz(
        file=file_stem + ".npz",
        obj=net)
    save_npz(
        file=file_stem + ".states",
        obj=trainer)


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

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)

    use_gpus = prepare_ch_context(args.num_gpus)
    # batch_size = args.batch_size

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_gpus=use_gpus,
        num_classes=args.num_classes,
        in_channels=args.in_channels)
    assert (hasattr(net, "classes"))
    assert (hasattr(net, "in_size"))

    train_data = get_train_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    val_data = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    trainer = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        momentum=args.momentum,
        num_epochs=args.num_epochs,
        train_data=train_data,
        val_data=val_data,
        logging_dir_path=args.save_dir,
        use_gpus=use_gpus)

    trainer.run()


if __name__ == "__main__":
    main()
