"""
    Script for training model on PyTorch.
"""

import os
import time
import logging
import argparse
import random
import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from common.logger_utils import initialize_logging
from common.train_log_param_saver import TrainLogParamSaver
from pytorch.utils import prepare_pt_context, prepare_model, validate
from pytorch.utils import report_accuracy, get_composite_metric, get_metric_name

from pytorch.dataset_utils import get_dataset_metainfo
from pytorch.dataset_utils import get_train_data_source, get_val_data_source


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
        "--batch-size-scale",
        type=int,
        default=1,
        help="manual batch-size increasing factor")
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
        "--poly-power",
        type=float,
        default=2,
        help="power value for poly LR scheduler")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="number of warmup epochs")
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-8,
        help="starting warmup learning rate")
    parser.add_argument(
        "--warmup-mode",
        type=str,
        default="linear",
        help="learning rate scheduler warmup mode. options are linear, poly and constant")
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
        "--gamma-wd-mult",
        type=float,
        default=1.0,
        help="weight decay multiplier for batchnorm gamma")
    parser.add_argument(
        "--beta-wd-mult",
        type=float,
        default=1.0,
        help="weight decay multiplier for batchnorm beta")
    parser.add_argument(
        "--bias-wd-mult",
        type=float,
        default=1.0,
        help="weight decay multiplier for bias")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="max_norm for gradient clipping")
    parser.add_argument(
        "--label-smoothing",
        action="store_true",
        help="use label smoothing")

    parser.add_argument(
        "--mixup",
        action="store_true",
        help="use mixup strategy")
    parser.add_argument(
        "--mixup-epoch-tail",
        type=int,
        default=15,
        help="number of epochs without mixup at the end of training")

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
        default="torch, torchvision",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="",
        help="list of pip packages for logging")

    parser.add_argument(
        "--tune-layers",
        type=str,
        default="",
        help="regexp for selecting layers for fine tuning")


def parse_args():
    """
    Parse python script parameters (common part).

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (PyTorch)",
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
    """
    Initialize all random generators by seed.

    Parameters:
    ----------
    seed : int
        Seed value.

    Returns
    -------
    int
        Generated seed value.
    """
    if seed <= 0:
        seed = np.random.randint(10000)
    else:
        cudnn.deterministic = True
        logging.warning(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down "
            "your training considerably! You may see unexpected behavior when restarting from checkpoints.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def prepare_trainer(net,
                    optimizer_name,
                    wd,
                    momentum,
                    lr_mode,
                    lr,
                    lr_decay_period,
                    lr_decay_epoch,
                    lr_decay,
                    num_epochs,
                    state_file_path):
    """
    Prepare trainer.

    Parameters:
    ----------
    net : Module
        Model.
    optimizer_name : str
        Name of optimizer.
    wd : float
        Weight decay rate.
    momentum : float
        Momentum value.
    lr_mode : str
        Learning rate scheduler mode.
    lr : float
        Learning rate.
    lr_decay_period : int
        Interval for periodic learning rate decays.
    lr_decay_epoch : str
        Epoches at which learning rate decays.
    lr_decay : float
        Decay rate of learning rate.
    num_epochs : int
        Number of training epochs.
    state_file_path : str
        Path for file with trainer state.

    Returns
    -------
    Optimizer
        Optimizer.
    LRScheduler
        Learning rate scheduler.
    int
        Start epoch.
    """
    optimizer_name = optimizer_name.lower()
    if (optimizer_name == "sgd") or (optimizer_name == "nag"):
        optimizer = torch.optim.SGD(
            params=net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
            nesterov=(optimizer_name == "nag"))
    else:
        raise ValueError("Usupported optimizer: {}".format(optimizer_name))

    if state_file_path:
        checkpoint = torch.load(state_file_path)
        if type(checkpoint) == dict:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = None
    else:
        start_epoch = None

    cudnn.benchmark = True

    lr_mode = lr_mode.lower()
    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(",")]
    if (lr_mode == "step") and (lr_decay_period != 0):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=lr_decay_period,
            gamma=lr_decay,
            last_epoch=-1)
    elif (lr_mode == "multistep") or ((lr_mode == "step") and (lr_decay_period == 0)):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_epoch,
            gamma=lr_decay,
            last_epoch=-1)
    elif lr_mode == "cosine":
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            last_epoch=(num_epochs - 1))
    else:
        raise ValueError("Usupported lr_scheduler: {}".format(lr_mode))

    return optimizer, lr_scheduler, start_epoch


def save_params(file_stem,
                state):
    """
    Save current model/trainer parameters.

    Parameters:
    ----------
    file_stem : str
        File stem (with path).
    state : dict
        Whole state of model & trainer.
    trainer : Trainer
        Trainer.
    """
    torch.save(
        obj=state["state_dict"],
        f=(file_stem + ".pth"))
    torch.save(
        obj=state,
        f=(file_stem + ".states"))


def train_epoch(epoch,
                net,
                train_metric,
                train_data,
                use_cuda,
                L,
                optimizer,
                # lr_scheduler,
                batch_size,
                log_interval):
    """
    Train model on particular epoch.

    Parameters:
    ----------
    epoch : int
        Epoch number.
    net : Module
        Model.
    train_metric : EvalMetric
        Metric object instance.
    train_data : DataLoader
        Data loader.
    use_cuda : bool
        Whether to use CUDA.
    L : Loss
        Loss function.
    optimizer : Optimizer
        Optimizer.
    batch_size : int
        Training batch size.
    log_interval : int
        Batch count period for logging.

    Returns
    -------
    float
        Loss value.
    """
    tic = time.time()
    net.train()
    train_metric.reset()
    train_loss = 0.0

    btic = time.time()
    for i, (data, target) in enumerate(train_data):
        if use_cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = net(data)
        loss = L(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_metric.update(
            labels=target,
            preds=output)

        if log_interval and not (i + 1) % log_interval:
            speed = batch_size * log_interval / (time.time() - btic)
            btic = time.time()
            train_accuracy_msg = report_accuracy(metric=train_metric)
            logging.info("Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}\tlr={:.5f}".format(
                epoch + 1, i, speed, train_accuracy_msg, optimizer.param_groups[0]["lr"]))

    throughput = int(batch_size * (i + 1) / (time.time() - tic))
    logging.info("[Epoch {}] speed: {:.2f} samples/sec\ttime cost: {:.2f} sec".format(
        epoch + 1, throughput, time.time() - tic))

    train_loss /= (i + 1)
    train_accuracy_msg = report_accuracy(metric=train_metric)
    logging.info("[Epoch {}] training: {}\tloss={:.4f}".format(
        epoch + 1, train_accuracy_msg, train_loss))

    return train_loss


def train_net(batch_size,
              num_epochs,
              start_epoch1,
              train_data,
              val_data,
              net,
              optimizer,
              lr_scheduler,
              lp_saver,
              log_interval,
              num_classes,
              val_metric,
              train_metric,
              use_cuda):
    """
    Main procedure for training model.

    Parameters:
    ----------
    batch_size : int
        Training batch size.
    num_epochs : int
        Number of training epochs.
    start_epoch1 : int
        Number of starting epoch (1-based).
    train_data : DataLoader
        Data loader (training subset).
    val_data : DataLoader
        Data loader (validation subset).
    net : Module
        Model.
    optimizer : Optimizer
        Optimizer.
    lr_scheduler : LRScheduler
        Learning rate scheduler.
    lp_saver : TrainLogParamSaver
        Model/trainer state saver.
    log_interval : int
        Batch count period for logging.
    num_classes : int
        Number of model classes.
    val_metric : EvalMetric
        Metric object instance (validation subset).
    train_metric : EvalMetric
        Metric object instance (training subset).
    use_cuda : bool
        Whether to use CUDA.
    """
    assert (num_classes > 0)

    L = nn.CrossEntropyLoss()
    if use_cuda:
        L = L.cuda()

    assert (type(start_epoch1) == int)
    assert (start_epoch1 >= 1)
    if start_epoch1 > 1:
        logging.info("Start training from [Epoch {}]".format(start_epoch1))
        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            use_cuda=use_cuda)
        val_accuracy_msg = report_accuracy(metric=val_metric)
        logging.info("[Epoch {}] validation: {}".format(start_epoch1 - 1, val_accuracy_msg))

    gtic = time.time()
    for epoch in range(start_epoch1 - 1, num_epochs):
        lr_scheduler.step()

        train_loss = train_epoch(
            epoch=epoch,
            net=net,
            train_metric=train_metric,
            train_data=train_data,
            use_cuda=use_cuda,
            L=L,
            optimizer=optimizer,
            # lr_scheduler,
            batch_size=batch_size,
            log_interval=log_interval)

        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            use_cuda=use_cuda)
        val_accuracy_msg = report_accuracy(metric=val_metric)
        logging.info("[Epoch {}] validation: {}".format(epoch + 1, val_accuracy_msg))

        if lp_saver is not None:
            state = {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            lp_saver_kwargs = {"state": state}
            val_acc_values = val_metric.get()[1]
            train_acc_values = train_metric.get()[1]
            val_acc_values = val_acc_values if type(val_acc_values) == list else [val_acc_values]
            train_acc_values = train_acc_values if type(train_acc_values) == list else [train_acc_values]
            lp_saver.epoch_test_end_callback(
                epoch1=(epoch + 1),
                params=(val_acc_values + train_acc_values + [train_loss, optimizer.param_groups[0]["lr"]]),
                **lp_saver_kwargs)

    logging.info("Total time cost: {:.2f} sec".format(time.time() - gtic))
    if lp_saver is not None:
        opt_metric_name = get_metric_name(val_metric, lp_saver.acc_ind)
        logging.info("Best {}: {:.4f} at {} epoch".format(
            opt_metric_name, lp_saver.best_eval_metric_value, lp_saver.best_eval_metric_epoch))


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

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda)
    real_net = net.module if hasattr(net, "module") else net
    assert (hasattr(real_net, "num_classes"))
    num_classes = real_net.num_classes

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)

    train_data = get_train_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)
    val_data = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=args.num_workers)

    optimizer, lr_scheduler, start_epoch = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        wd=args.wd,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        lr_decay_period=args.lr_decay_period,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay=args.lr_decay,
        num_epochs=args.num_epochs,
        state_file_path=args.resume_state)

    if args.save_dir and args.save_interval:
        param_names = ds_metainfo.val_metric_capts + ds_metainfo.train_metric_capts + ["Train.Loss", "LR"]
        lp_saver = TrainLogParamSaver(
            checkpoint_file_name_prefix="{}_{}".format(ds_metainfo.short_label, args.model),
            last_checkpoint_file_name_suffix="last",
            best_checkpoint_file_name_suffix=None,
            last_checkpoint_dir_path=args.save_dir,
            best_checkpoint_dir_path=None,
            last_checkpoint_file_count=2,
            best_checkpoint_file_count=2,
            checkpoint_file_save_callback=save_params,
            checkpoint_file_exts=(".pth", ".states"),
            save_interval=args.save_interval,
            num_epochs=args.num_epochs,
            param_names=param_names,
            acc_ind=ds_metainfo.saver_acc_ind,
            # bigger=[True],
            # mask=None,
            score_log_file_path=os.path.join(args.save_dir, "score.log"),
            score_log_attempt_value=args.attempt,
            best_map_log_file_path=os.path.join(args.save_dir, "best_map.log"))
    else:
        lp_saver = None

    train_net(
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        start_epoch1=args.start_epoch,
        train_data=train_data,
        val_data=val_data,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lp_saver=lp_saver,
        log_interval=args.log_interval,
        num_classes=num_classes,
        val_metric=get_composite_metric(ds_metainfo.val_metric_names, ds_metainfo.val_metric_extra_kwargs),
        train_metric=get_composite_metric(ds_metainfo.train_metric_names, ds_metainfo.train_metric_extra_kwargs),
        use_cuda=use_cuda)


if __name__ == "__main__":
    main()
