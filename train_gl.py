import argparse
import time
import logging
import os
import numpy as np
import random

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag

from common.logger_utils import initialize_logging
from common.train_log_param_saver import TrainLogParamSaver
from gluon.lr_scheduler import LRScheduler
from gluon.utils import prepare_mx_context, prepare_model, validate

from gluon.imagenet1k_utils import add_dataset_parser_arguments
from gluon.imagenet1k_utils import get_batch_fn
from gluon.imagenet1k_utils import get_train_data_source
from gluon.imagenet1k_utils import get_val_data_source
from gluon.imagenet1k_utils import get_dataset_metainfo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model for image classification (Gluon)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K_rec",
        help="dataset name. options are ImageNet1K and ImageNet1K_rec")

    args, _ = parser.parse_known_args()
    add_dataset_parser_arguments(parser, args.dataset)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model from gluon")
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
        default=20,
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
        help="random seed to be fixed")
    parser.add_argument(
        "--log-packages",
        type=str,
        default="mxnet",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="mxnet-cu100",
        help="list of pip packages for logging")

    parser.add_argument(
        "--tune-layers",
        type=str,
        default="",
        help="regexp for selecting layers for fine tuning")
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
                    wd,
                    momentum,
                    lr_mode,
                    lr,
                    lr_decay_period,
                    lr_decay_epoch,
                    lr_decay,
                    target_lr,
                    poly_power,
                    warmup_epochs,
                    warmup_lr,
                    warmup_mode,
                    batch_size,
                    num_epochs,
                    num_training_samples,
                    dtype,
                    gamma_wd_mult=1.0,
                    beta_wd_mult=1.0,
                    bias_wd_mult=1.0,
                    state_file_path=None):

    if gamma_wd_mult != 1.0:
        for k, v in net.collect_params(".*gamma").items():
            v.wd_mult = gamma_wd_mult

    if beta_wd_mult != 1.0:
        for k, v in net.collect_params(".*beta").items():
            v.wd_mult = beta_wd_mult

    if bias_wd_mult != 1.0:
        for k, v in net.collect_params(".*bias").items():
            v.wd_mult = bias_wd_mult

    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(",")]
    num_batches = num_training_samples // batch_size
    lr_scheduler = LRScheduler(
        mode=lr_mode,
        base_lr=lr,
        n_iters=num_batches,
        n_epochs=num_epochs,
        step=lr_decay_epoch,
        step_factor=lr_decay,
        target_lr=target_lr,
        power=poly_power,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        warmup_mode=warmup_mode)

    optimizer_params = {"learning_rate": lr,
                        "wd": wd,
                        "momentum": momentum,
                        "lr_scheduler": lr_scheduler}
    if dtype != "float32":
        optimizer_params["multi_precision"] = True

    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer=optimizer_name,
        optimizer_params=optimizer_params)

    if (state_file_path is not None) and state_file_path and os.path.exists(state_file_path):
        logging.info("Loading trainer states: {}".format(state_file_path))
        trainer.load_states(state_file_path)
        if trainer._optimizer.wd != wd:
            trainer._optimizer.wd = wd
            logging.info("Reset the weight decay: {}".format(wd))
        # lr_scheduler = trainer._optimizer.lr_scheduler
        trainer._optimizer.lr_scheduler = lr_scheduler

    return trainer, lr_scheduler


def save_params(file_stem,
                net,
                trainer):
    net.save_parameters(file_stem + ".params")
    trainer.save_states(file_stem + ".states")


def train_epoch(epoch,
                net,
                acc_top1_train,
                train_data,
                batch_fn,
                data_source_needs_reset,
                dtype,
                ctx,
                loss_func,
                trainer,
                lr_scheduler,
                batch_size,
                log_interval,
                mixup,
                mixup_epoch_tail,
                label_smoothing,
                num_classes,
                num_epochs,
                grad_clip_value,
                batch_size_scale):

    labels_list_inds = None
    batch_size_extend_count = 0
    tic = time.time()
    if data_source_needs_reset:
        train_data.reset()
    acc_top1_train.reset()
    train_loss = 0.0

    btic = time.time()
    for i, batch in enumerate(train_data):
        data_list, labels_list = batch_fn(batch, ctx)

        if mixup:
            labels_list_inds = labels_list
            labels_list = [Y.one_hot(depth=num_classes) for Y in labels_list]
            if epoch < num_epochs - mixup_epoch_tail:
                alpha = 1
                lam = np.random.beta(alpha, alpha)
                data_list = [lam * X + (1 - lam) * X[::-1] for X in data_list]
                labels_list = [lam * Y + (1 - lam) * Y[::-1] for Y in labels_list]
        elif label_smoothing:
            eta = 0.1
            on_value = 1 - eta + eta / num_classes
            off_value = eta / num_classes
            labels_list_inds = labels_list
            labels_list = [Y.one_hot(depth=num_classes, on_value=on_value, off_value=off_value) for Y in labels_list]

        with ag.record():
            outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
            loss_list = [loss_func(yhat, y.astype(dtype, copy=False)) for yhat, y in zip(outputs_list, labels_list)]
        for loss in loss_list:
            loss.backward()
        lr_scheduler.update(i, epoch)

        if grad_clip_value is not None:
            grads = [v.grad(ctx[0]) for v in net.collect_params().values() if v._grad is not None]
            gluon.utils.clip_global_norm(grads, max_norm=grad_clip_value)

        if batch_size_scale == 1:
            trainer.step(batch_size)
        else:
            if (i + 1) % batch_size_scale == 0:
                batch_size_extend_count = 0
                trainer.step(batch_size * batch_size_scale)
                for p in net.collect_params().values():
                    p.zero_grad()
            else:
                batch_size_extend_count += 1

        train_loss += sum([loss.mean().asscalar() for loss in loss_list]) / len(loss_list)

        acc_top1_train.update(
            labels=(labels_list if not (mixup or label_smoothing) else labels_list_inds),
            preds=outputs_list)

        if log_interval and not (i + 1) % log_interval:
            speed = batch_size * log_interval / (time.time() - btic)
            btic = time.time()
            _, top1 = acc_top1_train.get()
            err_top1_train = 1.0 - top1
            logging.info("Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\ttop1-err={:.4f}\tlr={:.5f}".format(
                epoch + 1, i, speed, err_top1_train, trainer.learning_rate))

    if (batch_size_scale != 1) and (batch_size_extend_count > 0):
        trainer.step(batch_size * batch_size_extend_count)
        for p in net.collect_params().values():
            p.zero_grad()

    throughput = int(batch_size * (i + 1) / (time.time() - tic))
    logging.info("[Epoch {}] speed: {:.2f} samples/sec\ttime cost: {:.2f} sec".format(
        epoch + 1, throughput, time.time() - tic))

    train_loss /= (i + 1)
    _, top1 = acc_top1_train.get()
    err_top1_train = 1.0 - top1
    logging.info("[Epoch {}] training: err-top1={:.4f}\tloss={:.4f}".format(
        epoch + 1, err_top1_train, train_loss))

    return err_top1_train, train_loss


def train_net(batch_size,
              num_epochs,
              start_epoch1,
              train_data,
              val_data,
              batch_fn,
              data_source_needs_reset,
              dtype,
              net,
              trainer,
              lr_scheduler,
              lp_saver,
              log_interval,
              mixup,
              mixup_epoch_tail,
              label_smoothing,
              num_classes,
              grad_clip_value,
              batch_size_scale,
              ctx):

    assert (not (mixup and label_smoothing))

    if batch_size_scale != 1:
        for p in net.collect_params().values():
            p.grad_req = "add"

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    acc_top1_val = mx.metric.Accuracy()
    acc_top5_val = mx.metric.TopKAccuracy(5)
    acc_top1_train = mx.metric.Accuracy()

    loss_func = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=(not (mixup or label_smoothing)))

    assert (type(start_epoch1) == int)
    assert (start_epoch1 >= 1)
    if start_epoch1 > 1:
        logging.info("Start training from [Epoch {}]".format(start_epoch1))
        err_top1_val, err_top5_val = validate(
            acc_top1=acc_top1_val,
            acc_top5=acc_top5_val,
            net=net,
            val_data=val_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx)
        logging.info("[Epoch {}] validation: err-top1={:.4f}\terr-top5={:.4f}".format(
            start_epoch1 - 1, err_top1_val, err_top5_val))

    gtic = time.time()
    for epoch in range(start_epoch1 - 1, num_epochs):
        err_top1_train, train_loss = train_epoch(
            epoch=epoch,
            net=net,
            acc_top1_train=acc_top1_train,
            train_data=train_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx,
            loss_func=loss_func,
            trainer=trainer,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            log_interval=log_interval,
            mixup=mixup,
            mixup_epoch_tail=mixup_epoch_tail,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
            num_epochs=num_epochs,
            grad_clip_value=grad_clip_value,
            batch_size_scale=batch_size_scale)

        err_top1_val, err_top5_val = validate(
            acc_top1=acc_top1_val,
            acc_top5=acc_top5_val,
            net=net,
            val_data=val_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx)

        logging.info("[Epoch {}] validation: err-top1={:.4f}\terr-top5={:.4f}".format(
            epoch + 1, err_top1_val, err_top5_val))

        if lp_saver is not None:
            lp_saver_kwargs = {"net": net, "trainer": trainer}
            lp_saver.epoch_test_end_callback(
                epoch1=(epoch + 1),
                params=[err_top1_val, err_top1_train, err_top5_val, train_loss, trainer.learning_rate],
                **lp_saver_kwargs)

    logging.info("Total time cost: {:.2f} sec".format(time.time() - gtic))
    if lp_saver is not None:
        logging.info("Best err-top5: {:.4f} at {} epoch".format(
            lp_saver.best_eval_metric_value, lp_saver.best_eval_metric_epoch))


def main():
    args = parse_args()
    args.seed = init_rand(seed=args.seed)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    ctx, batch_size = prepare_mx_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        dtype=args.dtype,
        tune_layers=args.tune_layers,
        classes=args.num_classes,
        in_channels=args.in_channels,
        ctx=ctx)

    assert (hasattr(net, "classes"))
    assert (hasattr(net, "in_size"))
    num_classes = net.classes
    input_image_size = net.in_size

    dataset_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    train_data = get_train_data_source(
        dataset_metainfo=dataset_metainfo,
        dataset_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_image_size=input_image_size)
    val_data = get_val_data_source(
        dataset_metainfo=dataset_metainfo,
        dataset_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_image_size=input_image_size,
        resize_inv_factor=args.resize_inv_factor)
    batch_fn = get_batch_fn(use_imgrec=dataset_metainfo.use_imgrec)

    num_training_samples = len(train_data._dataset) if not dataset_metainfo.use_imgrec else\
        dataset_metainfo.num_training_samples
    trainer, lr_scheduler = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        wd=args.wd,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        lr_decay_period=args.lr_decay_period,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay=args.lr_decay,
        target_lr=args.target_lr,
        poly_power=args.poly_power,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        warmup_mode=args.warmup_mode,
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        num_training_samples=num_training_samples,
        dtype=args.dtype,
        gamma_wd_mult=args.gamma_wd_mult,
        beta_wd_mult=args.beta_wd_mult,
        bias_wd_mult=args.bias_wd_mult,
        state_file_path=args.resume_state)

    if args.save_dir and args.save_interval:
        lp_saver = TrainLogParamSaver(
            checkpoint_file_name_prefix="{}_{}".format(dataset_metainfo.short_label, args.model),
            last_checkpoint_file_name_suffix="last",
            best_checkpoint_file_name_suffix=None,
            last_checkpoint_dir_path=args.save_dir,
            best_checkpoint_dir_path=None,
            last_checkpoint_file_count=2,
            best_checkpoint_file_count=2,
            checkpoint_file_save_callback=save_params,
            checkpoint_file_exts=(".params", ".states"),
            save_interval=args.save_interval,
            num_epochs=args.num_epochs,
            param_names=["Val.Top1", "Train.Top1", "Val.Top5", "Train.Loss", "LR"],
            acc_ind=2,
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
        batch_fn=batch_fn,
        data_source_needs_reset=dataset_metainfo.use_imgrec,
        dtype=args.dtype,
        net=net,
        trainer=trainer,
        lr_scheduler=lr_scheduler,
        lp_saver=lp_saver,
        log_interval=args.log_interval,
        mixup=args.mixup,
        mixup_epoch_tail=args.mixup_epoch_tail,
        label_smoothing=args.label_smoothing,
        num_classes=num_classes,
        grad_clip_value=args.grad_clip,
        batch_size_scale=args.batch_size_scale,
        ctx=ctx)


if __name__ == "__main__":
    main()
