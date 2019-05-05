"""
    Classification routines (for evaluation).
"""

__all__ = ['get_dataset_metainfo', 'add_eval_cls_parser_arguments', 'get_batch_fn', 'test']

import time
import logging
from .utils import calc_net_weight_count, validate, report_accuracy
from .model_stats import measure_model
from gluon.datasets.imagenet1k_cls_dataset import ImageNet1KMetaInfo
from gluon.datasets.imagenet1k_rec_cls_dataset import ImageNet1KRecMetaInfo
from gluon.datasets.cub200_2011_cls_dataset import CUB200MetaInfo
from gluon.datasets.cifar10_cls_dataset import CIFAR10MetaInfo
from gluon.datasets.cifar100_cls_dataset import CIFAR100MetaInfo
from gluon.datasets.svhn_cls_dataset import SVHNMetaInfo
from mxnet import gluon


def get_dataset_metainfo(dataset_name):
    dataset_metainfo_map = {
        "ImageNet1K": ImageNet1KMetaInfo,
        "ImageNet1K_rec": ImageNet1KRecMetaInfo,
        "CUB_200_2011": CUB200MetaInfo,
        "CIFAR10": CIFAR10MetaInfo,
        "CIFAR100": CIFAR100MetaInfo,
        "SVHN": SVHNMetaInfo,
    }
    if dataset_name in dataset_metainfo_map.keys():
        return dataset_metainfo_map[dataset_name]()
    else:
        raise Exception("Unrecognized dataset: {}".format(dataset_name))


def add_eval_cls_parser_arguments(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model from gluon.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="data type for training. default is float32")
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
        "--calc-flops-only",
        dest="calc_flops_only",
        action="store_true",
        help="calculate FLOPs without quality estimation")

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
        default="mxnet",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="mxnet-cu100",
        help="list of pip packages for logging")


def get_batch_fn(use_imgrec):
    if use_imgrec:
        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            return data, label
        return batch_fn
    else:
        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label
        return batch_fn


def test(net,
         val_data,
         batch_fn,
         data_source_needs_reset,
         val_metric,
         dtype,
         ctx,
         input_image_size,
         in_channels,
         calc_weight_count=False,
         calc_flops=False,
         calc_flops_only=True,
         extended_log=False):
    if not calc_flops_only:
        tic = time.time()
        validate(
            metric=val_metric,
            net=net,
            val_data=val_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx)
        accuracy_msg = report_accuracy(
            metric=val_metric,
            extended_log=extended_log)
        logging.info("Test: {}".format(accuracy_msg))
        logging.info("Time cost: {:.4f} sec".format(
            time.time() - tic))

    if calc_weight_count:
        weight_count = calc_net_weight_count(net)
        if not calc_flops:
            logging.info("Model: {} trainable parameters".format(weight_count))
    if calc_flops:
        num_flops, num_macs, num_params = measure_model(net, in_channels, input_image_size, ctx[0])
        assert (not calc_weight_count) or (weight_count == num_params)
        stat_msg = "Params: {params} ({params_m:.2f}M), FLOPs: {flops} ({flops_m:.2f}M)," \
                   " FLOPs/2: {flops2} ({flops2_m:.2f}M), MACs: {macs} ({macs_m:.2f}M)"
        logging.info(stat_msg.format(
            params=num_params, params_m=num_params / 1e6,
            flops=num_flops, flops_m=num_flops / 1e6,
            flops2=num_flops / 2, flops2_m=num_flops / 2 / 1e6,
            macs=num_macs, macs_m=num_macs / 1e6))
