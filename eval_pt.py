"""
    Script for evaluating trained model on PyTorch (validate/test).
"""

import os
import time
import logging
import argparse
from sys import version_info
from common.logger_utils import initialize_logging
from pytorch.utils import prepare_pt_context, prepare_model
from pytorch.utils import calc_net_weight_count, validate
from pytorch.utils import get_composite_metric
from pytorch.utils import report_accuracy
from pytorch.dataset_utils import get_dataset_metainfo
from pytorch.dataset_utils import get_val_data_source, get_test_data_source
from pytorch.model_stats import measure_model
from pytorch.pytorchcv.models.model_store import _model_sha1


def add_eval_cls_parser_arguments(parser):
    """
    Create python script parameters (for eval specific subpart).

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
        help="resume from previously saved parameters")
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
        "--remove-module",
        action="store_true",
        help="enable if stored model has module")
    parser.add_argument(
        "--data-subset",
        type=str,
        default="val",
        help="data subset. options are val and test")

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
        default="torch, torchvision",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="",
        help="list of pip packages for logging")

    parser.add_argument(
        "--disable-cudnn-autotune",
        action="store_true",
        help="disable cudnn autotune for segmentation models")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="show progress bar")
    parser.add_argument(
        "--all",
        action="store_true",
        help="test all pretrained models for partucular dataset")


def parse_args():
    """
    Parse python script parameters (common part).

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification/segmentation (PyTorch)",
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

    add_eval_cls_parser_arguments(parser)

    args = parser.parse_args()
    return args


def prepare_dataset_metainfo(args):
    """
    Get dataset metainfo by name of dataset.

    Parameters
    ----------
    args : ArgumentParser
        Main script arguments.

    Returns
    -------
    DatasetMetaInfo
        Dataset metainfo.
    """
    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)
    assert (ds_metainfo.ml_type != "imgseg") or (args.batch_size == 1)
    assert (ds_metainfo.ml_type != "imgseg") or args.disable_cudnn_autotune
    return ds_metainfo


def prepare_data_source(ds_metainfo,
                        data_subset,
                        batch_size,
                        num_workers):
    """
    Prepare data loader.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    data_subset : str
        Data subset.
    batch_size : int
        Batch size.
    num_workers : int
        Number of background workers.

    Returns
    -------
    DataLoader
        Data source.
    """
    assert (data_subset in ("val", "test"))
    if data_subset == "val":
        get_data_source_class = get_val_data_source
    else:
        get_data_source_class = get_test_data_source
    data_source = get_data_source_class(
        ds_metainfo=ds_metainfo,
        batch_size=batch_size,
        num_workers=num_workers)
    return data_source


def prepare_metric(ds_metainfo,
                   data_subset):
    """
    Prepare metric.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        Dataset metainfo.
    data_subset : str
        Data subset.

    Returns
    -------
    CompositeEvalMetric
        Metric object instance.
    """
    assert (data_subset in ("val", "test"))
    if data_subset == "val":
        metric_names = ds_metainfo.val_metric_names
        metric_extra_kwargs = ds_metainfo.val_metric_extra_kwargs
    else:
        metric_names = ds_metainfo.test_metric_names
        metric_extra_kwargs = ds_metainfo.test_metric_extra_kwargs
    metric = get_composite_metric(
        metric_names=metric_names,
        metric_extra_kwargs=metric_extra_kwargs)
    return metric


def update_input_image_size(net,
                            input_size):
    """
    Update input image size for model.

    Parameters:
    ----------
    net : Module
        Model.
    input_size : int
        Preliminary value for input image size.

    Returns
    -------
    tuple of 2 ints
        Spatial size of the expected input image.
    """
    real_net = net.module if hasattr(net, "module") else net
    input_image_size = real_net.in_size if hasattr(real_net, "in_size") else\
        ((input_size, input_size) if type(input_size) == int else input_size)
    return input_image_size


def calc_model_accuracy(net,
                        test_data,
                        metric,
                        use_cuda,
                        input_image_size,
                        in_channels,
                        calc_weight_count=False,
                        calc_flops=False,
                        calc_flops_only=True,
                        extended_log=False):
    """
    Estimating particular model accuracy.

    Parameters:
    ----------
    net : Module
        Model.
    test_data : DataLoader
        Data loader.
    metric : EvalMetric
        Metric object instance.
    use_cuda : bool
        Whether to use CUDA.
    input_image_size : tuple of 2 ints
        Spatial size of the expected input image.
    in_channels : int
        Number of input channels.
    calc_weight_count : bool, default False
        Whether to calculate count of weights.
    calc_flops : bool, default False
        Whether to calculate FLOPs.
    calc_flops_only : bool, default True
        Whether to only calculate FLOPs without testing.
    extended_log : bool, default False
        Whether to log more precise accuracy values.

    Returns
    -------
    list of floats
        Accuracy values.
    """
    if not calc_flops_only:
        tic = time.time()
        validate(
            metric=metric,
            net=net,
            val_data=test_data,
            use_cuda=use_cuda)
        accuracy_msg = report_accuracy(
            metric=metric,
            extended_log=extended_log)
        logging.info("Test: {}".format(accuracy_msg))
        logging.info("Time cost: {:.4f} sec".format(
            time.time() - tic))
        acc_values = metric.get()[1]
        acc_values = acc_values if type(acc_values) == list else [acc_values]
    else:
        acc_values = []

    if calc_weight_count:
        weight_count = calc_net_weight_count(net)
        if not calc_flops:
            logging.info("Model: {} trainable parameters".format(weight_count))
    if calc_flops:
        num_flops, num_macs, num_params = measure_model(net, in_channels, input_image_size)
        assert (not calc_weight_count) or (weight_count == num_params)
        stat_msg = "Params: {params} ({params_m:.2f}M), FLOPs: {flops} ({flops_m:.2f}M)," \
                   " FLOPs/2: {flops2} ({flops2_m:.2f}M), MACs: {macs} ({macs_m:.2f}M)"
        logging.info(stat_msg.format(
            params=num_params, params_m=num_params / 1e6,
            flops=num_flops, flops_m=num_flops / 1e6,
            flops2=num_flops / 2, flops2_m=num_flops / 2 / 1e6,
            macs=num_macs, macs_m=num_macs / 1e6))

    return acc_values


def test_model(args):
    """
    Main test routine.

    Parameters:
    ----------
    args : ArgumentParser
        Main script arguments.

    Returns
    -------
    float
        Main accuracy value.
    """
    ds_metainfo = prepare_dataset_metainfo(args=args)
    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)
    data_source = prepare_data_source(
        ds_metainfo=ds_metainfo,
        data_subset=args.data_subset,
        batch_size=batch_size,
        num_workers=args.num_workers)
    metric = prepare_metric(
        ds_metainfo=ds_metainfo,
        data_subset=args.data_subset)
    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda,
        num_classes=(args.num_classes if ds_metainfo.ml_type != "hpe" else None),
        in_channels=args.in_channels,
        net_extra_kwargs=ds_metainfo.test_net_extra_kwargs,
        load_ignore_extra=ds_metainfo.load_ignore_extra,
        remove_module=args.remove_module)
    input_image_size = update_input_image_size(
        net=net,
        input_size=(args.input_size if hasattr(args, "input_size") else None))
    if args.show_progress:
        from tqdm import tqdm
        data_source = tqdm(data_source)
    assert (args.use_pretrained or args.resume.strip() or args.calc_flops_only)
    acc_values = calc_model_accuracy(
        net=net,
        test_data=data_source,
        metric=metric,
        use_cuda=use_cuda,
        input_image_size=input_image_size,
        in_channels=args.in_channels,
        calc_weight_count=True,
        calc_flops=args.calc_flops,
        calc_flops_only=args.calc_flops_only,
        extended_log=True)
    return acc_values[ds_metainfo.saver_acc_ind] if len(acc_values) > 0 else None


def main():
    """
    Main body of script.
    """
    args = parse_args()

    if args.disable_cudnn_autotune:
        os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    if args.all:
        args.use_pretrained = True
        dataset_name_map = {
            "in1k": "ImageNet1K",
            "cub": "CUB200_2011",
            "cf10": "CIFAR10",
            "cf100": "CIFAR100",
            "svhn": "SVHN",
            "voc": "VOC",
            "ade20k": "ADE20K",
            "cs": "Cityscapes",
            "cocoseg": "CocoSeg",
            "cocohpe": "CocoHpe",
            "hp": "HPatches",
        }
        for model_name, model_metainfo in (_model_sha1.items() if version_info[0] >= 3 else _model_sha1.iteritems()):
            error, checksum, repo_release_tag, caption, paper, ds, img_size, scale, batch, rem = model_metainfo
            if (ds != "in1k") or (img_size == 0) or ((len(rem) > 0) and (rem[-1] == "*")):
                continue
            args.dataset = dataset_name_map[ds]
            args.model = model_name
            args.input_size = img_size
            args.resize_inv_factor = scale
            args.batch_size = batch
            logging.info("==============")
            logging.info("Checking model: {}".format(model_name))
            acc_value = test_model(args=args)
            if acc_value is not None:
                exp_value = int(error) * 1e-4
                if abs(acc_value - exp_value) > 2e-4:
                    logging.info("----> Wrong value detected (expected value: {})!".format(exp_value))
    else:
        test_model(args=args)


if __name__ == "__main__":
    main()
