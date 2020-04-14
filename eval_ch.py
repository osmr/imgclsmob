"""
    Script for evaluating trained model on Chainer (validate/test).
"""

import os
import time
import logging
import argparse
from sys import version_info
from chainer import global_config
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook
from common.logger_utils import initialize_logging
from chainer_.utils import prepare_ch_context, prepare_model, Predictor
from chainer_.utils import get_composite_metric, report_accuracy
from chainer_.dataset_utils import get_dataset_metainfo
from chainer_.dataset_utils import get_val_data_source, get_test_data_source
from chainer_.chainercv2.models.model_store import _model_sha1


def add_eval_parser_arguments(parser):
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
        "--calc-flops-only",
        dest="calc_flops_only",
        action="store_true",
        help="calculate FLOPs without quality estimation")
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
        default="chainer, chainercv",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="cupy-cuda101, chainer, chainercv",
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
    Create python script parameters (common part).

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification/segmentation (Chainer)",
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

    add_eval_parser_arguments(parser)

    args = parser.parse_args()
    return args


def calc_model_accuracy(net,
                        test_data,
                        metric,
                        calc_weight_count=False,
                        calc_flops_only=True,
                        extended_log=False):
    """
    Main test routine.

    Parameters:
    ----------
    net : Chain
        Model.
    test_data : dict
        Data loader.
    metric : EvalMetric
        Metric object instance.
    calc_weight_count : bool, default False
        Whether to calculate count of weights.
    extended_log : bool, default False
        Whether to log more precise accuracy values.
    ml_type : str, default 'imgcls'
        Machine learning type.

    Returns
    -------
    list of floats
        Accuracy values.
    """
    tic = time.time()

    predictor = Predictor(
        model=net,
        transform=None)

    if calc_weight_count:
        weight_count = net.count_params()
        logging.info("Model: {} trainable parameters".format(weight_count))

    if not calc_flops_only:
        in_values, out_values, rest_values = apply_to_iterator(
            func=predictor,
            iterator=test_data["iterator"],
            hook=ProgressHook(test_data["ds_len"]))
        assert (len(rest_values) == 1)
        assert (len(out_values) == 1)
        assert (len(in_values) == 1)

        if True:
            labels = iter(rest_values[0])
            preds = iter(out_values[0])
            inputs = iter(in_values[0])
            for label, pred, inputi in zip(labels, preds, inputs):
                metric.update(label, pred)
                del label
                del pred
                del inputi
        else:
            import numpy as np
            metric.update(
                labels=np.array(list(rest_values[0])),
                preds=np.array(list(out_values[0])))

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
    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)
    assert (ds_metainfo.ml_type != "imgseg") or (args.batch_size == 1)
    assert (ds_metainfo.ml_type != "imgseg") or args.disable_cudnn_autotune

    global_config.train = False
    use_gpus = prepare_ch_context(args.num_gpus)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_gpus=use_gpus,
        net_extra_kwargs=ds_metainfo.test_net_extra_kwargs,
        num_classes=(args.num_classes if ds_metainfo.ml_type != "hpe" else None),
        in_channels=args.in_channels)
    assert (hasattr(net, "classes") or (ds_metainfo.ml_type == "hpe"))
    assert (hasattr(net, "in_size"))

    get_test_data_source_class = get_val_data_source if args.data_subset == "val" else get_test_data_source
    test_data = get_test_data_source_class(
        ds_metainfo=ds_metainfo,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    if args.data_subset == "val":
        test_metric = get_composite_metric(
            metric_names=ds_metainfo.val_metric_names,
            metric_extra_kwargs=ds_metainfo.val_metric_extra_kwargs)
    else:
        test_metric = get_composite_metric(
            metric_names=ds_metainfo.test_metric_names,
            metric_extra_kwargs=ds_metainfo.test_metric_extra_kwargs)

    assert (args.use_pretrained or args.resume.strip())
    acc_values = calc_model_accuracy(
        net=net,
        test_data=test_data,
        metric=test_metric,
        calc_weight_count=True,
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
        for model_name, model_metainfo in (_model_sha1.items() if version_info[0] >= 3 else _model_sha1.iteritems()):
            error, checksum, repo_release_tag = model_metainfo
            args.model = model_name
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
