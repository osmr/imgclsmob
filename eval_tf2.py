"""
    Script for evaluating trained model on TensorFlow 2.0 (validate/test).
"""

import os
import time
import logging
import argparse
from sys import version_info
import tensorflow as tf
from common.logger_utils import initialize_logging
from tensorflow2.utils import prepare_model
from tensorflow2.tf2cv.models.model_store import _model_sha1
from tensorflow2.dataset_utils import get_dataset_metainfo, get_val_data_source, get_test_data_source
from tensorflow2.utils import get_composite_metric
from tensorflow2.utils import report_accuracy


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
        default="tensorflow-gpu",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="tensorflow-gpu",
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
        description="Evaluate a model for image classification/segmentation (TensorFlow 2.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K",
        help="dataset name. options are ImageNet1K, ImageNet1K_rec, CUB200_2011, CIFAR10, CIFAR100, SVHN, VOC2012, "
             "ADE20K, Cityscapes, COCO")
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


def test_model(args,
               use_cuda,
               data_format):
    """
    Main test routine.

    Parameters:
    ----------
    args : ArgumentParser
        Main script arguments.
    use_cuda : bool
        Whether to use CUDA.
    data_format : str
        The ordering of the dimensions in tensors.

    Returns
    -------
    float
        Main accuracy value.
    """
    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)
    assert (ds_metainfo.ml_type != "imgseg") or (args.batch_size == 1)
    assert (ds_metainfo.ml_type != "imgseg") or args.disable_cudnn_autotune

    batch_size = args.batch_size
    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        net_extra_kwargs=ds_metainfo.test_net_extra_kwargs,
        load_ignore_extra=ds_metainfo.load_ignore_extra,
        batch_size=batch_size,
        use_cuda=use_cuda)
    assert (hasattr(net, "in_size"))

    if not args.calc_flops_only:
        tic = time.time()

        get_test_data_source_class = get_val_data_source if args.data_subset == "val" else get_test_data_source
        test_data, total_img_count = get_test_data_source_class(
            ds_metainfo=ds_metainfo,
            batch_size=args.batch_size,
            data_format=data_format)
        if args.data_subset == "val":
            test_metric = get_composite_metric(
                metric_names=ds_metainfo.val_metric_names,
                metric_extra_kwargs=ds_metainfo.val_metric_extra_kwargs)
        else:
            test_metric = get_composite_metric(
                metric_names=ds_metainfo.test_metric_names,
                metric_extra_kwargs=ds_metainfo.test_metric_extra_kwargs)

        if args.show_progress:
            from tqdm import tqdm
            test_data = tqdm(test_data)

        processed_img_count = 0
        for test_images, test_labels in test_data:
            predictions = net(test_images)
            test_metric.update(test_labels, predictions)
            processed_img_count += len(test_images)
            if processed_img_count >= total_img_count:
                break

        accuracy_msg = report_accuracy(
            metric=test_metric,
            extended_log=True)
        logging.info("Test: {}".format(accuracy_msg))
        logging.info("Time cost: {:.4f} sec".format(
            time.time() - tic))
        acc_values = test_metric.get()[1]
        acc_values = acc_values if type(acc_values) == list else [acc_values]
    else:
        acc_values = []

    return acc_values


def main():
    """
    Main body of script.
    """
    args = parse_args()

    if args.disable_cudnn_autotune:
        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
        # os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        # os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    data_format = "channels_last"
    tf.keras.backend.set_image_data_format(data_format)

    use_cuda = (args.num_gpus > 0)

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
            error, checksum, repo_release_tag, ds, scale = model_metainfo
            args.dataset = dataset_name_map[ds]
            args.model = model_name
            args.resize_inv_factor = scale
            logging.info("==============")
            logging.info("Checking model: {}".format(model_name))
            acc_value = test_model(
                args=args,
                use_cuda=use_cuda,
                data_format=data_format)
            if acc_value is not None:
                exp_value = int(error) * 1e-4
                if abs(acc_value - exp_value) > 2e-4:
                    logging.info("----> Wrong value detected (expected value: {})!".format(exp_value))
            tf.keras.backend.clear_session()
    else:
        test_model(
            args=args,
            use_cuda=use_cuda,
            data_format=data_format)


if __name__ == "__main__":
    main()
