import os
import time
import logging
import argparse
from common.logger_utils import initialize_logging
from gluon.utils import prepare_mx_context, prepare_model
from gluon.utils import calc_net_weight_count, validate
from gluon.utils import get_composite_metric
from gluon.utils import report_accuracy
from gluon.dataset_utils import get_dataset_metainfo
from gluon.dataset_utils import get_batch_fn
from gluon.dataset_utils import get_val_data_source, get_test_data_source
from gluon.model_stats import measure_model


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
        "--dtype",
        type=str,
        default="float32",
        help="base data type for tensors")
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
        default="mxnet, numpy",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="mxnet-cu100",
        help="list of pip packages for logging")

    parser.add_argument(
        "--disable-cudnn-autotune",
        action="store_true",
        help="disable cudnn autotune for segmentation models")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="show progress bar")


def parse_args():
    """
    Create python script parameters (common part).

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification/segmentation (Gluon)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K_rec",
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


def test(net,
         test_data,
         batch_fn,
         data_source_needs_reset,
         metric,
         dtype,
         ctx,
         input_image_size,
         in_channels,
         calc_weight_count=False,
         calc_flops=False,
         calc_flops_only=True,
         extended_log=False):
    """
    Main test routine.

    Parameters:
    ----------
    net : HybridBlock
        Model.
    test_data : DataLoader or ImageRecordIter
        Data loader or ImRec-iterator.
    batch_fn : func
        Function for splitting data after extraction from data loader.
    data_source_needs_reset : bool
        Whether to reset data (if test_data is ImageRecordIter).
    metric : EvalMetric
        Metric object instance.
    dtype : str
        Base data type for tensors.
    ctx : Context
        MXNet context.
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
    """
    if not calc_flops_only:
        tic = time.time()
        validate(
            metric=metric,
            net=net,
            val_data=test_data,
            batch_fn=batch_fn,
            data_source_needs_reset=data_source_needs_reset,
            dtype=dtype,
            ctx=ctx)
        accuracy_msg = report_accuracy(
            metric=metric,
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

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)
    assert (ds_metainfo.ml_type != "imgseg") or (args.batch_size == 1)
    assert (ds_metainfo.ml_type != "imgseg") or args.disable_cudnn_autotune

    ctx, batch_size = prepare_mx_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        dtype=args.dtype,
        net_extra_kwargs=ds_metainfo.net_extra_kwargs,
        load_ignore_extra=ds_metainfo.load_ignore_extra,
        classes=args.num_classes,
        in_channels=args.in_channels,
        do_hybridize=(ds_metainfo.allow_hybridize and (not args.calc_flops)),
        ctx=ctx)
    assert (hasattr(net, "in_size"))
    input_image_size = net.in_size

    if args.data_subset == "val":
        get_test_data_source_class = get_val_data_source
        test_metric = get_composite_metric(
            metric_names=ds_metainfo.val_metric_names,
            metric_extra_kwargs=ds_metainfo.val_metric_extra_kwargs)
    else:
        get_test_data_source_class = get_test_data_source
        test_metric = get_composite_metric(
            metric_names=ds_metainfo.test_metric_names,
            metric_extra_kwargs=ds_metainfo.test_metric_extra_kwargs)
    test_data = get_test_data_source_class(
        ds_metainfo=ds_metainfo,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    batch_fn = get_batch_fn(use_imgrec=ds_metainfo.use_imgrec)

    if args.show_progress:
        from tqdm import tqdm
        test_data = tqdm(test_data)

    assert (args.use_pretrained or args.resume.strip() or args.calc_flops_only)
    test(
        net=net,
        test_data=test_data,
        batch_fn=batch_fn,
        data_source_needs_reset=ds_metainfo.use_imgrec,
        metric=test_metric,
        dtype=args.dtype,
        ctx=ctx,
        input_image_size=input_image_size,
        in_channels=args.in_channels,
        # calc_weight_count=(not log_file_exist),
        calc_weight_count=True,
        calc_flops=args.calc_flops,
        calc_flops_only=args.calc_flops_only,
        extended_log=True)


if __name__ == "__main__":
    main()
