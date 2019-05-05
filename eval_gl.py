import os
import argparse
from common.logger_utils import initialize_logging
from gluon.utils import prepare_mx_context, prepare_model
from gluon.utils import get_composite_metric
from gluon.cls_eval_utils import add_eval_cls_parser_arguments, test
from gluon.cls_eval_utils import get_dataset_metainfo
from gluon.cls_eval_utils import get_batch_fn
from gluon.imagenet1k_utils import get_val_data_source


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification (Gluon/ImageNet1K)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K_rec",
        help="dataset name. options are ImageNet1K, ImageNet1K_rec and CUB_200_2011")

    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    work_dir_path = os.path.join("..", "imgclsmob_data")
    dataset_metainfo.add_dataset_parser_arguments(parser, work_dir_path)

    add_eval_cls_parser_arguments(parser)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

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
        classes=args.num_classes,
        in_channels=args.in_channels,
        do_hybridize=(not args.calc_flops),
        ctx=ctx)

    assert (hasattr(net, "in_size"))
    input_image_size = net.in_size

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    val_data = get_val_data_source(
        dataset_metainfo=ds_metainfo,
        dataset_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_image_size=input_image_size,
        resize_inv_factor=args.resize_inv_factor)
    batch_fn = get_batch_fn(use_imgrec=ds_metainfo.use_imgrec)

    assert (args.use_pretrained or args.resume.strip() or args.calc_flops_only)
    test(
        net=net,
        val_data=val_data,
        batch_fn=batch_fn,
        data_source_needs_reset=ds_metainfo.use_imgrec,
        val_metric=get_composite_metric(ds_metainfo.val_metric_names),
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
