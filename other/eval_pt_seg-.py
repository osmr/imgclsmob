import argparse
import time
import logging

from common.logger_utils import initialize_logging
from pytorch.model_stats import measure_model
from pytorch.seg_utils import add_dataset_parser_arguments, get_test_data_loader, get_metainfo, validate1
from pytorch.utils import prepare_pt_context, prepare_model, calc_net_weight_count
from pytorch.metrics.seg_metrics import PixelAccuracyMetric, MeanIoUMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a model for image segmentation (PyTorch/VOC2012/ADE20K/Cityscapes/COCO)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        type=str,
        default="VOC",
        help='dataset name. options are VOC, ADE20K, Cityscapes, COCO')

    args, _ = parser.parse_known_args()
    add_dataset_parser_arguments(parser, args.dataset)

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='type of model to use. see model_provider for options.')
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        help='enable using pretrained model from github.')
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='resume from previously saved parameters if not None')
    parser.add_argument(
        '--calc-flops',
        dest='calc_flops',
        action='store_true',
        help='calculate FLOPs')
    parser.add_argument(
        '--calc-flops-only',
        dest='calc_flops_only',
        action='store_true',
        help='calculate FLOPs without quality estimation')
    parser.add_argument(
        '--remove-module',
        action='store_true',
        help='enable if stored model has module')

    parser.add_argument(
        '--num-gpus',
        type=int,
        default=0,
        help='number of gpus to use.')
    parser.add_argument(
        '-j',
        '--num-data-workers',
        dest='num_workers',
        default=4,
        type=int,
        help='number of preprocessing workers')

    parser.add_argument(
        '--save-dir',
        type=str,
        default='',
        help='directory of saved models and log-files')
    parser.add_argument(
        '--logging-file-name',
        type=str,
        default='train.log',
        help='filename of training log')

    parser.add_argument(
        '--log-packages',
        type=str,
        default='torch, torchvision',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def test(net,
         test_data,
         use_cuda,
         input_image_size,
         in_channels,
         num_classes,
         calc_weight_count=False,
         calc_flops=False,
         calc_flops_only=True,
         extended_log=False,
         dataset_metainfo=None):
    assert (dataset_metainfo is not None)
    if not calc_flops_only:
        metric = []
        pix_acc_macro_average = False
        metric.append(PixelAccuracyMetric(
            vague_idx=dataset_metainfo["vague_idx"],
            use_vague=dataset_metainfo["use_vague"],
            macro_average=pix_acc_macro_average))
        mean_iou_macro_average = False
        metric.append(MeanIoUMetric(
            num_classes=num_classes,
            vague_idx=dataset_metainfo["vague_idx"],
            use_vague=dataset_metainfo["use_vague"],
            bg_idx=dataset_metainfo["background_idx"],
            ignore_bg=dataset_metainfo["ignore_bg"],
            macro_average=mean_iou_macro_average))
        tic = time.time()
        accuracy_info = validate1(
            accuracy_metrics=metric,
            net=net,
            val_data=test_data,
            use_cuda=use_cuda)
        pix_acc = accuracy_info[0][1]
        mean_iou = accuracy_info[1][1]
        pix_macro = "macro" if pix_acc_macro_average else "micro"
        iou_macro = "macro" if mean_iou_macro_average else "micro"
        if extended_log:
            logging.info(
                "Test: {pix_macro}-pix_acc={pix_acc:.4f} ({pix_acc}), "
                "{iou_macro}-mean_iou={mean_iou:.4f} ({mean_iou})".format(
                    pix_macro=pix_macro, pix_acc=pix_acc, iou_macro=iou_macro, mean_iou=mean_iou))
        else:
            logging.info("Test: {pix_macro}-pix_acc={pix_acc:.4f}, {iou_macro}-mean_iou={mean_iou:.4f}".format(
                pix_macro=pix_macro, pix_acc=pix_acc, iou_macro=iou_macro, mean_iou=mean_iou))
        logging.info('Time cost: {:.4f} sec'.format(
            time.time() - tic))

    if calc_weight_count:
        weight_count = calc_net_weight_count(net)
        if not calc_flops:
            logging.info('Model: {} trainable parameters'.format(weight_count))
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


def main():
    args = parse_args()

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=1)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda,
        net_extra_kwargs={"aux": False, "fixed_size": False},
        load_ignore_extra=True,
        remove_module=args.remove_module)
    if hasattr(net, 'module'):
        input_image_size = net.module.in_size[0] if hasattr(net.module, 'in_size') else args.input_size
    else:
        input_image_size = net.in_size[0] if hasattr(net, 'in_size') else args.input_size

    test_data = get_test_data_loader(
        dataset_name=args.dataset,
        dataset_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers)

    assert (args.use_pretrained or args.resume.strip() or args.calc_flops_only)
    test(
        net=net,
        test_data=test_data,
        use_cuda=use_cuda,
        # calc_weight_count=(not log_file_exist),
        input_image_size=(input_image_size, input_image_size),
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        calc_weight_count=True,
        calc_flops=args.calc_flops,
        calc_flops_only=args.calc_flops_only,
        extended_log=True,
        dataset_metainfo=get_metainfo(args.dataset))


if __name__ == '__main__':
    main()
