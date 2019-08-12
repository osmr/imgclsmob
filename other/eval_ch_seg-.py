import argparse
import time
import logging

from chainer import cuda, global_config
from chainer import iterators

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from common.logger_utils import initialize_logging
from chainer_.utils import prepare_model
from chainer_.seg_utils1 import add_dataset_parser_arguments
from chainer_.seg_utils1 import get_test_dataset
from chainer_.seg_utils1 import SegPredictor
from chainer_.seg_utils1 import get_metainfo
from chainer_.metrics.seg_metrics import PixelAccuracyMetric, MeanIoUMetric


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a model for image segmentation (Chainer/VOC2012/ADE20K/Cityscapes/COCO)',
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
        help='enable using pretrained model from gluon.')
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='resume from previously saved parameters if not None')

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
        default='chainer, chainercv',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='cupy-cuda92, cupy-cuda100, chainer, chainercv',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def test(net,
         test_dataset,
         num_gpus,
         num_classes,
         calc_weight_count=False,
         extended_log=False,
         dataset_metainfo=None):
    assert (dataset_metainfo is not None)
    tic = time.time()

    it = iterators.SerialIterator(
        dataset=test_dataset,
        batch_size=1,
        repeat=False,
        shuffle=False)

    predictor = SegPredictor(base_model=net)

    if num_gpus > 0:
        predictor.to_gpu()

    if calc_weight_count:
        weight_count = net.count_params()
        logging.info('Model: {} trainable parameters'.format(weight_count))

    in_values, out_values, rest_values = apply_to_iterator(
        predictor.predict,
        it,
        hook=ProgressHook(len(test_dataset)))
    del in_values

    pred_labels, = out_values
    gt_labels, = rest_values

    metrics = []
    pix_acc_macro_average = False
    metrics.append(PixelAccuracyMetric(
        vague_idx=dataset_metainfo["vague_idx"],
        use_vague=dataset_metainfo["use_vague"],
        macro_average=pix_acc_macro_average))
    mean_iou_macro_average = False
    metrics.append(MeanIoUMetric(
        num_classes=num_classes,
        vague_idx=dataset_metainfo["vague_idx"],
        use_vague=dataset_metainfo["use_vague"],
        bg_idx=dataset_metainfo["background_idx"],
        ignore_bg=dataset_metainfo["ignore_bg"],
        macro_average=mean_iou_macro_average))

    labels = iter(gt_labels)
    preds = iter(pred_labels)
    for label, pred in zip(labels, preds):
        for metric in metrics:
            metric.update(label, pred)

    accuracy_info = [metric.get() for metric in metrics]
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


def main():
    args = parse_args()

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    global_config.train = False

    num_gpus = args.num_gpus
    if num_gpus > 0:
        cuda.get_device(0).use()

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        net_extra_kwargs={"aux": False, "fixed_size": False},
        use_gpus=(num_gpus > 0))

    test_dataset = get_test_dataset(
        dataset_name=args.dataset,
        dataset_dir=args.data_dir)

    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        test_dataset=test_dataset,
        num_gpus=num_gpus,
        num_classes=args.num_classes,
        calc_weight_count=True,
        extended_log=True,
        dataset_metainfo=get_metainfo(args.dataset))


if __name__ == '__main__':
    main()
