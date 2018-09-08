import argparse
import time
import logging
import numpy as np

from chainer import cuda, global_config
import chainer.functions as F

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from common.logger_utils import initialize_logging
from chainer_.imagenet_predictor import ImagenetPredictor
from chainer_.top_k_accuracy import top_k_accuracy
from chainer_.utils import get_val_data_iterator, prepare_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model for image classification (Chainer)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/imagenet',
        help='training and validation pictures to use.')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='type of model to use. see vision_model for options.')
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
        '--batch-size',
        type=int,
        default=32,
        help='training batch size per device (CPU/GPU).')

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
        default='cupy-cuda92, chainer, chainercv',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def test(net,
         val_iterator,
         val_dataset_len,
         num_gpus,
         calc_weight_count=False,
         extended_log=False):
    tic = time.time()

    predictor = ImagenetPredictor(base_model=net)

    if num_gpus > 0:
        predictor.to_gpu()

    if calc_weight_count:
        weight_count = net.count_params()
        logging.info('Model: {} trainable parameters'.format(weight_count))

    in_values, out_values, rest_values = apply_to_iterator(
        predictor.predict,
        val_iterator,
        hook=ProgressHook(val_dataset_len))
    del in_values

    pred_probs, = out_values
    gt_labels, = rest_values

    y = np.array(list(pred_probs))
    t = np.array(list(gt_labels))

    top1_acc = F.accuracy(
        y=y,
        t=t).data
    top5_acc = top_k_accuracy(
        y=y,
        t=t,
        k=5).data
    err_top1_val = 1.0 - top1_acc
    err_top5_val = 1.0 - top5_acc

    if extended_log:
        logging.info('Test: err-top1={top1:.4f} ({top1})\terr-top5={top5:.4f} ({top5})'.format(
            top1=err_top1_val, top5=err_top5_val))
    else:
        logging.info('Test: err-top1={top1:.4f}\terr-top5={top5:.4f}'.format(
            top1=err_top1_val, top5=err_top5_val))
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

    num_classes = 1000
    net = prepare_model(
        model_name=args.model,
        classes=num_classes,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        num_gpus=num_gpus)

    val_iterator, val_dataset_len = get_val_data_iterator(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=num_classes)

    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        val_iterator=val_iterator,
        val_dataset_len=val_dataset_len,
        num_gpus=num_gpus,
        calc_weight_count=True,
        extended_log=True)


if __name__ == '__main__':
    main()
