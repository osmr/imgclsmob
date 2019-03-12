import argparse
import time
import logging
import numpy as np

from chainer import cuda, global_config
import chainer.functions as F

from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from common.logger_utils import initialize_logging
from chainer_.utils import prepare_model
from chainer_.cifar import add_dataset_parser_arguments
from chainer_.cifar import get_val_data_iterator
from chainer_.cifar import CIFARPredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a model for image classification (Chainer/CIFAR/SVHN)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        type=str,
        default="CIFAR10",
        help='dataset name. options are CIFAR10, CIFAR100, and SVHN')

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

    predictor = CIFARPredictor(base_model=net)

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

    acc_val_value = F.accuracy(
        y=y,
        t=t).data
    err_val = 1.0 - acc_val_value

    if extended_log:
        logging.info('Test: err={err:.4f} ({err})'.format(
            err=err_val))
    else:
        logging.info('Test: err={err:.4f}'.format(
            err=err_val))
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
        num_gpus=num_gpus)

    val_iterator, val_dataset_len = get_val_data_iterator(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

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
