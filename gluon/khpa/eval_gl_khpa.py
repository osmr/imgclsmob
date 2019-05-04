import argparse
import time
import logging

import mxnet as mx

from common.logger_utils import initialize_logging
from gluon.utils import prepare_mx_context, prepare_model, calc_net_weight_count
from gluon.khpa.khpa_utils import add_dataset_parser_arguments
from gluon.khpa.khpa_utils import get_batch_fn
from gluon.khpa.khpa_utils import get_val_data_source
from gluon.khpa.khpa_utils import validate


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a model for image classification (Gluon/KHPA)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_dataset_parser_arguments(parser)

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
        '--dtype',
        type=str,
        default='float32',
        help='data type for training. default is float32')
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
        default=512,
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
        default='mxnet',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='mxnet-cu92',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def test(net,
         val_data,
         batch_fn,
         data_source_needs_reset,
         dtype,
         ctx,
         calc_weight_count=False,
         extended_log=False):
    rmse_calc = mx.metric.RMSE()

    tic = time.time()
    rmse_val_value = validate(
        metric_calc=rmse_calc,
        net=net,
        val_data=val_data,
        batch_fn=batch_fn,
        data_source_needs_reset=data_source_needs_reset,
        dtype=dtype,
        ctx=ctx)
    if calc_weight_count:
        weight_count = calc_net_weight_count(net)
        logging.info('Model: {} trainable parameters'.format(weight_count))
    if extended_log:
        logging.info('Test: rmse={rmse:.4f} ({rmse})'.format(
            rmse=rmse_val_value))
    else:
        logging.info('Test: rmse={rmse:.4f}'.format(
            rmse=rmse_val_value))
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

    ctx, batch_size = prepare_mx_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        dtype=args.dtype,
        tune_layers="",
        classes=args.num_classes,
        in_channels=args.in_channels,
        ctx=ctx)
    input_image_size = net.in_size if hasattr(net, 'in_size') else (args.input_size, args.input_size)

    val_data = get_val_data_source(
        dataset_args=args,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_image_size=input_image_size,
        resize_inv_factor=args.resize_inv_factor)
    batch_fn = get_batch_fn()

    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        val_data=val_data,
        batch_fn=batch_fn,
        data_source_needs_reset=args.use_rec,
        dtype=args.dtype,
        ctx=ctx,
        # calc_weight_count=(not log_file_exist),
        calc_weight_count=True,
        extended_log=True)


if __name__ == '__main__':
    main()
