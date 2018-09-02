import argparse
import time
import logging

import mxnet as mx

from common.logger_utils import initialize_logging
from gluon.utils import prepare_mx_context, prepare_model, get_data_rec, get_data_loader, calc_net_weight_count,\
    validate


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model for image classification (Gluon)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/imagenet',
        help='training and validation pictures to use.')
    parser.add_argument(
        '--rec-train',
        type=str,
        default='../imgclsmob_data/imagenet/rec/train.rec',
        help='the training data')
    parser.add_argument(
        '--rec-train-idx',
        type=str,
        default='../imgclsmob_data/imagenet/rec/train.idx',
        help='the index of training data')
    parser.add_argument(
        '--rec-val',
        type=str,
        default='../imgclsmob_data/imagenet/rec/val.rec',
        help='the validation data')
    parser.add_argument(
        '--rec-val-idx',
        type=str,
        default='../imgclsmob_data/imagenet/rec/val.idx',
        help='the index of validation data')
    parser.add_argument(
        '--use-rec',
        action='store_true',
        help='use image record iter for data input. default is false.')

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
        default='mxnet-cu92, gluoncv',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def test(net,
         val_data,
         batch_fn,
         use_rec,
         dtype,
         ctx,
         calc_weight_count=False,
         extended_log=False):
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    tic = time.time()
    err_top1_val, err_top5_val = validate(
        acc_top1=acc_top1,
        acc_top5=acc_top5,
        net=net,
        val_data=val_data,
        batch_fn=batch_fn,
        use_rec=use_rec,
        dtype=dtype,
        ctx=ctx)
    if calc_weight_count:
        weight_count = calc_net_weight_count(net)
        logging.info('Model: {} trainable parameters'.format(weight_count))
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

    ctx, batch_size = prepare_mx_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    num_classes = 1000
    net = prepare_model(
        model_name=args.model,
        classes=num_classes,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        batch_norm=args.batch_norm,
        last_gamma=args.last_gamma,
        dtype=args.dtype,
        tune_layers=args.tune_layers,
        ctx=ctx)

    if args.use_rec:
        train_data, val_data, batch_fn = get_data_rec(
            rec_train=args.rec_train,
            rec_train_idx=args.rec_train_idx,
            rec_val=args.rec_val,
            rec_val_idx=args.rec_val_idx,
            batch_size=batch_size,
            num_workers=args.num_workers)
    else:
        train_data, val_data, batch_fn = get_data_loader(
            data_dir=args.data_dir,
            batch_size=batch_size,
            num_workers=args.num_workers)

    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        val_data=val_data,
        batch_fn=batch_fn,
        use_rec=args.use_rec,
        dtype=args.dtype,
        ctx=ctx,
        #calc_weight_count=(not log_file_exist),
        calc_weight_count=True,
        extended_log=True)


if __name__ == '__main__':
    main()

