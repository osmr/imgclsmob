import argparse
import time
import logging

from common.logger_utils import initialize_logging
from pytorch.model_stats import measure_model
from pytorch.utils import prepare_pt_context, prepare_model, get_data_loader, calc_net_weight_count, validate,\
    AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model for image classification (PyTorch)',
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
        '--calc-flops',
        dest='calc_flops',
        action='store_true',
        help='calculate FLOPs')

    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='size of the input for model. default is 224')
    parser.add_argument(
        '--resize-inv-factor',
        type=float,
        default=0.875,
        help='inverted ratio for input image crop. default is 0.875')

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
         val_data,
         use_cuda,
         input_image_size,
         calc_weight_count=False,
         calc_flops=False,
         extended_log=False):
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    tic = time.time()
    err_top1_val, err_top5_val = validate(
        acc_top1=acc_top1,
        acc_top5=acc_top5,
        net=net,
        val_data=val_data,
        use_cuda=use_cuda)
    if calc_weight_count:
        weight_count = calc_net_weight_count(net)
        logging.info('Model: {} trainable parameters'.format(weight_count))
    if calc_flops:
        n_flops, n_params = measure_model(net, input_image_size, input_image_size)
        logging.info('Params: {} ({:.2f}M), FLOPs: {} ({:.2f}M)'.format(
            n_params, n_params / 1e6, n_flops, n_flops / 1e6))
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

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda)
    if hasattr(net, 'module'):
        input_image_size = net.module.in_size[0] if hasattr(net.module, 'in_size') else args.input_size
    else:
        input_image_size = net.in_size[0] if hasattr(net, 'in_size') else args.input_size

    train_data, val_data = get_data_loader(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_image_size=input_image_size,
        resize_inv_factor=args.resize_inv_factor)

    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        val_data=val_data,
        use_cuda=use_cuda,
        # calc_weight_count=(not log_file_exist),
        input_image_size=input_image_size,
        calc_weight_count=True,
        calc_flops=args.calc_flops,
        extended_log=True)


if __name__ == '__main__':
    main()
