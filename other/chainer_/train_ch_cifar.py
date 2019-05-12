import argparse
import numpy as np

import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer.serializers import save_npz

from common.logger_utils import initialize_logging
from chainer_.utils import prepare_model
from chainer_.cifar1 import add_dataset_parser_arguments
from chainer_.cifar1 import get_data_iterators


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for image classification (Chainer/CIFAR)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--dataset',
        type=str,
        default="CIFAR10",
        help='dataset name. options are CIFAR10 and CIFAR100')

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
        '--resume-state',
        type=str,
        default='',
        help='resume from previously saved optimizer state if not None')

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
        '--num-epochs',
        type=int,
        default=120,
        help='number of training epochs.')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=1,
        help='starting epoch for resuming, default is 1 for new training')
    parser.add_argument(
        '--attempt',
        type=int,
        default=1,
        help='current number of training')

    parser.add_argument(
        '--optimizer-name',
        type=str,
        default='nag',
        help='optimizer name')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='learning rate. default is 0.1')
    parser.add_argument(
        '--lr-mode',
        type=str,
        default='cosine',
        help='learning rate scheduler mode. options are step, poly and cosine')
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=0.1,
        help='decay rate of learning rate. default is 0.1')
    parser.add_argument(
        '--lr-decay-period',
        type=int,
        default=0,
        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument(
        '--lr-decay-epoch',
        type=str,
        default='40,60',
        help='epoches at which learning rate decays. default is 40,60.')
    parser.add_argument(
        '--target-lr',
        type=float,
        default=1e-8,
        help='ending learning rate; default is 1e-8')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='momentum value for optimizer; default is 0.9')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.0001,
        help='weight decay rate. default is 0.0001.')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        help='number of batches to wait before logging.')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=4,
        help='saving parameters epoch interval, best model will always be saved')
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
        '--seed',
        type=int,
        default=-1,
        help='Random seed to be fixed')
    parser.add_argument(
        '--log-packages',
        type=str,
        default='mxnet',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='mxnet-cu92, cupy-cuda100, gluoncv',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def init_rand(seed):
    if seed <= 0:
        seed = np.random.randint(10000)
    return seed


def prepare_trainer(net,
                    optimizer_name,
                    lr,
                    momentum,
                    num_epochs,
                    train_iter,
                    val_iter,
                    logging_dir_path,
                    num_gpus=0):
    if optimizer_name == "sgd":
        optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=momentum)
    elif optimizer_name == "nag":
        optimizer = chainer.optimizers.NesterovAG(lr=lr, momentum=momentum)
    else:
        raise Exception('Unsupported optimizer: {}'.format(optimizer_name))
    optimizer.setup(net)

    # devices = tuple(range(num_gpus)) if num_gpus > 0 else (-1, )
    devices = (0,) if num_gpus > 0 else (-1,)

    updater = training.updaters.StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        device=devices[0])
    trainer = training.Trainer(
        updater=updater,
        stop_trigger=(num_epochs, 'epoch'),
        out=logging_dir_path)

    val_interval = 100000, 'iteration'
    log_interval = 1000, 'iteration'

    trainer.extend(
        extension=extensions.Evaluator(
            val_iter,
            net,
            device=devices[0]),
        trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(
        extensions.snapshot_object(
            net,
            'model_iter_{.updater.iteration}'),
        trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy',
            'lr']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    return trainer


def save_params(file_stem,
                net,
                trainer):
    save_npz(
        file=file_stem + '.npz',
        obj=net)
    save_npz(
        file=file_stem + '.states',
        obj=trainer)


def main():
    args = parse_args()
    args.seed = init_rand(seed=args.seed)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    num_gpus = args.num_gpus
    if num_gpus > 0:
        cuda.get_device(0).use()
    batch_size = args.batch_size

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        num_gpus=num_gpus)

    train_iter, val_iter = get_data_iterators(
        batch_size=batch_size,
        num_workers=args.num_workers)

    trainer = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        momentum=args.momentum,
        num_epochs=args.num_epochs,
        train_iter=train_iter,
        val_iter=val_iter,
        logging_dir_path=args.save_dir,
        num_gpus=num_gpus)

    # if args.save_dir and args.save_interval:
    #     lp_saver = TrainLogParamSaver(
    #         checkpoint_file_name_prefix='imagenet_{}'.format(args.model),
    #         last_checkpoint_file_name_suffix="last",
    #         best_checkpoint_file_name_suffix=None,
    #         last_checkpoint_dir_path=args.save_dir,
    #         best_checkpoint_dir_path=None,
    #         last_checkpoint_file_count=2,
    #         best_checkpoint_file_count=2,
    #         checkpoint_file_save_callback=save_params,
    #         checkpoint_file_exts=['.npz', '.states'],
    #         save_interval=args.save_interval,
    #         num_epochs=args.num_epochs,
    #         param_names=['Val.Top1', 'Train.Top1', 'Val.Top5', 'Train.Loss', 'LR'],
    #         acc_ind=2,
    #         # bigger=[True],
    #         # mask=None,
    #         score_log_file_path=os.path.join(args.save_dir, 'score.log'),
    #         score_log_attempt_value=args.attempt,
    #         best_map_log_file_path=os.path.join(args.save_dir, 'best_map.log'))
    # else:
    #     lp_saver = None

    trainer.run()


if __name__ == '__main__':
    main()
