import argparse
import time
import logging
import os
import warnings
import random
import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# from common.logger_utils import initialize_logging
from cvutil.logger import initialize_logging
from common.train_log_param_saver import TrainLogParamSaver
from pytorch.cifar1 import add_dataset_parser_arguments, get_train_data_loader, get_val_data_loader
from pytorch.utils import prepare_pt_context, prepare_model, validate1, accuracy, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for image classification (PyTorch/CIFAR)',
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
        default=128,
        help='training batch size per device (CPU/GPU).')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
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
        help='learning rate. default is 0.1.')
    parser.add_argument(
        '--lr-mode',
        type=str,
        default='step',
        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=0.1,
        help='decay rate of learning rate. default is 0.1.')
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
        '--warmup-lr',
        type=float,
        default=0.0,
        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=0,
        help='number of warmup epochs.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.0001,
        help='weight decay rate. default is 0.0001.')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=200,
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
        default='torch, torchvision',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def init_rand(seed):
    if seed <= 0:
        seed = np.random.randint(10000)
    else:
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def prepare_trainer(net,
                    optimizer_name,
                    wd,
                    momentum,
                    lr_mode,
                    lr,
                    lr_decay_period,
                    lr_decay_epoch,
                    lr_decay,
                    # warmup_epochs,
                    # batch_size,
                    num_epochs,
                    # num_training_samples,
                    state_file_path):

    optimizer_name = optimizer_name.lower()
    if (optimizer_name == 'sgd') or (optimizer_name == 'nag'):
        optimizer = torch.optim.SGD(
            params=net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
            nesterov=(optimizer_name == 'nag'))
    else:
        raise ValueError("Usupported optimizer: {}".format(optimizer_name))

    if state_file_path:
        checkpoint = torch.load(state_file_path)
        if type(checkpoint) == dict:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = None
    else:
        start_epoch = None

    cudnn.benchmark = True

    lr_mode = lr_mode.lower()
    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(',')]
    if (lr_mode == 'step') and (lr_decay_period != 0):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=lr_decay_period,
            gamma=lr_decay,
            last_epoch=-1)
    elif (lr_mode == 'multistep') or ((lr_mode == 'step') and (lr_decay_period == 0)):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_epoch,
            gamma=lr_decay,
            last_epoch=-1)
    elif lr_mode == 'cosine':
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            last_epoch=(num_epochs - 1))
    else:
        raise ValueError("Usupported lr_scheduler: {}".format(lr_mode))

    return optimizer, lr_scheduler, start_epoch


def save_params(file_stem,
                state):
    torch.save(
        obj=state['state_dict'],
        f=(file_stem + '.pth'))
    torch.save(
        obj=state,
        f=(file_stem + '.states'))


def train_epoch(epoch,
                acc_metric_train,
                net,
                train_data,
                use_cuda,
                L,
                optimizer,
                # lr_scheduler,
                batch_size,
                log_interval):

    tic = time.time()
    net.train()
    acc_metric_train.reset()
    train_loss = 0.0

    btic = time.time()
    for i, (data, target) in enumerate(train_data):
        if use_cuda:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        output = net(data)
        loss = L(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        acc_train_value = accuracy(output, target, topk=(1, ))
        acc_metric_train.update(acc_train_value[0], data.size(0))

        if log_interval and not (i + 1) % log_interval:
            acc_train_value = acc_metric_train.avg.item()
            err_train_value = 1.0 - acc_train_value
            speed = batch_size * log_interval / (time.time() - btic)
            logging.info('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\terr={:.4f}\tlr={:.4f}'.format(
                epoch + 1, i, speed, err_train_value, optimizer.param_groups[0]['lr']))
            btic = time.time()

    acc_train_value = acc_metric_train.avg.item()
    err_train_value = 1.0 - acc_train_value
    train_loss /= (i + 1)
    throughput = int(batch_size * (i + 1) / (time.time() - tic))

    logging.info('[Epoch {}] training: err={:.4f}\tloss={:.4f}'.format(
        epoch + 1, err_train_value, train_loss))
    logging.info('[Epoch {}] speed: {:.2f} samples/sec\ttime cost: {:.2f} sec'.format(
        epoch + 1, throughput, time.time() - tic))

    return err_train_value, train_loss


def train_net(batch_size,
              num_epochs,
              start_epoch1,
              train_data,
              val_data,
              net,
              optimizer,
              lr_scheduler,
              lp_saver,
              log_interval,
              use_cuda):
    acc_metric_val = AverageMeter()
    acc_metric_train = AverageMeter()

    L = nn.CrossEntropyLoss()
    if use_cuda:
        L = L.cuda()

    assert (type(start_epoch1) == int)
    assert (start_epoch1 >= 1)
    if start_epoch1 > 1:
        logging.info('Start training from [Epoch {}]'.format(start_epoch1))
        err_val = validate1(
            accuracy_metric=acc_metric_val,
            net=net,
            val_data=val_data,
            use_cuda=use_cuda)
        logging.info('[Epoch {}] validation: err={:.4f}'.format(
            start_epoch1 - 1, err_val))

    gtic = time.time()
    for epoch in range(start_epoch1 - 1, num_epochs):
        lr_scheduler.step()

        err_train, train_loss = train_epoch(
            epoch,
            acc_metric_train,
            net,
            train_data,
            use_cuda,
            L,
            optimizer,
            # lr_scheduler,
            batch_size,
            log_interval)

        err_val = validate1(
            accuracy_metric=acc_metric_val,
            net=net,
            val_data=val_data,
            use_cuda=use_cuda)

        logging.info('[Epoch {}] validation: err={:.4f}'.format(
            epoch + 1, err_val))

        if lp_saver is not None:
            state = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            lp_saver_kwargs = {'state': state}
            lp_saver.epoch_test_end_callback(
                epoch1=(epoch + 1),
                params=[err_val, err_train, train_loss, optimizer.param_groups[0]['lr']],
                **lp_saver_kwargs)

    logging.info('Total time cost: {:.2f} sec'.format(time.time() - gtic))
    if lp_saver is not None:
        logging.info('Best err: {:.4f} at {} epoch'.format(
            lp_saver.best_eval_metric_value, lp_saver.best_eval_metric_epoch))


def main():
    args = parse_args()
    args.seed = init_rand(seed=args.seed)

    _, _ = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        main_script_path=__file__,
        script_args=args)

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda)

    train_data = get_train_data_loader(
        dataset_name=args.dataset,
        dataset_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers)

    val_data = get_val_data_loader(
        dataset_name=args.dataset,
        dataset_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers)

    # num_training_samples = 1281167
    optimizer, lr_scheduler, start_epoch = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        wd=args.wd,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        lr_decay_period=args.lr_decay_period,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay=args.lr_decay,
        # warmup_epochs=args.warmup_epochs,
        # batch_size=batch_size,
        num_epochs=args.num_epochs,
        # num_training_samples=num_training_samples,
        state_file_path=args.resume_state)
    # if start_epoch is not None:
    #     args.start_epoch = start_epoch

    if args.save_dir and args.save_interval:
        lp_saver = TrainLogParamSaver(
            checkpoint_file_name_prefix='{}_{}'.format(args.dataset.lower(), args.model),
            last_checkpoint_file_name_suffix="last",
            best_checkpoint_file_name_suffix=None,
            last_checkpoint_dir_path=args.save_dir,
            best_checkpoint_dir_path=None,
            last_checkpoint_file_count=2,
            best_checkpoint_file_count=2,
            checkpoint_file_save_callback=save_params,
            checkpoint_file_exts=('.pth', '.states'),
            save_interval=args.save_interval,
            num_epochs=args.num_epochs,
            param_names=['Val.Err', 'Train.Err', 'Train.Loss', 'LR'],
            acc_ind=0,
            # bigger=[True],
            # mask=None,
            score_log_file_path=os.path.join(args.save_dir, 'score.log'),
            score_log_attempt_value=args.attempt,
            best_map_log_file_path=os.path.join(args.save_dir, 'best_map.log'))
    else:
        lp_saver = None

    train_net(
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        start_epoch1=args.start_epoch,
        train_data=train_data,
        val_data=val_data,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lp_saver=lp_saver,
        log_interval=args.log_interval,
        use_cuda=use_cuda)


if __name__ == '__main__':
    main()
