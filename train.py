import argparse
import time
import logging
import os
import sys
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler
from gluoncv import utils as gutils

from env_stats import get_env_stats
from train_log_param_saver import TrainLogParamSaver

from models.shufflenet import *
from models.mobilenet import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification',
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
        default=32,
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
        '--last-gamma',
        action='store_true',
        help='whether to initialize the gamma of the last BN layer in each bottleneck to zero')
    parser.add_argument(
        '--use_se',
        action='store_true',
        help='use SE layers or not in resnext. default is false.')
    parser.add_argument(
        '--batch-norm',
        action='store_true',
        help='enable batch normalization or not in vgg. default is false.')

    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        help='number of batches to wait before logging.')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='saving parameters epoch interval, best model will always be saved')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='params',
        help='directory of saved models')
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
        default='mxnet-cu92, gluoncv',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def prepare_logger(log_file_path):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_dir_path = os.path.dirname(log_file_path)
    if log_dir_path and not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    return logger


def init_rand(seed):
    if seed <= 0:
        seed = np.random.randint(10000)
    gutils.random.seed(seed)
    return seed


def prepare_mx_context(num_gpus,
                       batch_size):
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size *= max(1, num_gpus)
    return ctx, batch_size


def get_data_rec(rec_train,
                 rec_train_idx,
                 rec_val,
                 rec_val_idx,
                 batch_size,
                 num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = 256,
        data_shape          = (3, 224, 224),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, batch_fn


def get_data_loader(data_dir,
                    batch_size,
                    num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256, keep_ratio=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_data, val_data, batch_fn


def _get_model(name, **kwargs):
    models = {
        'mobilenet1_0': mobilenet1_0,
        'mobilenet0_75': mobilenet0_75,
        'mobilenet0_5': mobilenet0_5,
        'mobilenet0_25': mobilenet0_25,
        'fd_mobilenet1_0': fd_mobilenet1_0,
        'fd_mobilenet0_75': fd_mobilenet0_75,
        'fd_mobilenet0_5': fd_mobilenet0_5,
        'fd_mobilenet0_25': fd_mobilenet0_25,
        'shufflenet1_0_g1': shufflenet1_0_g1,
        'shufflenet1_0_g2': shufflenet1_0_g2,
        'shufflenet1_0_g3': shufflenet1_0_g3,
        'shufflenet1_0_g4': shufflenet1_0_g4,
        'shufflenet1_0_g8': shufflenet1_0_g8,
        'shufflenet0_5_g1': shufflenet0_5_g1,
        'shufflenet0_5_g2': shufflenet0_5_g2,
        'shufflenet0_5_g3': shufflenet0_5_g3,
        'shufflenet0_5_g4': shufflenet0_5_g4,
        'shufflenet0_5_g8': shufflenet0_5_g8,
        'shufflenet0_25_g1': shufflenet0_25_g1,
        'shufflenet0_25_g2': shufflenet0_25_g2,
        'shufflenet0_25_g3': shufflenet0_25_g3,
        'shufflenet0_25_g4': shufflenet0_25_g4,
        'shufflenet0_25_g8': shufflenet0_25_g8,
        }
    try:
        net = get_model(name, **kwargs)
        return net
    except ValueError as e:
        upstream_supported = str(e)
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (upstream_supported, '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net

def prepare_model(model_name,
                  classes,
                  use_pretrained,
                  pretrained_model_file_path,
                  batch_norm,
                  use_se,
                  last_gamma,
                  dtype,
                  ctx):
    kwargs = {'ctx': ctx,
              'pretrained': use_pretrained,
              'classes': classes}

    if model_name.startswith('vgg'):
        kwargs['batch_norm'] = batch_norm
    elif model_name.startswith('resnext'):
        kwargs['use_se'] = use_se

    if last_gamma:
        kwargs['last_gamma'] = True

    net = _get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        net.load_parameters(
            filename=pretrained_model_file_path,
            ctx=ctx)

    net.cast(dtype)

    net.hybridize(
        static_alloc=True,
        static_shape=True)

    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    return net


def prepare_trainer(net,
                    optimizer_name,
                    wd,
                    momentum,
                    lr_mode,
                    lr,
                    lr_decay_period,
                    lr_decay_epoch,
                    lr_decay,
                    warmup_epochs,
                    batch_size,
                    num_epochs,
                    num_training_samples,
                    dtype):

    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(',')]
    num_batches = num_training_samples // batch_size
    lr_scheduler = LRScheduler(
        mode=lr_mode,
        baselr=lr,
        niters=num_batches,
        nepochs=num_epochs,
        step=lr_decay_epoch,
        step_factor=lr_decay,
        power=2,
        warmup_epochs=warmup_epochs)

    optimizer_params = {'wd': wd,
                        'momentum': momentum,
                        'lr_scheduler': lr_scheduler}
    if dtype != 'float32':
        optimizer_params['multi_precision'] = True

    trainer = gluon.Trainer(
        params=net.collect_params(),
        optimizer=optimizer_name,
        optimizer_params=optimizer_params)

    return trainer, lr_scheduler


def calc_net_weight_count(net):
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if param.shape is None:
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def save_params(file_path,
                net):
    net.save_parameters(file_path)


def test(acc_top1,
         acc_top5,
         net,
         val_data,
         batch_fn,
         use_rec,
         dtype,
         ctx):
    if use_rec:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for batch in val_data:
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)


def train(batch_size,
          num_epochs,
          start_epoch,
          train_data,
          val_data,
          batch_fn,
          use_rec,
          dtype,
          net,
          trainer,
          lr_scheduler,
          lp_saver,
          log_interval,
          ctx):

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    L = gluon.loss.SoftmaxCrossEntropyLoss()

    best_val_err_map = [1.0]

    assert (type(start_epoch) == int)
    assert (start_epoch >= 1)
    if start_epoch > 1:
        logging.info('Start training from [Epoch {}]'.format(start_epoch))
        err_top1_val, err_top5_val = test(
            acc_top1=acc_top1,
            acc_top5=acc_top5,
            net=net,
            val_data=val_data,
            batch_fn=batch_fn,
            use_rec=use_rec,
            dtype=dtype,
            ctx=ctx)
        logging.info('[Epoch {}] validation: err-top1={:.3f}\terr-top5={:.3f}'.format(
            start_epoch - 1, err_top1_val, err_top5_val))

    gtic = time.time()
    for epoch in range(start_epoch - 1, num_epochs):
        tic = time.time()
        if use_rec:
            train_data.reset()
        acc_top1.reset()
        train_loss = 0.0
        btic = time.time()

        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)
            with ag.record():
                outputs = [net(X.astype(dtype, copy=False)) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)

            if epoch == 0 and i == 0:
                weight_count = calc_net_weight_count(net)
                logging.info('Model: {} trainable parameters'.format(weight_count))
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            acc_top1.update(label, outputs)
            if log_interval and not (i + 1) % log_interval:
                _, top1 = acc_top1.get()
                err_top1_train = 1.0 - top1
                speed = batch_size * log_interval / (time.time() - btic)
                logging.info('Epoch[{}] Batch [{}]\tSpeed: {:.3f} samples/sec\ttop1-err={:.3f}\tlr={:.3f}'.format(
                    epoch + 1, i, speed, err_top1_train, trainer.learning_rate))
                btic = time.time()

        _, top1 = acc_top1.get()
        err_top1_train = 1.0 - top1
        train_loss /= (i + 1)
        throughput = int(batch_size * (i + 1) /(time.time() - tic))

        err_top1_val, err_top5_val = test(
            acc_top1=acc_top1,
            acc_top5=acc_top5,
            net=net,
            val_data=val_data,
            batch_fn=batch_fn,
            use_rec=use_rec,
            dtype=dtype,
            ctx=ctx)

        logging.info('[Epoch {}] training: err-top1={:.3f}\tloss={:.3f}'.format(
            epoch + 1, err_top1_train, train_loss))
        logging.info('[Epoch {}] speed: {:.3f} samples/sec\ttime cost: {:.3f}'.format(
            epoch + 1, throughput, time.time()-tic))
        logging.info('[Epoch {}] validation: err-top1={:.3f}\terr-top5={:.3f}'.format(
            epoch + 1, err_top1_val, err_top5_val))

        lp_saver_kwargs = {'net': net}
        lp_saver.epoch_test_end_callback(
            epoch=(epoch + 1),
            params=[err_top1_val, err_top1_train, err_top5_val, train_loss],
            **lp_saver_kwargs)

    logging.info('Total time cost: {:.3f} sec; best-err-top5: {:.3f}'.format(
        time.time() - gtic, best_val_err_map[0]))


def main():
    args = parse_args()
    args.seed = init_rand(seed=args.seed)
    prepare_logger(log_file_path=os.path.join(args.save_dir, args.logging_file_name))
    logging.info("Script command line:\n{}".format(" ".join(sys.argv)))
    logging.info("Script arguments:\n{}".format(args))
    logging.info("Env_stats:\n{}".format(get_env_stats(
        packages=args.log_packages.replace(' ', '').split(','),
        pip_packages=args.log_pip_packages.replace(' ', '').split(','))))

    ctx, batch_size = prepare_mx_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    classes = 1000
    net = prepare_model(
        model_name=args.model,
        classes=classes,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        batch_norm=args.batch_norm,
        use_se=args.use_se,
        last_gamma=args.last_gamma,
        dtype=args.dtype,
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

    num_training_samples = 1281167
    trainer, lr_scheduler = prepare_trainer(
        net=net,
        optimizer_name=args.optimizer_name,
        wd=args.wd,
        momentum=args.momentum,
        lr_mode=args.lr_mode,
        lr=args.lr,
        lr_decay_period=args.lr_decay_period,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay=args.lr_decay,
        warmup_epochs=args.warmup_epochs,
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        num_training_samples=num_training_samples,
        dtype=args.dtype)

    if args.save_dir and args.save_interval:
        lp_saver = TrainLogParamSaver(
            checkpoint_file_name_prefix='imagenet_{}'.format(args.model),
            last_checkpoint_file_name_suffix="last",
            best_checkpoint_file_name_suffix=None,
            last_checkpoint_dir_path=args.save_dir,
            best_checkpoint_dir_path=None,
            last_checkpoint_file_count=2,
            best_checkpoint_file_count=2,
            checkpoint_file_save_callback=save_params,
            save_interval=args.save_interval,
            num_epochs=args.num_epochs,
            param_names=['Val.Top1', 'Train.Top1', 'Val.Top5', 'Train.Loss'],
            acc_ind=2,
            #bigger=[True],
            #mask=None,
            score_log_file_path=os.path.join(args.save_dir, 'score.log'),
            score_log_attempt_value=1,
            best_map_log_file_path=os.path.join(args.save_dir, 'best_map.log'))
    else:
        lp_saver = None

    train(
        batch_size=batch_size,
        num_epochs=args.num_epochs,
        start_epoch=args.start_epoch,
        train_data=train_data,
        val_data=val_data,
        batch_fn=batch_fn,
        use_rec=args.use_rec,
        dtype=args.dtype,
        net=net,
        trainer=trainer,
        lr_scheduler=lr_scheduler,
        lp_saver=lp_saver,
        log_interval=args.log_interval,
        ctx=ctx)


if __name__ == '__main__':
    main()

