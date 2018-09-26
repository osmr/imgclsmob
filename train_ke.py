import argparse
import time
import logging
import os
import numpy as np
import random

import keras
import mxnet as mx

from common.logger_utils import initialize_logging
from common.train_log_param_saver import TrainLogParamSaver
from keras_.utils import prepare_ke_context, prepare_model, get_data_rec, get_data_generator, backend_agnostic_compile


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification (Keras)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        default='keras',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='keras, keras-mxnet, keras-applications, keras-preprocessing',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def init_rand(seed):
    if seed <= 0:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    return seed


# def prepare_trainer(net,
#                     optimizer_name,
#                     wd,
#                     momentum,
#                     lr,
#                     batch_size,
#                     num_epochs,
#                     dtype,
#                     state_file_path=None):
#
#     backend_agnostic_compile(
#         model=net,
#         loss='categorical_crossentropy',
#         optimizer=keras.optimizers.SGD(
#             lr=lr,
#             momentum=momentum,
#             decay=wd,
#             nesterov=False),
#         metrics=[keras.metrics.categorical_accuracy, keras.metrics.top_k_categorical_accuracy],
#         num_gpus=num_gpus)


def save_params(file_stem,
                net,
                trainer):
    net.save_parameters(file_stem + '.params')
    trainer.save_states(file_stem + '.states')


# def train_epoch(epoch,
#                 net,
#                 acc_top1_train,
#                 train_data,
#                 batch_fn,
#                 use_rec,
#                 dtype,
#                 ctx,
#                 loss_func,
#                 trainer,
#                 lr_scheduler,
#                 batch_size,
#                 log_interval,
#                 mixup,
#                 mixup_epoch_tail,
#                 num_classes,
#                 num_epochs):
#
#     labels_list_inds = None
#     tic = time.time()
#     if use_rec:
#         train_data.reset()
#     acc_top1_train.reset()
#     train_loss = 0.0
#
#     btic = time.time()
#     for i, batch in enumerate(train_data):
#         data_list, labels_list = batch_fn(batch, ctx)
#
#         if mixup:
#             labels_list_inds = labels_list
#             labels_list = [mx.nd.one_hot(Y, depth=num_classes) for Y in labels_list]
#             if epoch < num_epochs - mixup_epoch_tail:
#                 alpha = 1
#                 lam = np.random.beta(alpha, alpha)
#                 data_list = [lam * X + (1 - lam) * X[::-1] for X in data_list]
#                 labels_list = [lam * Y + (1 - lam) * Y[::-1] for Y in labels_list]
#
#         with ag.record():
#             outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
#             loss_list = [loss_func(yhat, y) for yhat, y in zip(outputs_list, labels_list)]
#         for loss in loss_list:
#             loss.backward()
#         lr_scheduler.update(i, epoch)
#         trainer.step(batch_size)
#
#         # if epoch == 0 and i == 0:
#         #     weight_count = calc_net_weight_count(net)
#         #     logging.info('Model: {} trainable parameters'.format(weight_count))
#         train_loss += sum([loss.mean().asscalar() for loss in loss_list]) / len(loss_list)
#
#         acc_top1_train.update(
#             labels=(labels_list if not mixup else labels_list_inds),
#             preds=outputs_list)
#
#         if log_interval and not (i + 1) % log_interval:
#             speed = batch_size * log_interval / (time.time() - btic)
#             btic = time.time()
#             _, top1 = acc_top1_train.get()
#             err_top1_train = 1.0 - top1
#             logging.info('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\ttop1-err={:.4f}\tlr={:.5f}'.format(
#                 epoch + 1, i, speed, err_top1_train, trainer.learning_rate))
#
#     throughput = int(batch_size * (i + 1) / (time.time() - tic))
#     logging.info('[Epoch {}] speed: {:.2f} samples/sec\ttime cost: {:.2f} sec'.format(
#         epoch + 1, throughput, time.time() - tic))
#
#     train_loss /= (i + 1)
#     _, top1 = acc_top1_train.get()
#     err_top1_train = 1.0 - top1
#     logging.info('[Epoch {}] training: err-top1={:.4f}\tloss={:.4f}'.format(
#         epoch + 1, err_top1_train, train_loss))
#
#     return err_top1_train, train_loss
#
#
# def train_net(batch_size,
#               num_epochs,
#               start_epoch1,
#               train_data,
#               val_data,
#               batch_fn,
#               use_rec,
#               dtype,
#               net,
#               trainer,
#               lr_scheduler,
#               lp_saver,
#               log_interval,
#               mixup,
#               mixup_epoch_tail,
#               num_classes,
#               ctx):
#
#     if isinstance(ctx, mx.Context):
#         ctx = [ctx]
#
#     acc_top1_val = mx.metric.Accuracy()
#     acc_top5_val = mx.metric.TopKAccuracy(5)
#     acc_top1_train = mx.metric.Accuracy()
#
#     loss_func = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=(not mixup))
#
#     assert (type(start_epoch1) == int)
#     assert (start_epoch1 >= 1)
#     if start_epoch1 > 1:
#         logging.info('Start training from [Epoch {}]'.format(start_epoch1))
#         err_top1_val, err_top5_val = validate(
#             acc_top1=acc_top1_val,
#             acc_top5=acc_top5_val,
#             net=net,
#             val_data=val_data,
#             batch_fn=batch_fn,
#             use_rec=use_rec,
#             dtype=dtype,
#             ctx=ctx)
#         logging.info('[Epoch {}] validation: err-top1={:.4f}\terr-top5={:.4f}'.format(
#             start_epoch1 - 1, err_top1_val, err_top5_val))
#
#     gtic = time.time()
#     for epoch in range(start_epoch1 - 1, num_epochs):
#         err_top1_train, train_loss = train_epoch(
#             epoch=epoch,
#             net=net,
#             acc_top1_train=acc_top1_train,
#             train_data=train_data,
#             batch_fn=batch_fn,
#             use_rec=use_rec,
#             dtype=dtype,
#             ctx=ctx,
#             loss_func=loss_func,
#             trainer=trainer,
#             lr_scheduler=lr_scheduler,
#             batch_size=batch_size,
#             log_interval=log_interval,
#             mixup=mixup,
#             mixup_epoch_tail=mixup_epoch_tail,
#             num_classes=num_classes,
#             num_epochs=num_epochs)
#
#         err_top1_val, err_top5_val = validate(
#             acc_top1=acc_top1_val,
#             acc_top5=acc_top5_val,
#             net=net,
#             val_data=val_data,
#             batch_fn=batch_fn,
#             use_rec=use_rec,
#             dtype=dtype,
#             ctx=ctx)
#
#         logging.info('[Epoch {}] validation: err-top1={:.4f}\terr-top5={:.4f}'.format(
#             epoch + 1, err_top1_val, err_top5_val))
#
#         if lp_saver is not None:
#             lp_saver_kwargs = {'net': net, 'trainer': trainer}
#             lp_saver.epoch_test_end_callback(
#                 epoch1=(epoch + 1),
#                 params=[err_top1_val, err_top1_train, err_top5_val, train_loss, trainer.learning_rate],
#                 **lp_saver_kwargs)
#
#     logging.info('Total time cost: {:.2f} sec'.format(time.time() - gtic))
#     if lp_saver is not None:
#         logging.info('Best err-top5: {:.4f} at {} epoch'.format(
#             lp_saver.best_eval_metric_value, lp_saver.best_eval_metric_epoch))


def main():
    args = parse_args()
    args.seed = init_rand(seed=args.seed)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    batch_size = prepare_ke_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    num_classes = 1000
    net = prepare_model(
        model_name=args.model,
        classes=num_classes,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip())

    train_data, val_data = get_data_rec(
        rec_train=args.rec_train,
        rec_train_idx=args.rec_train_idx,
        rec_val=args.rec_val,
        rec_val_idx=args.rec_val_idx,
        batch_size=batch_size,
        num_workers=args.num_workers)
    train_gen = get_data_generator(
        data_iterator=train_data,
        num_classes=num_classes)
    val_gen = get_data_generator(
        data_iterator=val_data,
        num_classes=num_classes)

    # num_training_samples = 1281167
    # prepare_trainer(
    #     net=net,
    #     optimizer_name=args.optimizer_name,
    #     wd=args.wd,
    #     momentum=args.momentum,
    #     lr=args.lr,
    #     batch_size=batch_size,
    #     num_epochs=args.num_epochs,
    #     dtype=args.dtype,
    #     state_file_path=args.resume_state)
    #
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
    #         checkpoint_file_exts=('.params', '.states'),
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
    #
    # train_net(
    #     batch_size=batch_size,
    #     num_epochs=args.num_epochs,
    #     start_epoch1=args.start_epoch,
    #     train_data=train_data,
    #     val_data=val_data,
    #     batch_fn=batch_fn,
    #     use_rec=args.use_rec,
    #     dtype=args.dtype,
    #     net=net,
    #     trainer=trainer,
    #     lr_scheduler=lr_scheduler,
    #     lp_saver=lp_saver,
    #     log_interval=args.log_interval,
    #     mixup=args.mixup,
    #     mixup_epoch_tail=args.mixup_epoch_tail,
    #     num_classes=num_classes,
    #     ctx=ctx)


if __name__ == '__main__':
    main()
