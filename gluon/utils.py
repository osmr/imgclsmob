import logging
import os
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data.vision import transforms

from .gluoncv2.model_provider import get_model


class ImageNet(ImageFolderDataset):
    """
    Load the ImageNet classification dataset.

    Refer to :doc:`../build/examples_datasets/imagenet` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets', 'imagenet'),
                 train=True,
                 transform=None):
        split = 'train' if train else 'val'
        root = os.path.join(root, split)
        super(ImageNet, self).__init__(root=root, flag=1, transform=transform)


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
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=num_workers,
        shuffle=True,
        batch_size=batch_size,

        data_shape=(3, 224, 224),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=(4. / 3.),
        min_aspect_ratio=(3. / 4.),
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,

        resize=256,
        data_shape=(3, 224, 224),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return train_data, val_data, batch_fn


def get_data_loader(data_dir,
                    batch_size,
                    num_workers):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
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
        ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_data, val_data, batch_fn


def prepare_model(model_name,
                  classes,
                  use_pretrained,
                  pretrained_model_file_path,
                  dtype,
                  tune_layers,
                  ctx):
    kwargs = {'ctx': ctx,
              'pretrained': use_pretrained,
              'classes': classes}

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        net.load_parameters(
            filename=pretrained_model_file_path,
            ctx=ctx)

    net.cast(dtype)

    net.hybridize(
        static_alloc=True,
        static_shape=True)

    if pretrained_model_file_path or use_pretrained:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    else:
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    if tune_layers:
        tune_layers_ptrn = tuple(tune_layers.split(','))
        params = net._collect_params_with_prefix()
        param_keys = list(params.keys())
        for key in param_keys:
            if not key.startswith(tune_layers_ptrn):
                params[key].grad_req = 'null'
            else:
                logging.info('Fine-tune parameter: {}'.format(key))
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    return net


def calc_net_weight_count(net):
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def validate(acc_top1,
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
        data_list, labels_list = batch_fn(batch, ctx)
        outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
        acc_top1.update(labels_list, outputs_list)
        acc_top5.update(labels_list, outputs_list)
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return 1.0 - top1, 1.0 - top5
