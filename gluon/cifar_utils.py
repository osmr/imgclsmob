"""
    CIFAR/SVHN dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'get_train_data_source', 'get_val_data_source',
           'get_num_training_samples']

import os
from mxnet import gluon
from gluon.datasets.cifar10_cls_dataset import CIFAR10MetaInfo, cifar10_train_transform, cifar10_val_transform
from gluon.datasets.cifar100_cls_dataset import CIFAR100MetaInfo
from gluon.datasets.svhn_cls_dataset import SVHNMetaInfo


def get_dataset_metainfo(dataset_name):
    if dataset_name == "CIFAR10":
        return CIFAR10MetaInfo
    elif dataset_name == "CIFAR100":
        return CIFAR100MetaInfo
    elif dataset_name == "SVHN":
        return SVHNMetaInfo
    else:
        raise Exception("Unrecognized dataset: {}".format(dataset_name))


def add_dataset_parser_arguments(parser,
                                 dataset_name):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data", dataset_metainfo.root_dir_name),
        help="path to directory with {} dataset".format(dataset_metainfo.label))
    parser.add_argument(
        "--num-classes",
        type=int,
        default=dataset_metainfo.num_classes,
        help="number of classes")
    parser.add_argument(
        '--in-channels',
        type=int,
        default=dataset_metainfo.in_channels,
        help='number of input channels')


def batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label


def get_num_training_samples(dataset_name):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
    return dataset_metainfo.num_training_samples


def get_train_data_source(dataset_name,
                          dataset_dir,
                          batch_size,
                          num_workers):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
    dataset_class = dataset_metainfo.dataset_class
    transform_train = cifar10_train_transform()
    dataset = dataset_class(
        root=dataset_dir,
        train=True).transform_first(fn=transform_train)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        last_batch="discard",
        num_workers=num_workers)


def get_val_data_source(dataset_name,
                        dataset_dir,
                        batch_size,
                        num_workers):
    dataset_metainfo = get_dataset_metainfo(dataset_name)
    dataset_class = dataset_metainfo.dataset_class
    transform_val = cifar10_val_transform()
    dataset = dataset_class(
        root=dataset_dir,
        train=False).transform_first(fn=transform_val)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)
