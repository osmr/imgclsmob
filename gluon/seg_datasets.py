"""
    Segmentation datasets (ADE20K) routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'get_train_data_source', 'get_val_data_source',
           'get_num_training_samples', 'validate1']

from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data.ade20k.segmentation import ADE20KSegmentation


def add_dataset_parser_arguments(parser,
                                 dataset_name):
    if dataset_name == "ADE20K":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/ade20k',
            help='path to directory with ADE20K dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=150,
            help='number of classes')
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')
    parser.add_argument(
        '--image-base-size',
        type=int,
        default=520,
        help='base image size')
    parser.add_argument(
        '--image-crop-size',
        type=int,
        default=480,
        help='crop image size')


def batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label


def get_num_training_samples(dataset_name):
    if dataset_name == "CIFAR10":
        return 50000
    elif dataset_name == "CIFAR100":
        return 50000
    elif dataset_name == "SVHN":
        return 73257
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))


def get_train_data_source(dataset_name,
                          dataset_dir,
                          batch_size,
                          num_workers):
    jitter_param = 0.4
    lighting_param = 0.1
    mean_rgb = (0.4914, 0.4822, 0.4465)
    std_rgb = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])

    if dataset_name == "ADE20K":
        dataset_class = ADE20KSegmentation
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    dataset = dataset_class(
        root=dataset_dir,
        train=True)

    return gluon.data.DataLoader(
        dataset=dataset.transform_first(fn=transform_train),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)


def get_val_data_source(dataset_name,
                        dataset_dir,
                        batch_size,
                        num_workers,
                        image_base_size,
                        image_crop_size):
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])

    if dataset_name == "ADE20K":
        dataset_class = ADE20KSegmentation
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    dataset = dataset_class(
        root=dataset_dir,
        split="val",
        mode="val",
        base_size=image_base_size,
        crop_size=image_crop_size,
        transform=transform_val)

    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)


def validate1(accuracy_metric,
              net,
              val_data,
              batch_fn,
              data_source_needs_reset,
              dtype,
              ctx):
    if data_source_needs_reset:
        val_data.reset()
    accuracy_metric.reset()
    for batch in val_data:
        data_list, labels_list = batch_fn(batch, ctx)
        outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
        accuracy_metric.update(labels_list, outputs_list)
    pix_acc, miou = accuracy_metric.get()
    return pix_acc, miou
