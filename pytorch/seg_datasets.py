"""
    Segmentation datasets (ADE20K/PascalVOC/COCO/Cityscapes) routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_val_data_loader', 'validate1']

from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


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
    elif dataset_name == "VOC":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/voc',
            help='path to directory with Pascal VOC dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=21,
            help='number of classes')
    elif dataset_name == "COCO":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/coco',
            help='path to directory with COCO dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=21,
            help='number of classes')
    elif dataset_name == "Cityscapes":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/cityscapes',
            help='path to directory with Cityscapes dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=19,
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


def get_val_data_loader(dataset_name,
                        dataset_dir,
                        batch_size,
                        num_workers):
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb),
    ])

    if dataset_name == "ADE20K":
        dataset = None
    elif dataset_name == "VOC":
        dataset = datasets.VOCSegmentation(
            root=dataset_dir,
            image_set="val",
            download=True,
            transform=transform_val)
    elif dataset_name == "COCO":
        dataset = None
    elif dataset_name == "Cityscapes":
        dataset = None
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return val_loader


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
    for batch in tqdm(val_data):
        data_list, labels_list = batch_fn(batch, ctx)
        outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
        accuracy_metric.update(labels_list, outputs_list)
    pix_acc, miou = accuracy_metric.get()
    return pix_acc, miou
