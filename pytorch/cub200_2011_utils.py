"""
    CUB-200-2011 fine-grained classification dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_train_data_loader', 'get_val_data_loader']

import math
import torch.utils.data
import torchvision.transforms as transforms
from pytorch.datasets.cub200_2011_cls_dataset import CUB200_2011


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/CUB_200_2011',
        help='path to directory with CUB-200-2011 dataset')
    parser.add_argument(
        '--input-size',
        type=int,
        default=448,
        help='size of the input for model')
    parser.add_argument(
        '--resize-inv-factor',
        type=float,
        default=0.74667,
        help='inverted ratio for input image crop')

    parser.add_argument(
        '--num-classes',
        type=int,
        default=200,
        help='number of classes')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')


def get_train_data_loader(dataset_dir,
                          batch_size,
                          num_workers,
                          input_image_size=448):
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)
    jitter_param = 0.4

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)])

    dataset = CUB200_2011(
        root=dataset_dir,
        train=True,
        transform=transform_train)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader


def get_val_data_loader(dataset_dir,
                        batch_size,
                        num_workers,
                        input_image_size=448,
                        resize_inv_factor=0.74667):
    assert (resize_inv_factor > 0.0)
    resize_value = int(math.ceil(float(input_image_size) / resize_inv_factor))

    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform_val = transforms.Compose([
        transforms.Resize(resize_value),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])

    dataset = CUB200_2011(
        root=dataset_dir,
        train=False,
        transform=transform_val)

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return val_loader
