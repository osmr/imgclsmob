import math
import os

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

__all__ = ['add_dataset_parser_arguments', 'get_train_data_loader', 'get_val_data_loader']


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/cifar10',
        help='path to directory with CIFAR-10 dataset')

    parser.add_argument(
        '--num-classes',
        type=int,
        default=10,
        help='number of classes')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')


def get_train_data_loader(data_dir,
                          batch_size,
                          num_workers,
                          input_image_size=224):
    mean_rgb = (0.4914, 0.4822, 0.4465)
    std_rgb = (0.2023, 0.1994, 0.2010)
    jitter_param = 0.4

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(input_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=jitter_param,
                    contrast=jitter_param,
                    saturation=jitter_param),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean_rgb,
                    std=std_rgb),
            ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader


def get_val_data_loader(data_dir,
                        batch_size,
                        num_workers,
                        input_image_size=224,
                        resize_inv_factor=0.875):
    assert (resize_inv_factor > 0.0)
    resize_value = int(math.ceil(float(input_image_size) / resize_inv_factor))

    mean_rgb = (0.4914, 0.4822, 0.4465)
    std_rgb = (0.2023, 0.1994, 0.2010)

    val_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'),
            transform=transforms.Compose([
                transforms.Resize(resize_value),
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean_rgb,
                    std=std_rgb),
            ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return val_loader
