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
        default='../imgclsmob_data/imagenet',
        help='path to directory with ImageNet-1K dataset')

    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='size of the input for model')
    parser.add_argument(
        '--resize-inv-factor',
        type=float,
        default=0.875,
        help='inverted ratio for input image crop')

    parser.add_argument(
        '--num-classes',
        type=int,
        default=1000,
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
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)
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

    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

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
