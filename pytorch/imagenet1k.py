import math
import os
import cv2
import numpy as np
from PIL import Image

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

    parser.add_argument(
        '--use-cv-resize',
        action='store_true',
        help='use OpenCV resize preprocessing')


def cv_loader(path):
    img = cv2.imread(path, flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class CvResize(object):
    """
    Resize the input PIL Image to the given size via OpenCV.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    interpolation : int, default PIL.Image.BILINEAR
        Interpolation method for resizing. By default uses bilinear
        interpolation.
    """
    def __init__(self,
                 size,
                 interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Resize image.

        Parameters
        ----------
        img : PIL.Image
            input image.

        Returns
        -------
        PIL.Image
            Resulted image.
        """
        if self.interpolation == Image.NEAREST:
            cv_interpolation = cv2.INTER_NEAREST
        elif self.interpolation == Image.BILINEAR:
            cv_interpolation = cv2.INTER_LINEAR
        elif self.interpolation == Image.BICUBIC:
            cv_interpolation = cv2.INTER_CUBIC
        elif self.interpolation == Image.LANCZOS:
            cv_interpolation = cv2.INTER_LANCZOS4
        else:
            raise ValueError()

        cv_img = np.array(img)

        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                out_size = (self.size, int(self.size * h / w))
            else:
                out_size = (int(self.size * w / h), self.size)
            cv_img = cv2.resize(cv_img, dsize=out_size, interpolation=cv_interpolation)
            return Image.fromarray(cv_img)
        else:
            cv_img = cv2.resize(cv_img, dsize=self.size, interpolation=cv_interpolation)
            return Image.fromarray(cv_img)


def get_train_data_loader(data_dir,
                          batch_size,
                          num_workers,
                          input_image_size=224):
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

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=transform_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader


def get_val_data_loader(data_dir,
                        batch_size,
                        num_workers,
                        input_image_size=224,
                        resize_inv_factor=0.875,
                        use_cv_resize=False):
    assert (resize_inv_factor > 0.0)
    resize_value = int(math.ceil(float(input_image_size) / resize_inv_factor))

    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform_test = transforms.Compose([
        CvResize(resize_value) if use_cv_resize else transforms.Resize(resize_value),
        transforms.CenterCrop(input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)])

    dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=transform_test)

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return val_loader
