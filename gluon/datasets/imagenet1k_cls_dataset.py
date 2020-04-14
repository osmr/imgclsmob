"""
    ImageNet-1K classification dataset.
"""

import os
import math
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data.vision import transforms
from .dataset_metainfo import DatasetMetaInfo


class ImageNet1K(ImageFolderDataset):
    """
    ImageNet-1K classification dataset.

    Refer to MXNet documentation for the description of this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "imagenet"),
                 mode="train",
                 transform=None):
        split = "train" if mode == "train" else "val"
        root = os.path.join(root, split)
        super(ImageNet1K, self).__init__(root=root, flag=1, transform=transform)


class ImageNet1KMetaInfo(DatasetMetaInfo):
    """
    Descriptor of ImageNet-1K dataset.
    """

    def __init__(self):
        super(ImageNet1KMetaInfo, self).__init__()
        self.label = "ImageNet1K"
        self.short_label = "imagenet"
        self.root_dir_name = "imagenet"
        self.dataset_class = ImageNet1K
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = 1000
        self.input_image_size = (224, 224)
        self.resize_inv_factor = 0.875
        self.aug_type = "aug0"
        self.train_metric_capts = ["Train.Top1"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err-top1"}]
        self.val_metric_capts = ["Val.Top1", "Val.Top5"]
        self.val_metric_names = ["Top1Error", "TopKError"]
        self.val_metric_extra_kwargs = [{"name": "err-top1"}, {"name": "err-top5", "top_k": 5}]
        self.saver_acc_ind = 1
        self.train_transform = imagenet_train_transform
        self.val_transform = imagenet_val_transform
        self.test_transform = imagenet_val_transform
        self.ml_type = "imgcls"
        self.mean_rgb = (0.485, 0.456, 0.406)
        self.std_rgb = (0.229, 0.224, 0.225)
        self.interpolation = 1
        self.loss_name = "SoftmaxCrossEntropy"

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        """
        Create python script parameters (for ImageNet-1K dataset metainfo).

        Parameters:
        ----------
        parser : ArgumentParser
            ArgumentParser instance.
        work_dir_path : str
            Path to working directory.
        """
        super(ImageNet1KMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--input-size",
            type=int,
            default=self.input_image_size[0],
            help="size of the input for model")
        parser.add_argument(
            "--resize-inv-factor",
            type=float,
            default=self.resize_inv_factor,
            help="inverted ratio for input image crop")
        parser.add_argument(
            "--aug-type",
            type=str,
            default="aug0",
            help="augmentation type. options are aug0, aug1, aug2")
        parser.add_argument(
            "--mean-rgb",
            nargs=3,
            type=float,
            default=self.mean_rgb,
            help="Mean of RGB channels in the dataset")
        parser.add_argument(
            "--std-rgb",
            nargs=3,
            type=float,
            default=self.std_rgb,
            help="STD of RGB channels in the dataset")
        parser.add_argument(
            "--interpolation",
            type=int,
            default=self.interpolation,
            help="Preprocessing interpolation")

    def update(self,
               args):
        """
        Update ImageNet-1K dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        super(ImageNet1KMetaInfo, self).update(args)
        self.input_image_size = (args.input_size, args.input_size)
        self.resize_inv_factor = args.resize_inv_factor
        self.aug_type = args.aug_type
        self.mean_rgb = args.mean_rgb
        self.std_rgb = args.std_rgb
        self.interpolation = args.interpolation


class ImgAugTransform(HybridBlock):
    """
    ImgAug-like transform (geometric, noise, and blur).
    """
    def __init__(self):
        super(ImgAugTransform, self).__init__()
        from imgaug import augmenters as iaa
        from imgaug import parameters as iap
        self.seq = iaa.Sequential(
            children=[
                iaa.Sequential(
                    children=[
                        iaa.Sequential(
                            children=[
                                iaa.OneOf(
                                    children=[
                                        iaa.Sometimes(
                                            p=0.95,
                                            then_list=iaa.Affine(
                                                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                                rotate=(-30, 30),
                                                shear=(-15, 15),
                                                order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                                mode="reflect",
                                                name="Affine")),
                                        iaa.Sometimes(
                                            p=0.05,
                                            then_list=iaa.PerspectiveTransform(
                                                scale=(0.01, 0.1)))],
                                    name="Blur"),
                                iaa.Sometimes(
                                    p=0.01,
                                    then_list=iaa.PiecewiseAffine(
                                        scale=(0.0, 0.01),
                                        nb_rows=(4, 20),
                                        nb_cols=(4, 20),
                                        order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                        mode="reflect",
                                        name="PiecewiseAffine"))],
                            random_order=True,
                            name="GeomTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.75,
                                    then_list=iaa.Add(
                                        value=(-10, 10),
                                        per_channel=0.5,
                                        name="Brightness")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.Emboss(
                                        alpha=(0.0, 0.5),
                                        strength=(0.5, 1.2),
                                        name="Emboss")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.Sharpen(
                                        alpha=(0.0, 0.5),
                                        lightness=(0.5, 1.2),
                                        name="Sharpen")),
                                iaa.Sometimes(
                                    p=0.25,
                                    then_list=iaa.ContrastNormalization(
                                        alpha=(0.5, 1.5),
                                        per_channel=0.5,
                                        name="ContrastNormalization"))
                            ],
                            random_order=True,
                            name="ColorTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.AdditiveGaussianNoise(
                                        loc=0,
                                        scale=(0.0, 10.0),
                                        per_channel=0.5,
                                        name="AdditiveGaussianNoise")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.SaltAndPepper(
                                        p=(0, 0.001),
                                        per_channel=0.5,
                                        name="SaltAndPepper"))],
                            random_order=True,
                            name="Noise"),
                        iaa.OneOf(
                            children=[
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.MedianBlur(
                                        k=3,
                                        name="MedianBlur")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.AverageBlur(
                                        k=(2, 4),
                                        name="AverageBlur")),
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.GaussianBlur(
                                        sigma=(0.0, 2.0),
                                        name="GaussianBlur"))],
                            name="Blur"),
                    ],
                    random_order=True,
                    name="MainProcess")])

    def hybrid_forward(self, F, x):
        img = x.asnumpy().copy()
        # cv2.imshow(winname="imgA", mat=img)
        img_aug = self.seq.augment_image(img)
        # cv2.imshow(winname="img_augA", mat=img_aug)
        # cv2.waitKey()
        x = mx.nd.array(img_aug, dtype=x.dtype, ctx=x.context)
        return x


def imagenet_train_transform(ds_metainfo,
                             jitter_param=0.4,
                             lighting_param=0.1):
    """
    Create image transform sequence for training subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    jitter_param : float
        How much to jitter values.
    lighting_param : float
        How much to noise intensity of the image.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    input_image_size = ds_metainfo.input_image_size
    if ds_metainfo.aug_type == "aug0":
        interpolation = ds_metainfo.interpolation
        transform_list = []
    elif ds_metainfo.aug_type == "aug1":
        interpolation = 10
        transform_list = []
    elif ds_metainfo.aug_type == "aug2":
        interpolation = 10
        transform_list = [
            ImgAugTransform()
        ]
    else:
        raise RuntimeError("Unknown augmentation type: {}\n".format(ds_metainfo.aug_type))

    transform_list += [
        transforms.RandomResizedCrop(
            size=input_image_size,
            interpolation=interpolation),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ds_metainfo.mean_rgb,
            std=ds_metainfo.std_rgb)
    ]

    return transforms.Compose(transform_list)


def imagenet_val_transform(ds_metainfo):
    """
    Create image transform sequence for validation subset.

    Parameters:
    ----------
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    input_image_size = ds_metainfo.input_image_size
    resize_value = calc_val_resize_value(
        input_image_size=ds_metainfo.input_image_size,
        resize_inv_factor=ds_metainfo.resize_inv_factor)
    return transforms.Compose([
        transforms.Resize(
            size=resize_value,
            keep_ratio=True,
            interpolation=ds_metainfo.interpolation),
        transforms.CenterCrop(size=input_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=ds_metainfo.mean_rgb,
            std=ds_metainfo.std_rgb)
    ])


def calc_val_resize_value(input_image_size=(224, 224),
                          resize_inv_factor=0.875):
    """
    Calculate image resize value for validation subset.

    Parameters:
    ----------
    input_image_size : tuple of 2 int
        Main script arguments.
    resize_inv_factor : float
        Resize inverted factor.

    Returns
    -------
    int
        Resize value.
    """
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))
    return resize_value
