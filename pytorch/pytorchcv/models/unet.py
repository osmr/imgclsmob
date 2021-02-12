"""
    U-Net for image segmentation, implemented in PyTorch.
    Original paper: 'U-Net: Convolutional Networks for Biomedical Image Segmentation,'
    https://arxiv.org/abs/1505.04597.
"""

__all__ = ['UNet', 'unet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv3x3_block, InterpolationBlock, Hourglass, Identity


class UNetBlock(nn.Module):
    """
    U-Net specific base block (double convolution).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDownStage(nn.Module):
    """
    U-Net specific downscale (encoder) stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetDownStage, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = UNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UNetUpStage(nn.Module):
    """
    U-Net specific upscale (decoder) stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetUpStage, self).__init__()
        self.conv = UNetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)
        self.up = InterpolationBlock(
            scale_factor=2,
            align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class UNetHead(nn.Module):
    """
    U-Net specific head.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(UNetHead, self).__init__()
        mid_channels = in_channels // 2
        self.conv1 = UNetBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    """
    U-Net model from 'U-Net: Convolutional Networks for Biomedical Image Segmentation,'
    https://arxiv.org/abs/1505.04597.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each stage in encoder and decoder.
    init_block_channels : int
        Number of output channels for the initial unit.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    num_classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(UNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        bias = True

        self.stem = UNetBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bias=bias)
        in_channels = init_block_channels

        down_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[0]):
            down_seq.add_module("down{}".format(i + 1), UNetDownStage(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias))
            in_channels = out_channels
            skip_seq.add_module("skip{}".format(i + 1), Identity())

        up_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[1]):
            if i == 0:
                up_seq.add_module("down{}".format(i + 1), InterpolationBlock(
                    scale_factor=2,
                    align_corners=True))
            else:
                up_seq.add_module("down{}".format(i + 1), UNetUpStage(
                    in_channels=(2 * in_channels),
                    out_channels=out_channels,
                    bias=bias))
            in_channels = out_channels
        up_seq = up_seq[::-1]

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            merge_type="cat")

        self.head = UNetHead(
            in_channels=(2 * in_channels),
            out_channels=num_classes,
            bias=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.hg(x)
        x = self.head(x)
        return x


def get_unet(model_name=None,
             pretrained=False,
             root=os.path.join("~", ".torch", "models"),
             **kwargs):
    """
    Create U-Net model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [[128, 256, 512, 512], [512, 256, 128, 64]]
    init_block_channels = 64

    net = UNet(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def unet_cityscapes(num_classes=19, **kwargs):
    """
    U-Net model for Cityscapes from 'U-Net: Convolutional Networks for Biomedical Image Segmentation,'
    https://arxiv.org/abs/1505.04597.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_unet(num_classes=num_classes, model_name="unet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        unet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != unet_cityscapes or weight_count == 13396499)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
