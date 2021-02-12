"""
    LinkNet for image segmentation, implemented in PyTorch.
    Original paper: 'LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation,'
    https://arxiv.org/abs/1707.03718.
"""

__all__ = ['LinkNet', 'linknet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1_block, conv3x3_block, deconv3x3_block, Hourglass, Identity
from .resnet import resnet18


class DecoderStage(nn.Module):
    """
    LinkNet specific decoder stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the deconvolution.
    out_padding : int or tuple/list of 2 int
        Output padding value for deconvolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 output_padding,
                 bias):
        super(DecoderStage, self).__init__()
        mid_channels = in_channels // 4

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias)
        self.conv2 = deconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            out_padding=output_padding,
            bias=bias)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LinkNetHead(nn.Module):
    """
    LinkNet head block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(LinkNetHead, self).__init__()
        mid_channels = in_channels // 2

        self.conv1 = deconv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2,
            padding=1,
            out_padding=1,
            bias=True)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            bias=True)
        self.conv3 = nn.ConvTranspose2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LinkNet(nn.Module):
    """
    LinkNet model from 'LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation,'
    https://arxiv.org/abs/1707.03718.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels form feature extractor.
    channels : list of int
        Number of output channels for the first unit of each stage.
    dilations : list of list of int
        Dilation values for each unit.
    dropout_rates : list of float
        Parameter of dropout layer for each stage.
    downs : list of int
        Whether to downscale or upscale in each stage.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images in encoder.
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
                 backbone,
                 backbone_out_channels,
                 channels,
                 strides,
                 output_paddings,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(LinkNet, self).__init__()
        assert (in_channels == 3)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        bias = False

        self.stem = backbone.init_block
        in_channels = backbone_out_channels

        down_seq = nn.Sequential()
        down_seq.add_module("down1", backbone.stage1)
        down_seq.add_module("down2", backbone.stage2)
        down_seq.add_module("down3", backbone.stage3)
        down_seq.add_module("down4", backbone.stage4)

        up_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i, out_channels in enumerate(channels):
            up_seq.add_module("up{}".format(i + 1), DecoderStage(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides[i],
                output_padding=output_paddings[i],
                bias=bias))
            in_channels = out_channels
            skip_seq.add_module("skip{}".format(i + 1), Identity())
        up_seq = up_seq[::-1]

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq)

        self.head = LinkNetHead(
            in_channels=in_channels,
            out_channels=num_classes)

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


def get_linknet(backbone,
                backbone_out_channels,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create LinkNet model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels form feature extractor.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [256, 128, 64, 64]
    strides = [2, 2, 2, 1]
    output_paddings = [1, 1, 1, 0]

    net = LinkNet(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
        strides=strides,
        output_paddings=output_paddings,
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


def linknet_cityscapes(pretrained_backbone=False, num_classes=19, **kwargs):
    """
    LinkNet model for Cityscapes from 'LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation,'
    https://arxiv.org/abs/1707.03718.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone[-1]
    backbone_out_channels = 512
    return get_linknet(backbone=backbone, backbone_out_channels=backbone_out_channels, num_classes=num_classes,
                       model_name="linknet_cityscapes", **kwargs)


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
        linknet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != linknet_cityscapes or weight_count == 11535699)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
