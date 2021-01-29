"""
    FPENet for image segmentation, implemented in PyTorch.
    Original paper: 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.
"""

__all__ = ['FPENet', 'fpenet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, SEBlock, InterpolationBlock, MultiOutputSequential


class FPEBlock(nn.Module):
    """
    FPENet block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    """
    def __init__(self,
                 channels):
        super(FPEBlock, self).__init__()
        dilations = [1, 2, 4, 8]
        assert (channels % len(dilations) == 0)
        mid_channels = channels // len(dilations)

        self.blocks = nn.Sequential()
        for i, dilation in enumerate(dilations):
            self.blocks.add_module("block{}".format(i + 1), conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                groups=mid_channels,
                dilation=dilation,
                padding=dilation))

    def forward(self, x):
        xs = torch.chunk(x, chunks=len(self.blocks._modules), dim=1)
        ys = []
        for bi, xsi in zip(self.blocks._modules.values(), xs):
            if len(ys) == 0:
                ys.append(bi(xsi))
            else:
                ys.append(bi(xsi + ys[-1]))
        x = torch.cat(ys, dim=1)
        return x


class FPEUnit(nn.Module):
    """
    FPENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int
        Bottleneck factor.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_factor,
                 use_se):
        super(FPEUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        self.use_se = use_se
        mid1_channels = in_channels * bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid1_channels,
            stride=stride)
        self.blocks = FPEBlock(channels=mid1_channels)
        self.conv2 = conv1x1_block(
            in_channels=mid1_channels,
            out_channels=out_channels,
            activation=None)
        if self.use_se:
            self.se = SEBlock(channels=out_channels)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class FPEStage(nn.Module):
    """
    FPENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 use_se):
        super(FPEStage, self).__init__()
        self.use_block = (layers > 1)
        if self.use_block:
            self.down = FPEUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                bottleneck_factor=4,
                use_se=use_se)
            self.blocks = nn.Sequential()
            for i in range(layers - 1):
                self.blocks.add_module("block{}".format(i + 1), FPEUnit(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    bottleneck_factor=1,
                    use_se=use_se))
        else:
            self.down = FPEUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                bottleneck_factor=1,
                use_se=use_se)

    def forward(self, x):
        x = self.down(x)
        if self.use_block:
            y = self.blocks(x)
            x = x + y
        return x


class MEUBlock(nn.Module):
    """
    FPENet specific mutual embedding upsample (MEU) block.

    Parameters:
    ----------
    in_channels_high : int
        Number of input channels for x_high.
    in_channels_low : int
        Number of input channels for x_low.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels_high,
                 in_channels_low,
                 out_channels):
        super(MEUBlock, self).__init__()
        self.conv_high = conv1x1_block(
            in_channels=in_channels_high,
            out_channels=out_channels,
            activation=None)
        self.conv_low = conv1x1_block(
            in_channels=in_channels_low,
            out_channels=out_channels,
            activation=None)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_w_high = conv1x1(
            in_channels=out_channels,
            out_channels=out_channels)
        self.conv_w_low = conv1x1(
            in_channels=1,
            out_channels=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.up = InterpolationBlock(
            scale_factor=2,
            align_corners=True)

    def forward(self, x_high, x_low):
        x_high = self.conv_high(x_high)
        x_low = self.conv_low(x_low)

        w_high = self.pool(x_high)
        w_high = self.conv_w_high(w_high)
        w_high = self.relu(w_high)
        w_high = self.sigmoid(w_high)

        w_low = torch.mean(x_low, dim=1, keepdim=True)
        w_low = self.conv_w_low(w_low)
        w_low = self.sigmoid(w_low)

        x_high = self.up(x_high)

        x_high = x_high * w_low
        x_low = x_low * w_high

        out = x_high + x_low
        return out


class FPENet(nn.Module):
    """
    FPENet model from 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    meu_channels : list of int
        Number of output channels for MEU blocks.
    use_se : bool
        Whether to use SE-module.
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
                 layers,
                 channels,
                 init_block_channels,
                 meu_channels,
                 use_se,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(FPENet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        self.stem = conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2)
        in_channels = init_block_channels

        self.encoder = MultiOutputSequential(return_last=False)
        for i, (layers_i, out_channels) in enumerate(zip(layers, channels)):
            stage = FPEStage(
                in_channels=in_channels,
                out_channels=out_channels,
                layers=layers_i,
                use_se=use_se)
            stage.do_output = True
            self.encoder.add_module("stage{}".format(i + 1), stage)
            in_channels = out_channels

        self.meu1 = MEUBlock(
            in_channels_high=channels[-1],
            in_channels_low=channels[-2],
            out_channels=meu_channels[0])
        self.meu2 = MEUBlock(
            in_channels_high=meu_channels[0],
            in_channels_low=channels[-3],
            out_channels=meu_channels[1])
        in_channels = meu_channels[1]

        self.classifier = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

        self.up = InterpolationBlock(
            scale_factor=2,
            align_corners=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        y = self.encoder(x)
        x = self.meu1(y[2], y[1])
        x = self.meu2(x, y[0])
        x = self.classifier(x)
        x = self.up(x)
        return x


def get_fpenet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create FPENet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    width = 16
    channels = [int(width * (2 ** i)) for i in range(3)]
    init_block_channels = width
    layers = [1, 3, 9]
    meu_channels = [64, 32]
    use_se = False

    net = FPENet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        meu_channels=meu_channels,
        use_se=use_se,
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


def fpenet_cityscapes(num_classes=19, **kwargs):
    """
    FPENet model for Cityscapes from 'Feature Pyramid Encoding Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1909.08599.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_fpenet(num_classes=num_classes, model_name="fpenet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False

    in_size = (1024, 2048)

    models = [
        fpenet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fpenet_cityscapes or weight_count == 115125)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
