"""
    ContextNet for image segmentation, implemented in PyTorch.
    Original paper: 'ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time,'
    https://arxiv.org/abs/1805.04554.
"""

__all__ = ['ContextNet', 'ctxnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwsconv3x3_block, InterpolationBlock


class CtxShallowNet(nn.Module):
    """
    ContextNet specific shallow net (spatial detail encoder).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid1_channels : int
        Number of middle #1 channels.
    mid2_channels : int
        Number of middle #2 channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 mid1_channels,
                 mid2_channels,
                 out_channels):
        super(CtxShallowNet, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid1_channels,
            stride=2,
            padding=0)
        self.conv2 = dwsconv3x3_block(
            in_channels=mid1_channels,
            out_channels=mid2_channels,
            stride=2)
        self.conv3 = dwsconv3x3_block(
            in_channels=mid2_channels,
            out_channels=out_channels,
            stride=2)
        self.conv4 = dwsconv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class LinearBottleneck(nn.Module):
    """
    So-called 'Linear Bottleneck' layer (from MobileNetV2). It is used as a CtxDeepNet encoder unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    expansion : bool
        Whether do expansion of channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion):
        super(LinearBottleneck, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels

        self.block = nn.Sequential(
            conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels),
            dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride),
            conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None),
        )

    def forward(self, x):
        if self.residual:
            identity = x

        x = self.block(x)

        if self.residual:
            x = x + identity
        return x


class CtxDeepNet(nn.Module):
    """
    ContextNet specific deep net (regular encoder).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    init_block_channels : int
        Number of channels for init block.
    """
    def __init__(self,
                 in_channels,
                 init_block_channels):
        super(CtxDeepNet, self).__init__()
        layers = [1, 1, 3, 3, 2, 2]
        channels_per_layers = [32, 32, 48, 64, 96, 128]
        downsample = [0, 0, 1, 1, 0, 0]

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            padding=0))
        in_channels = init_block_channels
        for i, out_channels in enumerate(channels_per_layers):
            stage = nn.Sequential()
            expansion = (i != 0)
            for j in range(layers[i]):
                stride = 2 if (j == 0) and (downsample[i] == 1) else 1
                stage.add_module("unit{}".format(j + 1), LinearBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

    def forward(self, x):
        x = self.features(x)
        return x


class FeatureFusion(nn.Module):
    """
    ContextNet specific feature fusion block.

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
        super(FeatureFusion, self).__init__()
        self.conv_high = conv1x1_block(
            in_channels=in_channels_high,
            out_channels=out_channels,
            bias=True,
            activation=None)
        self.up = InterpolationBlock(
            scale_factor=4,
            align_corners=True)
        self.dw_conv_low = dwconv3x3_block(
            in_channels=in_channels_low,
            out_channels=out_channels)
        self.pw_conv_low = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bias=True,
            activation=None)
        self.activ = nn.ReLU(True)

    def forward(self, x_high, x_low):
        x_high = self.conv_high(x_high)

        x_low = self.up(x_low)
        x_low = self.dw_conv_low(x_low)
        x_low = self.pw_conv_low(x_low)

        out = x_high + x_low
        out = self.activ(out)
        return out


class CtxHead(nn.Module):
    """
    ContextNet specific head/classifier block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of output channels/classes.
    """
    def __init__(self,
                 in_channels,
                 num_classes):
        super(CtxHead, self).__init__()
        self.conv1 = dwsconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels)
        self.conv2 = dwsconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv3 = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class CtxAuxHead(nn.Module):
    """
    ContextNet specific auxiliary head/classifier block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of output channels/classes.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 num_classes):
        super(CtxAuxHead, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=num_classes,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class ContextNet(nn.Module):
    """
    ContextNet model from 'ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time,'
    https://arxiv.org/abs/1805.04554.

    Parameters:
    ----------
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
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(ContextNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.aux = aux
        self.fixed_size = fixed_size

        self.features_high = CtxShallowNet(
            in_channels=in_channels,
            mid1_channels=32,
            mid2_channels=64,
            out_channels=128)
        self.down = InterpolationBlock(
            scale_factor=4,
            align_corners=True,
            up=False)
        self.features_low = CtxDeepNet(
            in_channels=in_channels,
            init_block_channels=32)
        self.fusion = FeatureFusion(
            in_channels_high=128,
            in_channels_low=128,
            out_channels=128)
        self.head = CtxHead(
            in_channels=128,
            num_classes=num_classes)
        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=True)

        if self.aux:
            self.aux_head = CtxAuxHead(
                in_channels=128,
                mid_channels=32,
                num_classes=num_classes)

    def forward(self, x):
        x_high = self.features_high(x)

        x_low = self.down(x)
        x_low = self.features_low(x_low)

        x = self.fusion(x_high, x_low)
        x = self.head(x)
        x = self.up(x)

        if self.aux:
            y = self.aux_head(x_high)
            y = self.up(y)
            return x, y
        else:
            return x


def get_ctxnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create ContextNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = ContextNet(
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


def ctxnet_cityscapes(num_classes=19, **kwargs):
    """
    ContextNet model for Cityscapes from 'ContextNet: Exploring Context and Detail for Semantic Segmentation in
    Real-time,' https://arxiv.org/abs/1805.04554.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ctxnet(num_classes=num_classes, model_name="ctxnet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    aux = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        ctxnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != ctxnet_cityscapes or weight_count == 914118)
        else:
            assert (model != ctxnet_cityscapes or weight_count == 876563)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        ys = net(x)
        y = ys[0] if aux else ys
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))
        if aux:
            assert (tuple(ys[1].size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
