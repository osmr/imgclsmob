"""
    DABNet for image segmentation, implemented in PyTorch.
    Original paper: 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.
"""

__all__ = ['DABNet', 'dabnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv3x3, conv3x3_block, ConvBlock, NormActivation, Concurrent, InterpolationBlock


class DwaConvBlock(nn.Module):
    """
    Depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(DwaConvBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=channels,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=activation)
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, padding),
            dilation=(1, dilation),
            groups=channels,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def dwa_conv3x3_block(channels,
                      stride=1,
                      padding=1,
                      dilation=1,
                      bias=False,
                      use_bn=True,
                      bn_eps=1e-5,
                      activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 version of the depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return DwaConvBlock(
        channels=channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class DABUnit(nn.Module):
    """
    DABNet unit.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for a dilated branch in the unit.
    """
    def __init__(self,
                 channels,
                 dilation):
        super(DABUnit, self).__init__()
        mid_channels = channels // 2

        self.norm_activ1 = NormActivation(
            in_channels=channels,
            activation=(lambda: nn.PReLU(channels)))
        self.conv1 = conv3x3_block(
            in_channels=channels,
            out_channels=mid_channels,
            activation=(lambda: nn.PReLU(mid_channels)))

        self.branches = Concurrent(stack=True)
        self.branches.add_module("branches1", dwa_conv3x3_block(
            channels=mid_channels,
            activation=(lambda: nn.PReLU(mid_channels))))
        self.branches.add_module("branches2", dwa_conv3x3_block(
            channels=mid_channels,
            padding=dilation,
            dilation=dilation,
            activation=(lambda: nn.PReLU(mid_channels))))

        self.norm_activ2 = NormActivation(
            in_channels=mid_channels,
            activation=(lambda: nn.PReLU(mid_channels)))
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=channels)

    def forward(self, x):
        identity = x

        x = self.norm_activ1(x)
        x = self.conv1(x)

        x = self.branches(x)
        x = x.sum(dim=1)

        x = self.norm_activ2(x)
        x = self.conv2(x)

        x = x + identity
        return x


class PoolDownBlock(nn.Module):
    """
    DABNet specific simple downsample block via pooling.

    Parameters:
    ----------
    ratio : int
        Number of downsamples.
    """
    def __init__(self,
                 ratio):
        super(PoolDownBlock, self).__init__()
        self.pools = nn.Sequential()
        for i in range(ratio):
            self.pools.add_module("pool{}".format(i + 1), nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1))

    def forward(self, x):
        x = self.pools(x)
        return x


class DownBlock(nn.Module):
    """
    DABNet specific downsample block.

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
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels < self.out_channels:
            channels = out_channels - in_channels
        else:
            channels = out_channels

        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=channels,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        y = self.conv(x)

        if self.in_channels < self.out_channels:
            z = self.pool(x)
            y = torch.cat((y, z), dim=1)

        y = self.norm_activ(y)
        return y


class DABStage(nn.Module):
    """
    DABNet stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    down_times : int
        Number of downsamples for a simple downsampling.
    dilations : list of int
        Dilations for block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 down_times,
                 dilations):
        super(DABStage, self).__init__()
        self.use_block = (len(dilations) > 0)

        self.down1 = PoolDownBlock(down_times)

        if self.use_block:
            self.down2 = DownBlock(
                in_channels=in_channels + 3,
                out_channels=out_channels)
            self.block = nn.Sequential()
            for i, dilation_i in enumerate(dilations):
                self.block.add_module("unit{}".format(i + 1), DABUnit(
                    channels=out_channels,
                    dilation=dilation_i))

        channels1 = 2 * out_channels + 3
        self.norm_activ = NormActivation(
            in_channels=channels1,
            activation=(lambda: nn.PReLU(channels1)))

    def forward(self, x, y):
        x = self.down1(x)
        if self.use_block:
            y = self.down2(y)
            z = self.block(y)
            x = torch.cat((z, y, x), dim=1)
        else:
            x = torch.cat((y, x), dim=1)
        x = self.norm_activ(x)
        return x


class DABInitBlock(nn.Module):
    """
    DABNet specific initial block.

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
        super(DABInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            activation=(lambda: nn.PReLU(out_channels)))
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=(lambda: nn.PReLU(out_channels)))
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DABNet(nn.Module):
    """
    DABNet model from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

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
        super(DABNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes

        self.init_conv = DABInitBlock(
            in_channels=in_channels,
            out_channels=32)

        self.stage1 = DABStage(
            in_channels=3,
            out_channels=16,
            down_times=1,
            dilations=[])

        self.stage2 = DABStage(
            in_channels=32,
            out_channels=64,
            down_times=2,
            dilations=[2, 2, 2])

        self.stage3 = DABStage(
            in_channels=128,
            out_channels=128,
            down_times=3,
            dilations=[4, 4, 8, 8, 16, 16])

        in_channels = 2 * 128 + 3
        self.classifier = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes)

        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=False)

    def forward(self, x):
        y = self.init_conv(x)

        y = self.stage1(x, y)
        y = self.stage2(x, y)
        y = self.stage3(x, y)

        y = self.classifier(y)
        y = self.up(y, size=x.size()[2:])

        return y


def get_dabnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create DABNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = DABNet(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def dabnet_cityscapes(num_classes=19, **kwargs):
    """
    DABNet model for Cityscapes from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dabnet(num_classes=num_classes, model_name="dabnet_cityscapes", **kwargs)


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
        dabnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dabnet_cityscapes or weight_count == 756643)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, 19, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
