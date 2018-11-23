"""
    DRN, implemented in PyTorch.
    Original paper: 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.
"""

__all__ = ['DRNC', 'drnc26', ]

import os
import torch.nn as nn
import torch.nn.init as init


class DRNConv(nn.Module):
    """
    DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 activate):
        super(DRNConv, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def drn_conv1x1(in_channels,
                out_channels,
                stride,
                activate):
    """
    1x1 version of the DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        dilation=1,
        activate=activate)


def drn_conv3x3(in_channels,
                out_channels,
                stride,
                dilation,
                activate):
    """
    3x3 version of the DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        activate=activate)


class DRNBlock(nn.Module):
    """
    Simple DRN block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for convolution layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation):
        super(DRNBlock, self).__init__()
        self.conv1 = drn_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            activate=True)
        self.conv2 = drn_conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            dilation=dilation,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DRNBottleneck(nn.Module):
    """
    DRN bottleneck block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation):
        super(DRNBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = drn_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=1,
            activate=True)
        self.conv2 = drn_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            dilation=dilation,
            activate=True)
        self.conv3 = drn_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DRNUnit(nn.Module):
    """
    DRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layers.
    residual : bool
        Whether do residual calculations.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation,
                 bottleneck,
                 residual):
        super(DRNUnit, self).__init__()
        assert ((not bottleneck) or residual)
        self.residual = residual
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = DRNBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation)
        else:
            self.body = DRNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation)
        # if self.residual and self.resize_identity:
        if self.resize_identity:
            self.identity_conv = drn_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activate=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        if self.residual:
            x = x + identity
        x = self.activ(x)
        return x


class DRNInitBlock(nn.Module):
    """
    DRN specific initial block.

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
        super(DRNInitBlock, self).__init__()
        self.conv = DRNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            activate=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class DRNC(nn.Module):
    """
    DRN-C model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of list of int
        Dilation values for 3x3 convolution layers for each unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 dilations,
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DRNC, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", DRNInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            bottleneck_i = bottleneck and (1 < i < 6)
            residual = bottleneck_i or (i < 6)
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (0 < i < 4) else 1
                stage.add_module("unit{}".format(j + 1), DRNUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilation=dilations[i][j],
                    bottleneck=bottleneck_i,
                    residual=residual))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.output = nn.Sequential()
        self.output.add_module('final_pool', nn.AvgPool2d(
            kernel_size=28,
            stride=1))
        self.output.add_module('final_conv', nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_drn(blocks,
            model_name=None,
            pretrained=False,
            root=os.path.join('~', '.torch', 'models'),
            **kwargs):
    """
    Create DRN model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 26:
        layers = [1, 1, 2, 2, 2, 2, 1, 1]
    elif blocks == 42:
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 58:
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    else:
        raise ValueError("Unsupported DRN with number of blocks: {}".format(blocks))

    if blocks < 50:
        channels_per_layers = [16, 32, 64, 128, 256, 512, 512, 512]
        bottleneck = False
    else:
        channels_per_layers = [16, 32, 256, 512, 1024, 2048, 512, 512]
        bottleneck = True

    dilations_per_layers = [1, 1, 1, 1, 2, 4, 2, 1]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    dilations = [[ci] * li for (ci, li) in zip(dilations_per_layers, layers)]

    init_block_channels = channels_per_layers[0]

    net = DRNC(
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
        bottleneck=bottleneck,
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


def drnc26(**kwargs):
    """
    DRN-C-26 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=26, model_name="drnc26", **kwargs)


def drnc42(**kwargs):
    """
    DRN-C-42 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=42, model_name="drnc42", **kwargs)


def drnc58(**kwargs):
    """
    DRN-C-58 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=58, model_name="drnc58", **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        drnc26,
        drnc42,
        drnc58,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != drnc26 or weight_count == 21126584)
        assert (model != drnc42 or weight_count == 31234744)
        assert (model != drnc58 or weight_count == 41591608)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
