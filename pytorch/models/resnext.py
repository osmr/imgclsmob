"""
    ResNeXt & SE-ResNeXt, implemented in PyTorch.
    Original papers:
    - 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
    - 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['ResNeXt', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d', 'seresnext50_32x4d',
           'seresnext101_32x4d', 'seresnext101_64x4d', 'resnext_conv3x3', 'resnext_conv1x1']

import os
import math
import torch.nn as nn
import torch.nn.init as init
from .common import SEBlock


class ResNeXtConv(nn.Module):
    """
    ResNeXt specific convolution block.

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
    groups : int
        Number of groups.
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 activate):
        super(ResNeXtConv, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
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


def resnext_conv1x1(in_channels,
                    out_channels,
                    stride,
                    activate):
    """
    1x1 version of the ResNeXt specific convolution block.

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
    return ResNeXtConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=1,
        activate=activate)


def resnext_conv3x3(in_channels,
                    out_channels,
                    stride,
                    groups,
                    activate):
    """
    3x3 version of the ResNeXt specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int
        Number of groups.
    activate : bool
        Whether activate the convolution block.
    """
    return ResNeXtConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        activate=activate)


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt bottleneck block for residual path in ResNeXt unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 bottleneck_width):
        super(ResNeXtBottleneck, self).__init__()
        mid_channels = out_channels // 4
        D = int(math.floor(mid_channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.conv1 = resnext_conv1x1(
            in_channels=in_channels,
            out_channels=group_width,
            stride=1,
            activate=True)
        self.conv2 = resnext_conv3x3(
            in_channels=group_width,
            out_channels=group_width,
            stride=stride,
            groups=cardinality,
            activate=True)
        self.conv3 = resnext_conv1x1(
            in_channels=group_width,
            out_channels=out_channels,
            stride=1,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ResNeXtUnit(nn.Module):
    """
    ResNeXt unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    use_se : bool
        Whether to use SE block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 bottleneck_width,
                 use_se):
        super(ResNeXtUnit, self).__init__()
        self.use_se = use_se
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = ResNeXtBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width)
        if self.use_se:
            self.se = SEBlock(channels=out_channels)
        if self.resize_identity:
            self.identity_conv = resnext_conv1x1(
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
        if self.use_se:
            x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class ResNeXtInitBlock(nn.Module):
    """
    ResNeXt specific initial block.

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
        super(ResNeXtInitBlock, self).__init__()
        self.conv = ResNeXtConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            groups=1,
            activate=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResNeXt(nn.Module):
    """
    ResNeXt model from 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
    Also this class implements SE-ResNeXt from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    use_se : bool
        Whether to use SE block.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 cardinality,
                 bottleneck_width,
                 use_se,
                 in_channels=3,
                 num_classes=1000):
        super(ResNeXt, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResNeXtInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), ResNeXtUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    use_se=use_se))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_resnext(blocks,
                cardinality,
                bottleneck_width,
                use_se=False,
                model_name=None,
                pretrained=False,
                root=os.path.join('~', '.torch', 'models'),
                **kwargs):
    """
    Create ResNeXt or SE-ResNeXt model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    use_se : bool
        Whether to use SE block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported ResNeXt with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ResNeXt(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        use_se=use_se,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        import torch
        from .model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file(
            model_name=model_name,
            local_model_store_dir_path=root)))

    return net


def resnext50_32x4d(**kwargs):
    """
    ResNeXt-50 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="resnext50_32x4d", **kwargs)


def resnext101_32x4d(**kwargs):
    """
    ResNeXt-101 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="resnext101_32x4d", **kwargs)


def resnext101_64x4d(**kwargs):
    """
    ResNeXt-101 (64x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="resnext101_64x4d", **kwargs)


def seresnext50_32x4d(**kwargs):
    """
    SE-ResNeXt-50 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=50, cardinality=32, bottleneck_width=4, use_se=True, model_name="seresnext50_32x4d",
                       **kwargs)


def seresnext101_32x4d(**kwargs):
    """
    SE-ResNeXt-101 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=32, bottleneck_width=4, use_se=True, model_name="seresnext101_32x4d",
                       **kwargs)


def seresnext101_64x4d(**kwargs):
    """
    SE-ResNeXt-101 (64x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnext(blocks=101, cardinality=64, bottleneck_width=4, use_se=True, model_name="seresnext101_64x4d",
                       **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = True

    models = [
        # resnext50_32x4d,
        resnext101_32x4d,
        resnext101_64x4d,
        seresnext50_32x4d,
        seresnext101_32x4d,
        # seresnext101_64x4d,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        # print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnext50_32x4d or weight_count == 25028904)
        assert (model != resnext101_32x4d or weight_count == 44177704)
        assert (model != resnext101_64x4d or weight_count == 83455272)
        assert (model != seresnext50_32x4d or weight_count == 27559896)
        assert (model != seresnext101_32x4d or weight_count == 48955416)
        assert (model != seresnext101_64x4d or weight_count == 88232984)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

