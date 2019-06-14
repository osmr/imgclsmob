"""
    BagNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.
"""

__all__ = ['BagNet', 'bagnet9', 'bagnet17', 'bagnet33']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, conv1x1_block, conv3x3_block, ConvBlock


class BagNetBottleneck(nn.Module):
    """
    BagNet bottleneck block for residual path in BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size of the second convolution.
    stride : int or tuple/list of 2 int
        Strides of the second convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bottleneck_factor=4):
        super(BagNetBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class BagNetUnit(nn.Module):
    """
    BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size of the second body convolution.
    stride : int or tuple/list of 2 int
        Strides of the second body convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride):
        super(BagNetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = BagNetBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride)
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
        x = self.body(x)
        if x.size(-1) != identity.size(-1):
            diff = identity.size(-1) - x.size(-1)
            identity = identity[:, :, :-diff, :-diff]
        x = x + identity
        x = self.activ(x)
        return x


class BagNetInitBlock(nn.Module):
    """
    BagNet specific initial block.

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
        super(BagNetInitBlock, self).__init__()
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BagNet(nn.Module):
    """
    BagNet model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_pool_size : int
        Size of the pooling windows for final pool.
    normal_kernel_sizes : list of int
        Count of the first units with 3x3 convolution window size for each stage.
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
                 final_pool_size,
                 normal_kernel_sizes,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(BagNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", BagNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != len(channels) - 1) else 1
                kernel_size = 3 if j < normal_kernel_sizes[i] else 1
                stage.add_module("unit{}".format(j + 1), BagNetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=final_pool_size,
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


def get_bagnet(field,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create BagNet model with specific parameters.

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

    layers = [3, 4, 6, 3]

    if field == 9:
        normal_kernel_sizes = [1, 1, 0, 0]
        final_pool_size = 27
    elif field == 17:
        normal_kernel_sizes = [1, 1, 1, 0]
        final_pool_size = 26
    elif field == 33:
        normal_kernel_sizes = [1, 1, 1, 1]
        final_pool_size = 24
    else:
        raise ValueError("Unsupported BagNet with field: {}".format(field))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BagNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_pool_size=final_pool_size,
        normal_kernel_sizes=normal_kernel_sizes,
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


def bagnet9(**kwargs):
    """
    BagNet-9 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=9, model_name="bagnet9", **kwargs)


def bagnet17(**kwargs):
    """
    BagNet-17 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=17, model_name="bagnet17", **kwargs)


def bagnet33(**kwargs):
    """
    BagNet-33 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=33, model_name="bagnet33", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        bagnet9,
        bagnet17,
        bagnet33,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bagnet9 or weight_count == 15688744)
        assert (model != bagnet17 or weight_count == 16213032)
        assert (model != bagnet33 or weight_count == 18310184)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
