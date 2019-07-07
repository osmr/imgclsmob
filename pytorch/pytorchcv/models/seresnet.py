"""
    SE-ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['SEResNet', 'seresnet10', 'seresnet12', 'seresnet14', 'seresnet16', 'seresnet18', 'seresnet26',
           'seresnetbc26b', 'seresnet34', 'seresnetbc38b', 'seresnet50', 'seresnet50b', 'seresnet101', 'seresnet101b',
           'seresnet152', 'seresnet152b', 'seresnet200', 'seresnet200b', 'SEResUnit', 'get_seresnet']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, SEBlock
from .resnet import ResBlock, ResBottleneck, ResInitBlock


class SEResUnit(nn.Module):
    """
    SE-ResNet unit.

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
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 conv1_stride):
        super(SEResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                conv1_stride=conv1_stride)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
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
        x = self.body(x)
        x = self.se(x)
        x = x + identity
        x = self.activ(x)
        return x


class SEResNet(nn.Module):
    """
    SE-ResNet model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
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
                 bottleneck,
                 conv1_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(SEResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), SEResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
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


def get_seresnet(blocks,
                 bottleneck=None,
                 conv1_stride=True,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".torch", "models"),
                 **kwargs):
    """
    Create SE-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported SE-ResNet with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SEResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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


def seresnet10(**kwargs):
    """
    SE-ResNet-10 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=10, model_name="seresnet10", **kwargs)


def seresnet12(**kwargs):
    """
    SE-ResNet-12 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=12, model_name="seresnet12", **kwargs)


def seresnet14(**kwargs):
    """
    SE-ResNet-14 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=14, model_name="seresnet14", **kwargs)


def seresnet16(**kwargs):
    """
    SE-ResNet-16 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=16, model_name="seresnet16", **kwargs)


def seresnet18(**kwargs):
    """
    SE-ResNet-18 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=18, model_name="seresnet18", **kwargs)


def seresnet26(**kwargs):
    """
    SE-ResNet-26 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=26, bottleneck=False, model_name="seresnet26", **kwargs)


def seresnetbc26b(**kwargs):
    """
    SE-ResNet-BC-26b model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=26, bottleneck=True, conv1_stride=False, model_name="seresnetbc26b", **kwargs)


def seresnet34(**kwargs):
    """
    SE-ResNet-34 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=34, model_name="seresnet34", **kwargs)


def seresnetbc38b(**kwargs):
    """
    SE-ResNet-BC-38b model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=38, bottleneck=True, conv1_stride=False, model_name="seresnetbc38b", **kwargs)


def seresnet50(**kwargs):
    """
    SE-ResNet-50 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=50, model_name="seresnet50", **kwargs)


def seresnet50b(**kwargs):
    """
    SE-ResNet-50 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=50, conv1_stride=False, model_name="seresnet50b", **kwargs)


def seresnet101(**kwargs):
    """
    SE-ResNet-101 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=101, model_name="seresnet101", **kwargs)


def seresnet101b(**kwargs):
    """
    SE-ResNet-101 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=101, conv1_stride=False, model_name="seresnet101b", **kwargs)


def seresnet152(**kwargs):
    """
    SE-ResNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=152, model_name="seresnet152", **kwargs)


def seresnet152b(**kwargs):
    """
    SE-ResNet-152 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=152, conv1_stride=False, model_name="seresnet152b", **kwargs)


def seresnet200(**kwargs):
    """
    SE-ResNet-200 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=200, model_name="seresnet200", **kwargs)


def seresnet200b(**kwargs):
    """
    SE-ResNet-200 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_seresnet(blocks=200, conv1_stride=False, model_name="seresnet200b", **kwargs)


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
        seresnet10,
        seresnet12,
        seresnet14,
        seresnet16,
        seresnet18,
        seresnet26,
        seresnetbc26b,
        seresnet34,
        seresnetbc38b,
        seresnet50,
        seresnet50b,
        seresnet101,
        seresnet101b,
        seresnet152,
        seresnet152b,
        seresnet200,
        seresnet200b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != seresnet10 or weight_count == 5463332)
        assert (model != seresnet12 or weight_count == 5537896)
        assert (model != seresnet14 or weight_count == 5835504)
        assert (model != seresnet16 or weight_count == 7024640)
        assert (model != seresnet18 or weight_count == 11778592)
        assert (model != seresnet26 or weight_count == 18093852)
        assert (model != seresnetbc26b or weight_count == 17395976)
        assert (model != seresnet34 or weight_count == 21958868)
        assert (model != seresnetbc38b or weight_count == 24026616)
        assert (model != seresnet50 or weight_count == 28088024)
        assert (model != seresnet50b or weight_count == 28088024)
        assert (model != seresnet101 or weight_count == 49326872)
        assert (model != seresnet101b or weight_count == 49326872)
        assert (model != seresnet152 or weight_count == 66821848)
        assert (model != seresnet152b or weight_count == 66821848)
        assert (model != seresnet200 or weight_count == 71835864)
        assert (model != seresnet200b or weight_count == 71835864)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
