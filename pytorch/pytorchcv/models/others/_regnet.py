"""
    RegNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.
"""

__all__ = ['RegNet', 'regnetx002']

import os
import numpy as np
import torch.nn as nn
from common import conv1x1_block, conv3x3_block


class RegNetBottleneck(nn.Module):
    """
    RegNet bottleneck block for residual path in RegNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_factor=4):
        super(RegNetBottleneck, self).__init__()
        self.resize = (stride > 1)
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class RegNetUnit(nn.Module):
    """
    RegNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(RegNetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = RegNetBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
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
            identity = self.identity_block(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class RegNet(nn.Module):
    """
    RegNet model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
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
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(RegNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), RegNetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = nn.Sequential()
        if dropout_rate > 0.0:
            self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=in_channels,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_regnet(width_init,
               width_slope,
               width_mult,
               depth,
               groups,
               use_se=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create RegNet model with specific parameters.

    Parameters:
    ----------
    width_init : float
        Initial width value.
    width_slope : float
        Width slope value.
    width_mult : float
        Width multiplier value.
    groups : int
        Number of groups.
    depth : int
        Depth value.
    use_se : bool, default False
        Whether to use SE-module.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    q = 8
    assert width_slope >= 0 and width_init > 0 and width_mult > 1 and width_init % q == 0

    # Generate continuous per-block widths:
    widths_cont = np.arange(depth) * width_slope + width_init

    # Generate quantized per-block widths:
    width_exps = np.round(np.log(widths_cont / width_init) / np.log(width_mult))
    widths = width_init * np.power(width_mult, width_exps)
    widths = np.round(np.divide(widths, q)) * q
    num_stages = len(np.unique(widths))

    # Generate per stage widths and depths (assumes widths are sorted):
    stage_widths, stage_depths = np.unique(widths, return_counts=True)

    # Adjusts the compatibility of widths and groups:
    stage_groups = [groups for _ in range(num_stages)]
    stage_groups = [min(g, w_bot) for g, w_bot in zip(stage_groups, stage_widths)]
    stage_widths = [int(round(w_bot / g) * g) for w_bot, g in zip(stage_widths, stage_groups)]

    channels = [[ci] * li for (ci, li) in zip(stage_widths, stage_depths)]

    init_block_channels = 32
    channels = []

    net = RegNet(
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


def regnetx002(**kwargs):
    """
    RegNetX-200MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=24, width_slope=36.44, width_mult=2.49, depth=13, groups=8, model_name="regnetx002",
                      **kwargs)


def regnetx004(**kwargs):
    """
    RegNetX-400MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=24, width_slope=24.48, width_mult=2.54, depth=22, groups=16, model_name="regnetx004",
                      **kwargs)


def regnetx006(**kwargs):
    """
    RegNetX-600MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=48, width_slope=36.97, width_mult=2.24, depth=16, groups=24, model_name="regnetx006",
                      **kwargs)


def regnetx008(**kwargs):
    """
    RegNetX-800MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=56, width_slope=35.73, width_mult=2.28, depth=16, groups=16, model_name="regnetx008",
                      **kwargs)


def regnetx016(**kwargs):
    """
    RegNetX-1.6GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=80, width_slope=34.01, width_mult=2.25, depth=18, groups=24, model_name="regnetx016",
                      **kwargs)


def regnetx032(**kwargs):
    """
    RegNetX-3.2GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=88, width_slope=26.31, width_mult=2.25, depth=25, groups=48, model_name="regnetx032",
                      **kwargs)


def regnetx040(**kwargs):
    """
    RegNetX-4.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=96, width_slope=38.65, width_mult=2.43, depth=23, groups=40, model_name="regnetx040",
                      **kwargs)


def regnetx064(**kwargs):
    """
    RegNetX-6.4GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=184, width_slope=60.83, width_mult=2.07, depth=17, groups=56, model_name="regnetx064",
                      **kwargs)


def regnetx080(**kwargs):
    """
    RegNetX-8.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=80, width_slope=49.56, width_mult=2.88, depth=23, groups=120, model_name="regnetx080",
                      **kwargs)


def regnetx120(**kwargs):
    """
    RegNetX-12GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=168, width_slope=73.36, width_mult=2.37, depth=19, groups=112,
                      model_name="regnetx120", **kwargs)


def regnetx160(**kwargs):
    """
    RegNetX-16GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=216, width_slope=55.59, width_mult=2.1, depth=22, groups=128, model_name="regnetx160",
                      **kwargs)


def regnetx320(**kwargs):
    """
    RegNetX-32GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=320, width_slope=69.86, width_mult=2.0, depth=23, groups=168, model_name="regnetx320",
                      **kwargs)


def regnety002(**kwargs):
    """
    RegNetY-200MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=24, width_slope=36.44, width_mult=2.49, depth=13, groups=8, use_se=True,
                      model_name="regnety002", **kwargs)


def regnety004(**kwargs):
    """
    RegNetY-400MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=48, width_slope=27.89, width_mult=2.09, depth=16, groups=8, use_se=True,
                      model_name="regnety004", **kwargs)


def regnety006(**kwargs):
    """
    RegNetY-600MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=48, width_slope=32.54, width_mult=2.32, depth=15, groups=16, use_se=True,
                      model_name="regnety006", **kwargs)


def regnety008(**kwargs):
    """
    RegNetY-800MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=56, width_slope=38.84, width_mult=2.4, depth=14, groups=16, use_se=True,
                      model_name="regnety008", **kwargs)


def regnety016(**kwargs):
    """
    RegNetY-1.6GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=48, width_slope=20.71, width_mult=2.65, depth=27, groups=24, use_se=True,
                      model_name="regnety016", **kwargs)


def regnety032(**kwargs):
    """
    RegNetY-3.2GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=80, width_slope=42.63, width_mult=2.66, depth=21, groups=24, use_se=True,
                      model_name="regnety032", **kwargs)


def regnety040(**kwargs):
    """
    RegNetY-4.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=96, width_slope=31.41, width_mult=2.24, depth=22, groups=64, use_se=True,
                      model_name="regnety040", **kwargs)


def regnety064(**kwargs):
    """
    RegNetY-6.4GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=112, width_slope=33.22, width_mult=2.27, depth=25, groups=72, use_se=True,
                      model_name="regnety064", **kwargs)


def regnety080(**kwargs):
    """
    RegNetY-8.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=192, width_slope=76.82, width_mult=2.19, depth=17, groups=56, use_se=True,
                      model_name="regnety080", **kwargs)


def regnety120(**kwargs):
    """
    RegNetY-12GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=168, width_slope=73.36, width_mult=2.37, depth=19, groups=112, use_se=True,
                      model_name="regnety120", **kwargs)


def regnety160(**kwargs):
    """
    RegNetY-16GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=200, width_slope=106.23, width_mult=2.48, depth=18, groups=112, use_se=True,
                      model_name="regnety160", **kwargs)


def regnety320(**kwargs):
    """
    RegNetY-32GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(width_init=232, width_slope=115.89, width_mult=2.53, depth=20, groups=232, use_se=True,
                      model_name="regnety320", **kwargs)


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
        regnetx002,
        regnetx004,
        regnetx006,
        regnetx008,
        regnetx016,
        regnetx032,
        regnetx040,
        regnetx064,
        regnetx080,
        regnetx120,
        regnetx160,
        regnetx320,
        regnety002,
        regnety004,
        regnety006,
        regnety008,
        regnety016,
        regnety032,
        regnety040,
        regnety064,
        regnety080,
        regnety120,
        regnety160,
        regnety320,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != regnetx002 or weight_count == 2684792)
        assert (model != regnetx004 or weight_count == 5157512)
        assert (model != regnetx006 or weight_count == 6196040)
        assert (model != regnetx008 or weight_count == 7259656)
        assert (model != regnetx016 or weight_count == 9190136)
        assert (model != regnetx032 or weight_count == 15296552)
        assert (model != regnetx040 or weight_count == 22118248)
        assert (model != regnetx064 or weight_count == 26209256)
        assert (model != regnetx080 or weight_count == 39572648)
        assert (model != regnetx120 or weight_count == 46106056)
        assert (model != regnetx160 or weight_count == 54278536)
        assert (model != regnetx320 or weight_count == 107811560)
        assert (model != regnety002 or weight_count == 3162996)
        assert (model != regnety004 or weight_count == 4344144)
        assert (model != regnety006 or weight_count == 6055160)
        assert (model != regnety008 or weight_count == 6263168)
        assert (model != regnety016 or weight_count == 11202430)
        assert (model != regnety032 or weight_count == 19436338)
        assert (model != regnety040 or weight_count == 20646656)
        assert (model != regnety064 or weight_count == 30583252)
        assert (model != regnety080 or weight_count == 39180068)
        assert (model != regnety120 or weight_count == 51822544)
        assert (model != regnety160 or weight_count == 83590140)
        assert (model != regnety320 or weight_count == 145046770)

        batch = 14
        size = 224
        x = torch.randn(batch, 3, size, size)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (batch, 1000))


if __name__ == "__main__":
    _test()
