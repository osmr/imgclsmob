"""
    TResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'TResNet: High Performance GPU-Dedicated Architecture,' https://arxiv.org/abs/2003.13630.

    NB: Not tested!
"""

__all__ = ['TResNet', 'tresnet_m', 'tresnet_l', 'tresnet_xl']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv1x1_block, conv3x3_block, SEBlock


def anti_aliased_downsample(x):
    """
    Anti-Aliased Downsample operation.

    Parameters:
    ----------
    x : Tensor
        Input tensor.

    Returns:
    -------
    Tensor
        Resulted tensor.
    """
    channels = x.shape[1]

    weight = torch.tensor([1., 2., 1.], dtype=x.dtype, device=x.device)
    weight = weight[:, None] * weight[None, :]
    weight = weight / torch.sum(weight)
    weight = weight[None, None, :, :].repeat((channels, 1, 1, 1))

    x_pad = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
    x = F.conv2d(x_pad, weight, stride=2, padding=0, groups=channels)
    return x


class TResBlock(nn.Module):
    """
    Simple TResNet block for residual path in TResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    activation : str
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 activation):
        super(TResBlock, self).__init__()
        self.resize = (stride > 1)

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation)
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation=activation)
        self.se = SEBlock(
            channels=out_channels,
            mid_channels=max(out_channels // 4, 64))

    def __call__(self, x):
        x = self.conv1(x)
        if self.resize:
            x = anti_aliased_downsample(x)
        x = self.conv2(x)
        x = self.se(x)
        return x


class TResBottleneck(nn.Module):
    """
    TResNet bottleneck block for residual path in TResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    use_se : bool
        Whether to use SE-module.
    activation : str
        Activation function or name of activation function.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_se,
                 activation,
                 bottleneck_factor=4):
        super(TResBottleneck, self).__init__()
        self.use_se = use_se
        self.resize = (stride > 1)
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=activation)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            activation=activation)
        if self.resize:
            self.pool = nn.AvgPool2d(
                kernel_size=3,
                stride=stride,
                padding=1)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                mid_channels=max(mid_channels * bottleneck_factor // 8, 64))
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=activation)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.resize:
            x = self.pool(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        return x


class ResADownBlock(nn.Module):
    """
    TResNet downsample block for the identity branch of a residual unit.

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
        super(ResADownBlock, self).__init__()
        assert (stride > 1)

        self.pool = nn.AvgPool2d(
            kernel_size=stride,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None)

    def __call__(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class TResUnit(nn.Module):
    """
    TResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    use_se : bool
        Whether to use SE-module.
    activation : str
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_se,
                 activation,
                 bottleneck=True):
        super(TResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = TResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_se=use_se,
                activation=activation)
        else:
            self.body = TResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=activation)
        if self.resize_identity:
            self.identity_block = ResADownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        self.activ = nn.ReLU(inplace=True)

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_block(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


def space_to_depth(x):
    """
    Space-to-Depth operation.

    Parameters:
    ----------
    x : Tensor
        Input tensor.

    Returns:
    -------
    Tensor
        Resulted tensor.
    """
    k = 4
    batch, channels, height, width = x.size()
    new_height = height // k
    new_width = width // k
    new_channels = channels * k * k
    x = x.view(batch, channels, new_height, k, new_width, k)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    x = x.view(batch, new_channels, new_height, new_width)
    return x


class TResInitBlock(nn.Module):
    """
    TResNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : str
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation):
        super(TResInitBlock, self).__init__()
        mid_channels = in_channels * 16

        self.conv = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=activation)

    def __call__(self, x):
        x = space_to_depth(x)
        x = anti_aliased_downsample(x)
        x = self.conv(x)
        return x


class TResNet(nn.Module):
    """
    TResNet model from 'TResNet: High Performance GPU-Dedicated Architecture,' https://arxiv.org/abs/2003.13630.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : list of bool
        Whether to use a bottleneck or simple block in units for each stage.
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
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(TResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        activation = (lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.features = nn.Sequential()
        self.features.add_module("init_block", TResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            activation=activation))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                use_se = not (i == len(channels) - 1)
                stage.add_module("unit{}".format(j + 1), TResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_se=use_se,
                    bottleneck=bottleneck[i],
                    activation=activation))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = nn.Sequential()
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


def get_tresnet(version,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create TResNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of TResNet ('m', 'l' or 'xl').
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if version == "m":
        layers = [3, 4, 11, 3]
        width_scale = 1.0
    elif version == "l":
        layers = [4, 5, 18, 3]
        width_scale = 1.2
    elif version == "xl":
        layers = [4, 5, 24, 3]
        width_scale = 1.3
    else:
        raise ValueError("Unsupported TResNet version {}".format(version))

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if width_scale != 1.0:
        init_block_channels = int(init_block_channels * width_scale)
        channels_per_layers = [init_block_channels * (2 ** i) for i in range(len(channels_per_layers))]

    bottleneck = [False, False, True, True]
    bottleneck_factor = 4
    channels_per_layers = [ci * bottleneck_factor if bi else ci for (ci, bi) in zip(channels_per_layers, bottleneck)]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = TResNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def tresnet_m(**kwargs):
    """
    TResNet-M model from 'TResNet: High Performance GPU-Dedicated Architecture,' https://arxiv.org/abs/2003.13630.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_tresnet(version="m", model_name="tresnet_m", **kwargs)


def tresnet_l(**kwargs):
    """
    TResNet-L model from 'TResNet: High Performance GPU-Dedicated Architecture,' https://arxiv.org/abs/2003.13630.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_tresnet(version="l", model_name="tresnet_l", **kwargs)


def tresnet_xl(**kwargs):
    """
    TResNet-XL model from 'TResNet: High Performance GPU-Dedicated Architecture,' https://arxiv.org/abs/2003.13630.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_tresnet(version="xl", model_name="tresnet_xl", **kwargs)


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
        (tresnet_m, 224),
        (tresnet_l, 224),
        (tresnet_xl, 224),
    ]

    for model, size in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != tresnet_m or weight_count == 31389032)
        assert (model != tresnet_l or weight_count == 55989256)
        assert (model != tresnet_xl or weight_count == 78436244)

        batch = 1
        x = torch.randn(batch, 3, size, size)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (batch, 1000))


if __name__ == "__main__":
    _test()
