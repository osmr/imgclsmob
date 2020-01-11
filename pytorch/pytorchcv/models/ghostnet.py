"""
    GhostNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.
"""

__all__ = ['GhostNet', 'ghostnet']

import os
import math
import torch
import torch.nn as nn
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block,\
    dwsconv3x3_block, SEBlock


class GhostHSigmoid(nn.Module):
    """
    Approximated sigmoid function, specific for GhostNet.
    """

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=1.0)


class GhostConvBlock(nn.Module):
    """
    GhostNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(GhostConvBlock, self).__init__()
        main_out_channels = math.ceil(0.5 * out_channels)
        cheap_out_channels = out_channels - main_out_channels

        self.main_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=main_out_channels,
            activation=activation)
        self.cheap_conv = dwconv3x3_block(
            in_channels=main_out_channels,
            out_channels=cheap_out_channels,
            activation=activation)

    def forward(self, x):
        x = self.main_conv(x)
        y = self.cheap_conv(x)
        return torch.cat((x, y), dim=1)


class GhostExpBlock(nn.Module):
    """
    GhostNet expansion block for residual path in GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_kernel3,
                 exp_factor,
                 use_se):
        super(GhostExpBlock, self).__init__()
        self.use_dw_conv = (stride != 1)
        self.use_se = use_se
        mid_channels = int(math.ceil(exp_factor * in_channels))

        self.exp_conv = GhostConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels)
        if self.use_dw_conv:
            dw_conv_class = dwconv3x3_block if use_kernel3 else dwconv5x5_block
            self.dw_conv = dw_conv_class(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=None)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=4,
                out_activation=GhostHSigmoid())
        self.pw_conv = GhostConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.exp_conv(x)
        if self.use_dw_conv:
            x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class GhostUnit(nn.Module):
    """
    GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_kernel3,
                 exp_factor,
                 use_se):
        super(GhostUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = GhostExpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            use_kernel3=use_kernel3,
            exp_factor=exp_factor,
            use_se=use_se)
        if self.resize_identity:
            self.identity_conv = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                pw_activation=None)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class GhostClassifier(nn.Module):
    """
    GhostNet classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(GhostClassifier, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GhostNet(nn.Module):
    """
    GhostNet model from 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
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
                 final_block_channels,
                 classifier_mid_channels,
                 kernels3,
                 exp_factors,
                 use_se,
                 first_stride,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(GhostNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                use_kernel3 = kernels3[i][j] == 1
                exp_factor = exp_factors[i][j]
                use_se_flag = use_se[i][j] == 1
                stage.add_module("unit{}".format(j + 1), GhostUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor,
                    use_se=use_se_flag))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = GhostClassifier(
            in_channels=in_channels,
            out_channels=num_classes,
            mid_channels=classifier_mid_channels)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_ghostnet(width_scale=1.0,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".torch", "models"),
                 **kwargs):
    """
    Create GhostNet model with specific parameters.

    Parameters:
    ----------
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 16
    channels = [[16], [24, 24], [40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160, 160, 160]]
    kernels3 = [[1], [1, 1], [0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    exp_factors = [[1], [3, 3], [3, 3], [6, 2.5, 2.3, 2.3, 6, 6], [6, 6, 6, 6, 6]]
    use_se = [[0], [0, 0], [1, 1], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1]]
    final_block_channels = 960
    classifier_mid_channels = 1280
    first_stride = False

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale, divisor=4) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale, divisor=4)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale, divisor=4)

    net = GhostNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        use_se=use_se,
        first_stride=first_stride,
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


def ghostnet(**kwargs):
    """
    GhostNet model from 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_ghostnet(model_name="ghostnet", **kwargs)


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
        ghostnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ghostnet or weight_count == 5180840)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
