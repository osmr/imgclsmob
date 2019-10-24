"""
    MobileNetV3 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
"""

__all__ = ['MobileNetV3', 'mobilenetv3_small_w7d20', 'mobilenetv3_small_wd2', 'mobilenetv3_small_w3d4',
           'mobilenetv3_small_w1', 'mobilenetv3_small_w5d4', 'mobilenetv3_large_w7d20', 'mobilenetv3_large_wd2',
           'mobilenetv3_large_w3d4', 'mobilenetv3_large_w1', 'mobilenetv3_large_w5d4']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block, SEBlock,\
    HSwish


class MobileNetV3Unit(nn.Module):
    """
    MobileNetV3 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    exp_channels : int
        Number of middle (expanded) channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    activation : str
        Activation function or name of activation function.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_channels,
                 stride,
                 use_kernel3,
                 activation,
                 use_se):
        super(MobileNetV3Unit, self).__init__()
        assert (exp_channels >= out_channels)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se
        self.use_exp_conv = exp_channels != out_channels
        mid_channels = exp_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation)
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=4,
                approx_sigmoid=True,
                round_mid=True)
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV3FinalBlock(nn.Module):
    """
    MobileNetV3 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_se):
        super(MobileNetV3FinalBlock, self).__init__()
        self.use_se = use_se

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation="hswish")
        if self.use_se:
            self.se = SEBlock(
                channels=out_channels,
                reduction=4,
                approx_sigmoid=True,
                round_mid=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3 classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 dropout_rate):
        super(MobileNetV3Classifier, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.activ = HSwish(inplace=True)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class MobileNetV3(nn.Module):
    """
    MobileNetV3 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    exp_channels : list of list of int
        Number of middle (expanded) channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    use_relu : list of list of int/bool
        Using ReLU activation flag for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
    final_use_se : bool
        Whether to use SE-module in the final block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 exp_channels,
                 init_block_channels,
                 final_block_channels,
                 classifier_mid_channels,
                 kernels3,
                 use_relu,
                 use_se,
                 first_stride,
                 final_use_se,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MobileNetV3, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            activation="hswish"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                exp_channels_ij = exp_channels[i][j]
                stride = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                use_kernel3 = kernels3[i][j] == 1
                activation = "relu" if use_relu[i][j] == 1 else "hswish"
                use_se_flag = use_se[i][j] == 1
                stage.add_module("unit{}".format(j + 1), MobileNetV3Unit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    exp_channels=exp_channels_ij,
                    use_kernel3=use_kernel3,
                    stride=stride,
                    activation=activation,
                    use_se=use_se_flag))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', MobileNetV3FinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            use_se=final_use_se))
        in_channels = final_block_channels
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = MobileNetV3Classifier(
            in_channels=in_channels,
            out_channels=num_classes,
            mid_channels=classifier_mid_channels,
            dropout_rate=0.2)

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


def get_mobilenetv3(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".torch", "models"),
                    **kwargs):
    """
    Create MobileNetV3 model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('small' or 'large').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if version == "small":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        first_stride = True
        final_block_channels = 576
    elif version == "large":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        first_stride = False
        final_block_channels = 960
    else:
        raise ValueError("Unsupported MobileNetV3 version {}".format(version))

    final_use_se = False
    classifier_mid_channels = 1280

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        exp_channels = [[round_channels(cij * width_scale) for cij in ci] for ci in exp_channels]
        init_block_channels = round_channels(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale)

    net = MobileNetV3(
        channels=channels,
        exp_channels=exp_channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        use_relu=use_relu,
        use_se=use_se,
        first_stride=first_stride,
        final_use_se=final_use_se,
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


def mobilenetv3_small_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_small_wd2(**kwargs):
    """
    MobileNetV3 Small 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.5, model_name="mobilenetv3_small_wd2", **kwargs)


def mobilenetv3_small_w3d4(**kwargs):
    """
    MobileNetV3 Small 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.75, model_name="mobilenetv3_small_w3d4", **kwargs)


def mobilenetv3_small_w1(**kwargs):
    """
    MobileNetV3 Small 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.0, model_name="mobilenetv3_small_w1", **kwargs)


def mobilenetv3_small_w5d4(**kwargs):
    """
    MobileNetV3 Small 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.25, model_name="mobilenetv3_small_w5d4", **kwargs)


def mobilenetv3_large_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_large_wd2(**kwargs):
    """
    MobileNetV3 Large 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.5, model_name="mobilenetv3_large_wd2", **kwargs)


def mobilenetv3_large_w3d4(**kwargs):
    """
    MobileNetV3 Large 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.75, model_name="mobilenetv3_large_w3d4", **kwargs)


def mobilenetv3_large_w1(**kwargs):
    """
    MobileNetV3 Large 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.0, model_name="mobilenetv3_large_w1", **kwargs)


def mobilenetv3_large_w5d4(**kwargs):
    """
    MobileNetV3 Large 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.25, model_name="mobilenetv3_large_w5d4", **kwargs)


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
        mobilenetv3_small_w7d20,
        mobilenetv3_small_wd2,
        mobilenetv3_small_w3d4,
        mobilenetv3_small_w1,
        mobilenetv3_small_w5d4,
        mobilenetv3_large_w7d20,
        mobilenetv3_large_wd2,
        mobilenetv3_large_w3d4,
        mobilenetv3_large_w1,
        mobilenetv3_large_w5d4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenetv3_small_w7d20 or weight_count == 2159600)
        assert (model != mobilenetv3_small_wd2 or weight_count == 2288976)
        assert (model != mobilenetv3_small_w3d4 or weight_count == 2581312)
        assert (model != mobilenetv3_small_w1 or weight_count == 2945288)
        assert (model != mobilenetv3_small_w5d4 or weight_count == 3643632)
        assert (model != mobilenetv3_large_w7d20 or weight_count == 2943080)
        assert (model != mobilenetv3_large_wd2 or weight_count == 3334896)
        assert (model != mobilenetv3_large_w3d4 or weight_count == 4263496)
        assert (model != mobilenetv3_large_w1 or weight_count == 5481752)
        assert (model != mobilenetv3_large_w5d4 or weight_count == 7459144)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
