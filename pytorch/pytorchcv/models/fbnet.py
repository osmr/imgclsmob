"""
    FBNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,'
    https://arxiv.org/abs/1812.03443.
"""

__all__ = ['FBNet', 'fbnet_cb']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block


class FBNetUnit(nn.Module):
    """
    FBNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    bn_eps : float
        Small float added to variance in Batch norm.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int
        Expansion factor for each unit.
    activation : str, default 'relu'
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bn_eps,
                 use_kernel3,
                 exp_factor,
                 activation="relu"):
        super(FBNetUnit, self).__init__()
        assert (exp_factor >= 1)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_exp_conv = True
        mid_channels = exp_factor * in_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_eps=bn_eps,
                activation=activation)
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                bn_eps=bn_eps,
                activation=activation)
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                bn_eps=bn_eps,
                activation=activation)
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class FBNetInitBlock(nn.Module):
    """
    FBNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps):
        super(FBNetInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bn_eps=bn_eps)
        self.conv2 = FBNetUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            bn_eps=bn_eps,
            use_kernel3=True,
            exp_factor=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FBNet(nn.Module):
    """
    FBNet model from 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search,'
    https://arxiv.org/abs/1812.03443.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 kernels3,
                 exp_factors,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(FBNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", FBNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                use_kernel3 = kernels3[i][j] == 1
                exp_factor = exp_factors[i][j]
                stage.add_module("unit{}".format(j + 1), FBNetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bn_eps=bn_eps,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            bn_eps=bn_eps))
        in_channels = final_block_channels
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


def get_fbnet(version,
              bn_eps=1e-5,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create FBNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('a', 'b' or 'c').
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if version == "c":
        init_block_channels = 16
        final_block_channels = 1984
        channels = [[24, 24, 24], [32, 32, 32, 32], [64, 64, 64, 64, 112, 112, 112, 112], [184, 184, 184, 184, 352]]
        kernels3 = [[1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
        exp_factors = [[6, 1, 1], [6, 3, 6, 6], [6, 3, 6, 6, 6, 6, 6, 3], [6, 6, 6, 6, 6]]
    else:
        raise ValueError("Unsupported FBNet version {}".format(version))

    net = FBNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        bn_eps=bn_eps,
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


def fbnet_cb(**kwargs):
    """
    FBNet-Cb model (bn_eps=1e-3) from 'FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural
    Architecture Search,' https://arxiv.org/abs/1812.03443.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_fbnet(version="c", bn_eps=1e-3, model_name="fbnet_cb", **kwargs)


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
        fbnet_cb,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != fbnet_cb or weight_count == 5572200)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
