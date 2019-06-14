"""
    MSDNet for CIFAR-10, implemented in PyTorch.
    Original paper: 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.
"""

__all__ = ['CIFAR10MSDNet', 'msdnet22_cifar10']

import os
import math
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3_block
from .msdnet import MultiOutputSequential, MSDFeatureBlock


class CIFAR10MSDInitLayer(nn.Module):
    """
    MSDNet initial (so-called first) layer for CIFAR-10.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : list/tuple of int
        Number of output channels for each scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(CIFAR10MSDInitLayer, self).__init__()

        self.scale_blocks = MultiOutputSequential()
        for i, out_channels_per_scale in enumerate(out_channels):
            stride = 1 if i == 0 else 2
            self.scale_blocks.add_module('scale_block{}'.format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels_per_scale,
                stride=stride))
            in_channels = out_channels_per_scale

    def forward(self, x):
        y = self.scale_blocks(x)
        return y


class CIFAR10MSDClassifier(nn.Module):
    """
    MSDNet classifier for CIFAR-10.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of classification classes.
    """

    def __init__(self,
                 in_channels,
                 num_classes):
        super(CIFAR10MSDClassifier, self).__init__()
        mid_channels = 128

        self.features = nn.Sequential()
        self.features.add_module("conv1", conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2))
        self.features.add_module("conv2", conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=2))
        self.features.add_module("pool", nn.AvgPool2d(
            kernel_size=2,
            stride=2))

        self.output = nn.Linear(
            in_features=mid_channels,
            out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class CIFAR10MSDNet(nn.Module):
    """
    MSDNet model for CIFAR-10 from 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.

    Parameters:
    ----------
    channels : list of list of list of int
        Number of output channels for each unit.
    init_layer_channels : list of int
        Number of output channels for the initial layer.
    num_feature_blocks : int
        Number of subnets.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factors : list of list of int
        Bottleneck factor for each layers and for each input scale.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    num_classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_layer_channels,
                 num_feature_blocks,
                 use_bottleneck,
                 bottleneck_factors,
                 in_channels=3,
                 in_size=(32, 32),
                 num_classes=10):
        super(CIFAR10MSDNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.init_layer = CIFAR10MSDInitLayer(
            in_channels=in_channels,
            out_channels=init_layer_channels)
        in_channels = init_layer_channels

        self.feature_blocks = nn.Sequential()
        self.classifiers = nn.Sequential()
        for i in range(num_feature_blocks):
            self.feature_blocks.add_module("block{}".format(i + 1), MSDFeatureBlock(
                in_channels=in_channels,
                out_channels=channels[i],
                use_bottleneck=use_bottleneck,
                bottleneck_factors=bottleneck_factors[i]))
            in_channels = channels[i][-1]
            self.classifiers.add_module("classifier{}".format(i + 1), CIFAR10MSDClassifier(
                in_channels=in_channels[-1],
                num_classes=num_classes))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, only_last=True):
        x = self.init_layer(x)
        outs = []
        for feature_block, classifier in zip(self.feature_blocks, self.classifiers):
            x = feature_block(x)
            y = classifier(x[-1])
            outs.append(y)
        if only_last:
            return outs[-1]
        else:
            return outs


def get_msdnet_cifar10(blocks,
                       model_name=None,
                       pretrained=False,
                       root=os.path.join("~", ".torch", "models"),
                       **kwargs):
    """
    Create MSDNet model for CIFAR-10 with specific parameters.

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

    assert (blocks == 22)

    num_scales = 3
    num_feature_blocks = 10
    base = 4
    step = 2
    reduction_rate = 0.5
    growth = 6
    growth_factor = [1, 2, 4, 4]
    use_bottleneck = True
    bottleneck_factor_per_scales = [1, 2, 4, 4]

    assert (reduction_rate > 0.0)
    init_layer_channels = [16 * c for c in growth_factor[:num_scales]]

    step_mode = "even"
    layers_per_subnets = [base]
    for i in range(num_feature_blocks - 1):
        layers_per_subnets.append(step if step_mode == 'even' else step * i + 1)
    total_layers = sum(layers_per_subnets)

    interval = math.ceil(total_layers / num_scales)
    global_layer_ind = 0

    channels = []
    bottleneck_factors = []

    in_channels_tmp = init_layer_channels
    in_scales = num_scales
    for i in range(num_feature_blocks):
        layers_per_subnet = layers_per_subnets[i]
        scales_i = []
        channels_i = []
        bottleneck_factors_i = []
        for j in range(layers_per_subnet):
            out_scales = int(num_scales - math.floor(global_layer_ind / interval))
            global_layer_ind += 1
            scales_i += [out_scales]
            scale_offset = num_scales - out_scales

            in_dec_scales = num_scales - len(in_channels_tmp)
            out_channels = [in_channels_tmp[scale_offset - in_dec_scales + k] + growth * growth_factor[scale_offset + k]
                            for k in range(out_scales)]
            in_dec_scales = num_scales - len(in_channels_tmp)
            bottleneck_factors_ij = bottleneck_factor_per_scales[in_dec_scales:][:len(in_channels_tmp)]

            in_channels_tmp = out_channels
            channels_i += [out_channels]
            bottleneck_factors_i += [bottleneck_factors_ij]

            if in_scales > out_scales:
                assert (in_channels_tmp[0] % growth_factor[scale_offset] == 0)
                out_channels1 = int(math.floor(in_channels_tmp[0] / growth_factor[scale_offset] * reduction_rate))
                out_channels = [out_channels1 * growth_factor[scale_offset + k] for k in range(out_scales)]
                in_channels_tmp = out_channels
                channels_i += [out_channels]
                bottleneck_factors_i += [[]]
            in_scales = out_scales

        in_scales = scales_i[-1]
        channels += [channels_i]
        bottleneck_factors += [bottleneck_factors_i]

    net = CIFAR10MSDNet(
        channels=channels,
        init_layer_channels=init_layer_channels,
        num_feature_blocks=num_feature_blocks,
        use_bottleneck=use_bottleneck,
        bottleneck_factors=bottleneck_factors,
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


def msdnet22_cifar10(**kwargs):
    """
    MSDNet-22 model for CIFAR-10 from 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_msdnet_cifar10(blocks=22, model_name="msdnet22_cifar10", **kwargs)


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
        msdnet22_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != msdnet22_cifar10 or weight_count == 4839544)  # 5440864

        x = torch.randn(1, 3, 32, 32)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()
