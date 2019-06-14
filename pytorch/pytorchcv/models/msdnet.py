"""
    MSDNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.
"""

__all__ = ['MSDNet', 'msdnet22', 'MultiOutputSequential', 'MSDFeatureBlock']

import os
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block
from .resnet import ResInitBlock


class MultiOutputSequential(nn.Sequential):
    """
    A sequential container for modules. Modules will be executed in the order they are added. Output value contains
    results from all modules.
    """
    def __init__(self, *args):
        super(MultiOutputSequential, self).__init__(*args)

    def forward(self, x):
        outs = []
        for module in self._modules.values():
            x = module(x)
            outs.append(x)
        return outs


class MultiBlockSequential(nn.Sequential):
    """
    A sequential container for modules. Modules will be executed in the order they are added. Input is a list with
    length equal to number of modules.
    """
    def __init__(self, *args):
        super(MultiBlockSequential, self).__init__(*args)

    def forward(self, x):
        outs = []
        for module, x_i in zip(self._modules.values(), x):
            y = module(x_i)
            outs.append(y)
        return outs


class MSDBaseBlock(nn.Module):
    """
    MSDNet base block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factor : int
        Bottleneck factor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_bottleneck,
                 bottleneck_factor):
        super(MSDBaseBlock, self).__init__()
        self.use_bottleneck = use_bottleneck
        mid_channels = min(in_channels, bottleneck_factor * out_channels) if use_bottleneck else in_channels

        if self.use_bottleneck:
            self.bn_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
        self.conv = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=stride)

    def forward(self, x):
        if self.use_bottleneck:
            x = self.bn_conv(x)
        x = self.conv(x)
        return x


class MSDFirstScaleBlock(nn.Module):
    """
    MSDNet first scale dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factor : int
        Bottleneck factor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factor):
        super(MSDFirstScaleBlock, self).__init__()
        assert (out_channels > in_channels)
        inc_channels = out_channels - in_channels

        self.block = MSDBaseBlock(
            in_channels=in_channels,
            out_channels=inc_channels,
            stride=1,
            use_bottleneck=use_bottleneck,
            bottleneck_factor=bottleneck_factor)

    def forward(self, x):
        y = self.block(x)
        y = torch.cat((x, y), dim=1)
        return y


class MSDScaleBlock(nn.Module):
    """
    MSDNet ordinary scale dense block.

    Parameters:
    ----------
    in_channels_prev : int
        Number of input channels for the previous scale.
    in_channels : int
        Number of input channels for the current scale.
    out_channels : int
        Number of output channels.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factor_prev : int
        Bottleneck factor for the previous scale.
    bottleneck_factor : int
        Bottleneck factor for the current scale.
    """

    def __init__(self,
                 in_channels_prev,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factor_prev,
                 bottleneck_factor):
        super(MSDScaleBlock, self).__init__()
        assert (out_channels > in_channels)
        assert (out_channels % 2 == 0)
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels // 2

        self.down_block = MSDBaseBlock(
            in_channels=in_channels_prev,
            out_channels=mid_channels,
            stride=2,
            use_bottleneck=use_bottleneck,
            bottleneck_factor=bottleneck_factor_prev)
        self.curr_block = MSDBaseBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=1,
            use_bottleneck=use_bottleneck,
            bottleneck_factor=bottleneck_factor)

    def forward(self, x_prev, x):
        y_prev = self.down_block(x_prev)
        y = self.curr_block(x)
        x = torch.cat((x, y_prev, y), dim=1)
        return x


class MSDInitLayer(nn.Module):
    """
    MSDNet initial (so-called first) layer.

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
        super(MSDInitLayer, self).__init__()
        self.scale_blocks = MultiOutputSequential()
        for i, out_channels_per_scale in enumerate(out_channels):
            if i == 0:
                self.scale_blocks.add_module('scale_block{}'.format(i + 1), ResInitBlock(
                    in_channels=in_channels,
                    out_channels=out_channels_per_scale))
            else:
                self.scale_blocks.add_module('scale_block{}'.format(i + 1), conv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels_per_scale,
                    stride=2))
            in_channels = out_channels_per_scale

    def forward(self, x):
        y = self.scale_blocks(x)
        return y


class MSDLayer(nn.Module):
    """
    MSDNet ordinary layer.

    Parameters:
    ----------
    in_channels : list/tuple of int
        Number of input channels for each input scale.
    out_channels : list/tuple of int
        Number of output channels for each output scale.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factors : list/tuple of int
        Bottleneck factor for each input scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factors):
        super(MSDLayer, self).__init__()
        in_scales = len(in_channels)
        out_scales = len(out_channels)
        self.dec_scales = in_scales - out_scales
        assert (self.dec_scales >= 0)

        self.scale_blocks = nn.Sequential()
        for i in range(out_scales):
            if (i == 0) and (self.dec_scales == 0):
                self.scale_blocks.add_module('scale_block{}'.format(i + 1), MSDFirstScaleBlock(
                    in_channels=in_channels[self.dec_scales + i],
                    out_channels=out_channels[i],
                    use_bottleneck=use_bottleneck,
                    bottleneck_factor=bottleneck_factors[self.dec_scales + i]))
            else:
                self.scale_blocks.add_module('scale_block{}'.format(i + 1), MSDScaleBlock(
                    in_channels_prev=in_channels[self.dec_scales + i - 1],
                    in_channels=in_channels[self.dec_scales + i],
                    out_channels=out_channels[i],
                    use_bottleneck=use_bottleneck,
                    bottleneck_factor_prev=bottleneck_factors[self.dec_scales + i - 1],
                    bottleneck_factor=bottleneck_factors[self.dec_scales + i]))

    def forward(self, x):
        outs = []
        for i in range(len(self.scale_blocks)):
            if (i == 0) and (self.dec_scales == 0):
                y = self.scale_blocks[i](x[i])
            else:
                y = self.scale_blocks[i](
                    x_prev=x[self.dec_scales + i - 1],
                    x=x[self.dec_scales + i])
            outs.append(y)
        return outs


class MSDTransitionLayer(nn.Module):
    """
    MSDNet transition layer.

    Parameters:
    ----------
    in_channels : list/tuple of int
        Number of input channels for each scale.
    out_channels : list/tuple of int
        Number of output channels for each scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super(MSDTransitionLayer, self).__init__()
        assert (len(in_channels) == len(out_channels))

        self.scale_blocks = MultiBlockSequential()
        for i in range(len(out_channels)):
            self.scale_blocks.add_module('scale_block{}'.format(i + 1), conv1x1_block(
                in_channels=in_channels[i],
                out_channels=out_channels[i]))

    def forward(self, x):
        y = self.scale_blocks(x)
        return y


class MSDFeatureBlock(nn.Module):
    """
    MSDNet feature block (stage of cascade, so-called block).

    Parameters:
    ----------
    in_channels : list of list of int
        Number of input channels for each layer and for each input scale.
    out_channels : list of list of int
        Number of output channels for each layer and for each output scale.
    use_bottleneck : bool
        Whether to use a bottleneck.
    bottleneck_factors : list of list of int
        Bottleneck factor for each layer and for each input scale.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bottleneck,
                 bottleneck_factors):
        super(MSDFeatureBlock, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_channels_per_layer in enumerate(out_channels):
            if len(bottleneck_factors[i]) == 0:
                self.blocks.add_module('trans{}'.format(i + 1), MSDTransitionLayer(
                    in_channels=in_channels,
                    out_channels=out_channels_per_layer))
            else:
                self.blocks.add_module('layer{}'.format(i + 1), MSDLayer(
                    in_channels=in_channels,
                    out_channels=out_channels_per_layer,
                    use_bottleneck=use_bottleneck,
                    bottleneck_factors=bottleneck_factors[i]))
            in_channels = out_channels_per_layer

    def forward(self, x):
        x = self.blocks(x)
        return x


class MSDClassifier(nn.Module):
    """
    MSDNet classifier.

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
        super(MSDClassifier, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv1", conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=2))
        self.features.add_module("conv2", conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=2))
        self.features.add_module("pool", nn.AvgPool2d(
            kernel_size=2,
            stride=2))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class MSDNet(nn.Module):
    """
    MSDNet model from 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
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
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_layer_channels,
                 num_feature_blocks,
                 use_bottleneck,
                 bottleneck_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(MSDNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.init_layer = MSDInitLayer(
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
            self.classifiers.add_module("classifier{}".format(i + 1), MSDClassifier(
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


def get_msdnet(blocks,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create MSDNet model with specific parameters.

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

    num_scales = 4
    num_feature_blocks = 10
    base = 4
    step = 2
    reduction_rate = 0.5
    growth = 6
    growth_factor = [1, 2, 4, 4]
    use_bottleneck = True
    bottleneck_factor_per_scales = [1, 2, 4, 4]

    assert (reduction_rate > 0.0)
    init_layer_channels = [64 * c for c in growth_factor[:num_scales]]

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

    net = MSDNet(
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


def msdnet22(**kwargs):
    """
    MSDNet-22 model from 'Multi-Scale Dense Networks for Resource Efficient Image Classification,'
    https://arxiv.org/abs/1703.09844.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_msdnet(blocks=22, model_name="msdnet22", **kwargs)


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
        msdnet22,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != msdnet22 or weight_count == 20106676)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
