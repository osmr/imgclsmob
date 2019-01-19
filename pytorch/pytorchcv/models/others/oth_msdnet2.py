from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import math
from inspect import isfunction
import torch
import torch.nn as nn

__all__ = ['oth_msdnet_cifar10_2']


class MultiInputSequential(nn.Sequential):
    """
    A sequential container for modules. Modules will be executed in the order they are added.
    Input is a list with length equal to number of modules. Output value concatenates results from all modules.
    """
    def __init__(self, *args):
        super(MultiInputSequential, self).__init__(*args)

    def forward(self, x):
        outs = []
        for module, x_i in zip(self._modules.values(), x):
            y = module(x_i)
            outs.append(y)
        out = torch.cat(tuple(outs), dim=1)
        return out


class MultiOutputSequential(nn.Sequential):
    """
    A sequential container for modules. Modules will be executed in the order they are added.
    Output value contains results from all modules.
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
    A sequential container for modules. Modules will be executed in the order they are added.
    Input is a list with length equal to number of modules.
    """
    def __init__(self, *args):
        super(MultiBlockSequential, self).__init__(*args)

    def forward(self, x):
        outs = []
        for module, x_i in zip(self._modules.values(), x):
            y = module(x_i)
            outs.append(y)
        return outs


class SesquialteralSequential(nn.Sequential):
    """
    A sequential container for modules with double results. The first result is forwarded sequentially. The second
    results are collected. Modules will be executed in the order they are added.
    """
    def __init__(self, *args):
        super(SesquialteralSequential, self).__init__(*args)

    def forward(self, x):
        out = []
        for module in self._modules.values():
            x, y = module(x)
            out.append(y)
        return out


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            assert (activation is not None)
            if isfunction(activation):
                self.activ = activation()
            elif isinstance(activation, str):
                if activation == "relu":
                    self.activ = nn.ReLU(inplace=True)
                elif activation == "relu6":
                    self.activ = nn.ReLU6(inplace=True)
                else:
                    raise NotImplementedError()
            else:
                self.activ = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  groups=1,
                  bias=False,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
        activation=activation,
        activate=activate)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  activation=(lambda: nn.ReLU(inplace=True)),
                  activate=True):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        activation=activation,
        activate=activate)


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
                out_channels=mid_channels,
                bias=True)
        self.conv = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=stride,
            bias=True)

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
    MSDNet initial (first) layer.

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
            stride = 1 if i == 0 else 2
            self.scale_blocks.add_module('scale_block{}'.format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels_per_scale,
                stride=stride,
                bias=True))
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
        outputs = []
        for i in range(len(self.scale_blocks)):
            if (i == 0) and (self.dec_scales == 0):
                y = self.scale_blocks[i](x[i])
            else:
                y = self.scale_blocks[i](
                    x_prev=x[self.dec_scales + i - 1],
                    x=x[self.dec_scales + i])
            outputs.append(y)
        return outputs


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
                out_channels=out_channels[i],
                bias=True))

    def forward(self, x):
        y = self.scale_blocks(x)
        return y


class MSDFeatureBlock(nn.Module):
    """
    MSDNet feature block (stage of cascade).

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
        super(MSDClassifier, self).__init__()
        mid_channels = 128

        self.features = nn.Sequential()
        self.features.add_module("conv1", conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2,
            bias=True))
        self.features.add_module("conv2", conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=2,
            bias=True))
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


class MSDNet(nn.Module):

    def __init__(self,
                 channels,
                 init_layer_channels,
                 cls_channels,
                 num_feature_blocks,
                 use_bottleneck,
                 bottleneck_factors,
                 in_channels,
                 in_size=(32, 32),
                 num_classes=10):
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
                in_channels=cls_channels[i],
                num_classes=num_classes))

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


def get_oth_msdnet_cifar10(in_channels,
                           num_classes):
    num_scales = 3
    num_feature_blocks = 10
    base = 4
    step = 2
    step_mode = "even"
    reduction_rate = 0.5
    growth = 6
    growth_factor = [1, 2, 4, 4]
    use_bottleneck = True
    total_bottleneck_factor = [1, 2, 4, 4]

    assert (reduction_rate > 0.0)
    init_layer_out_channels = [32 * c for c in growth_factor[:3]]

    layers_per_subnets = [base]
    for i in range(num_feature_blocks - 1):
        layers_per_subnets.append(step if step_mode == 'even' else step * i + 1)
    total_layers = sum(layers_per_subnets)

    interval = math.ceil(total_layers / num_scales)
    total_layer_ind = 0
    scales = []

    total_out_channels = []
    cls_channels = []
    bottleneck_factors_list = []

    in_channels_tmp = init_layer_out_channels
    in_scales = num_scales
    for i in range(num_feature_blocks):
        layers_per_subnet = layers_per_subnets[i]
        scales_i = []

        channels_i = []
        bottleneck_factors_i = []
        for j in range(layers_per_subnet):
            out_scales = int(num_scales - math.floor(total_layer_ind / interval))
            total_layer_ind += 1
            scales_i += [out_scales]
            scale_offset = num_scales - out_scales

            in_dec_scales = num_scales - len(in_channels_tmp)
            out_channels = [in_channels_tmp[scale_offset - in_dec_scales + k] + growth * growth_factor[scale_offset + k] for
                            k in range(out_scales)]
            in_dec_scales = num_scales - len(in_channels_tmp)
            bottleneck_factors = total_bottleneck_factor[in_dec_scales:][:len(in_channels_tmp)]

            in_channels_tmp = out_channels
            channels_i += [out_channels]
            bottleneck_factors_i += [bottleneck_factors]

            if in_scales > out_scales:
                out_channels1 = int(math.floor(float(in_channels_tmp[0]) / growth_factor[scale_offset] * reduction_rate))
                out_channels = [out_channels1 * growth_factor[scale_offset + k] for k in range(out_scales)]
                in_channels_tmp = out_channels
                channels_i += [out_channels]
                bottleneck_factors_i += [[]]
            in_scales = out_scales

        in_dec_scales = num_scales - len(in_channels_tmp)
        in_channels1 = in_channels_tmp[0] // growth_factor[in_dec_scales]
        cls_channels_i = in_channels1 * growth_factor[num_scales]
        cls_channels += [cls_channels_i]
        bottleneck_factors_list += [bottleneck_factors_i]

        scales += [scales_i]
        in_scales = scales_i[-1]
        total_out_channels += [channels_i]

    # print("total_out_channels={}".format(total_out_channels))

    return MSDNet(
        channels=total_out_channels,
        init_layer_channels=init_layer_out_channels,
        cls_channels=cls_channels,
        num_feature_blocks=num_feature_blocks,
        use_bottleneck=use_bottleneck,
        bottleneck_factors=bottleneck_factors_list,
        in_channels=in_channels,
        num_classes=num_classes)


def oth_msdnet_cifar10_2(in_channels=3, num_classes=10, in_size=(32, 32), pretrained=False):
    return get_oth_msdnet_cifar10(
        in_channels=in_channels,
        in_size=in_size,
        num_classes=num_classes)


def load_model(net,
               file_path,
               ignore_extra=True):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        oth_msdnet_cifar10_2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_msdnet_cifar10_2 or weight_count == 5440864)

        x = Variable(torch.randn(1, 3, 32, 32))
        y = net(x)
        assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()
