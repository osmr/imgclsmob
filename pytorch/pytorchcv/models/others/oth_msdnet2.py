from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import math
from inspect import isfunction
import torch
import torch.nn as nn

__all__ = ['oth_msdnet_cifar10_2']


class MultiOutputSequential(nn.Sequential):
    """
    A sequential container for modules.
    Blocks will be executed in the order they are added.
    Output value contains results from all modules.
    """
    def __init__(self, *args):
        super(MultiOutputSequential, self).__init__(*args)

    def forward(self, x):
        out = []
        for module in self._modules.values():
            x = module(x)
            out.append(x)
        return out


class SesquialteralSequential(nn.Sequential):
    """
    A sequential container for modules with double results. The first result is forwarded sequentially. The second
    results are collected. Blocks will be executed in the order they are added.
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


class MSDFirstLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_scales,
                 growth_factors):
        super(MSDFirstLayer, self).__init__()
        self.num_scales = num_scales

        self.scale_blocks = MultiOutputSequential()
        for i in range(self.num_scales):
            stride = 1 if i == 0 else 2
            mid_channels = int(out_channels * growth_factors[i])
            self.scale_blocks.add_module('scale_block{}'.format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride,
                bias=True))
            in_channels = mid_channels

    def forward(self, x):
        y = self.scale_blocks(x)
        return y


class _DynamicInputDenseBlock(nn.Module):

    def __init__(self, conv_modules):
        super(_DynamicInputDenseBlock, self).__init__()
        self.conv_modules = conv_modules

    def forward(self, x):
        """
        Use the first element as raw input, and stream the rest of
        the inputs through the list of modules, then apply concatenation.
        expect x to be [identity, first input, second input, ..]
        and len(x) - len(self.conv_modules) = 1 for identity

        :param x: Input
        :return: Concatenation of the input with 1 or more module outputs
        """
        # Init output
        out = x[0]

        # Apply all given modules and return output
        for i, module in enumerate(self.conv_modules):
            out = torch.cat((out, module(x[i + 1])), dim=1)
        return out


class MSDLayer(nn.Module):
    """
    Creates a regular/transition MSDLayer. this layer uses DenseNet like concatenation on each scale,
    and performs spatial reduction between scales. if input and output scales are different, than this
    class creates a transition layer and the first layer (with the largest spatial size) is dropped.

    :param current_channels: number of input channels
    :param in_scales: number of input scales
    :param out_scales: number of output scales
    :param orig_scales: number of scales in the first layer of the MSDNet
    :param args: other arguments
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_scales,
                 out_scales,
                 orig_scales,
                 bottleneck,
                 bottleneck_factor,
                 growth_factor):
        super(MSDLayer, self).__init__()

        # Init vars
        self.current_channels = in_channels
        self.out_channels = out_channels
        self.in_scales = in_scales
        self.out_scales = out_scales
        self.orig_scales = orig_scales
        self.bottleneck = bottleneck
        self.bottleneck_factor = bottleneck_factor
        self.growth_factor = growth_factor

        # Calculate number of channels to drop and number of
        # all dropped channels
        self.to_drop = in_scales - out_scales
        self.dropped = orig_scales - out_scales # Use this as an offset
        self.subnets = self.get_subnets()

    def get_subnets(self):
        """
        Builds the different scales of the MSD network layer.

        :return: A list of scale modules
        """
        subnets = nn.ModuleList()

        # If this is a transition layer
        if self.to_drop:
            # Create a reduced feature map for the first scale
            # self.dropped > 0 since out_scales < in_scales < orig_scales
            in_channels1 = self.current_channels * self.growth_factor[self.dropped - 1]
            in_channels2 = self.current_channels * self.growth_factor[self.dropped]
            out_channels = self.out_channels * self.growth_factor[self.dropped]
            bn_width1 = self.bottleneck_factor[self.dropped - 1]
            bn_width2 = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_down_densenet(
                in_channels1,
                in_channels2,
                out_channels,
                self.bottleneck,
                bn_width1,
                bn_width2))
        else:
            # Create a normal first scale
            in_channels = self.current_channels * self.growth_factor[self.dropped]
            out_channels = self.out_channels * self.growth_factor[self.dropped]
            bn_width = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_densenet(
                in_channels,
                out_channels,
                self.bottleneck,
                bn_width))

        # Build second+ scales
        for scale in range(1, self.out_scales):
            in_channels1 = self.current_channels * self.growth_factor[self.dropped + scale - 1]
            in_channels2 = self.current_channels * self.growth_factor[self.dropped + scale]
            out_channels = self.out_channels * self.growth_factor[self.dropped + scale]
            bn_width1 = self.bottleneck_factor[self.dropped + scale - 1]
            bn_width2 = self.bottleneck_factor[self.dropped + scale]
            subnets.append(self.build_down_densenet(
                in_channels1,
                in_channels2,
                out_channels,
                self.bottleneck,
                bn_width1,
                bn_width2))

        return subnets

    def build_down_densenet(self,
                            in_channels1,
                            in_channels2,
                            out_channels,
                            bottleneck,
                            bn_width1,
                            bn_width2):
        """
        Builds a scale sub-network for scales 2 and up.

        :param in_channels1: number of same scale input channels
        :param in_channels2: number of upper scale input channels
        :param out_channels: number of output channels
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width1: The first input width of the bottleneck factor
        :param bn_width2: The first input width of the bottleneck factor
        :return: A scale module
        """
        conv_module1 = self.convolve(
            in_channels1,
            int(out_channels/2),
            'down',
            bottleneck,
            bn_width1)
        conv_module2 = self.convolve(
            in_channels2,
            int(out_channels/2),
            'normal',
            bottleneck,
            bn_width2)
        conv_modules = [conv_module1, conv_module2]
        return _DynamicInputDenseBlock(nn.ModuleList(conv_modules))

    def build_densenet(self,
                       in_channels,
                       out_channels,
                       bottleneck,
                       bn_width):
        """
        Builds a scale sub-network for the first layer

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width: The width of the bottleneck factor
        :return: A scale module
        """
        conv_module = self.convolve(
            in_channels,
            out_channels,
            'normal',
            bottleneck,
            bn_width)
        return _DynamicInputDenseBlock(nn.ModuleList([conv_module]))

    def convolve(self,
                 in_channels,
                 out_channels,
                 conv_type,
                 bottleneck,
                 bn_width=4):
        """
        Doing the main convolution of a specific scale in the
        MSD network

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param conv_type: convolution type
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width: The width of the bottleneck factor
        :return: A Sequential module of the main convolution
        """
        conv = nn.Sequential()
        tmp_channels = in_channels

        # Bottleneck before the convolution
        if bottleneck:
            tmp_channels = int(min([in_channels, bn_width * out_channels]))
            conv.add_module('Bottleneck', conv1x1_block(
                in_channels=in_channels,
                out_channels=tmp_channels,
                bias=True))

        if conv_type == 'normal':
            conv.add_module('Spatial_forward', conv3x3_block(
                in_channels=tmp_channels,
                out_channels=out_channels,
                bias=True))
        elif conv_type == 'down':
            conv.add_module('Spatial_down', conv3x3_block(
                in_channels=tmp_channels,
                out_channels=out_channels,
                stride=2,
                bias=True))
        else:
            raise NotImplementedError

        return conv

    def forward(self, x):
        cur_input = []
        outputs = []

        # Prepare the different scales' inputs of the
        # current transition/regular layer
        if self.to_drop: # Transition
            for scale in range(0, self.out_scales):
                last_same_scale = x[self.to_drop + scale]
                last_upper_scale = x[self.to_drop + scale - 1]
                cur_input.append([last_same_scale,
                                  last_upper_scale,
                                  last_same_scale])
        else: # Regular

            # Add first scale's input
            cur_input.append([x[0], x[0]])

            # Add second+ scales' input
            for scale in range(1, self.out_scales):
                last_same_scale = x[scale]
                last_upper_scale = x[scale - 1]
                cur_input.append([last_same_scale,
                                  last_upper_scale,
                                  last_same_scale])

        # Flow inputs in subnets and fill outputs
        for scale in range(0, self.out_scales):
            outputs.append(self.subnets[scale](cur_input[scale]))

        return outputs


class Transition(nn.Sequential):
    """
    Performs 1x1 convolution to increase channels size after reducing a spatial size reduction
    in transition layer.

    :param channels_in: channels before the transition
    :param channels_out: channels after reduction
    :param out_scales: number of scales after the transition
    :param offset: gap between original number of scales to out_scales
    :param growth_factor: densenet channel growth factor
    :return: A Parallel trainable array with the scales after channel
             reduction
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_scales,
                 offset,
                 growth_factor):
        super(Transition, self).__init__()

        # Define a parallel stream for the different scales
        self.scale_nets = nn.ModuleList()
        for i in range(out_scales):
            in_channels_i = in_channels * growth_factor[offset + i]
            out_channels_i = out_channels * growth_factor[offset + i]
            self.scale_nets.append(conv1x1_block(
                in_channels=in_channels_i,
                out_channels=out_channels_i,
                bias=True))

    def forward(self, x):
        out = []
        for i, scale_net in enumerate(self.scale_nets):
            out.append(scale_net(x[i]))
        return out


class CifarClassifier(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes):
        super(CifarClassifier, self).__init__()
        mid_channels = 128

        self.features = nn.Sequential(
            conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2,
                bias=True),
            conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=2,
                bias=True),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2)
        )

        self.classifier = nn.Linear(
            in_features=mid_channels,
            out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MSDNet(nn.Module):

    def __init__(self,
                 init_block_out_channels,
                 scales,
                 num_scales,
                 num_blocks,
                 base,
                 steps,
                 reduction_rate,
                 growth,
                 growth_factor,
                 bottleneck,
                 bottleneck_factor,
                 in_channels,
                 num_classes):
        super(MSDNet, self).__init__()

        # Init arguments
        self.num_blocks = num_blocks
        self.base = base
        self.reduction_rate = reduction_rate
        self.growth = growth
        self.growth_factor = growth_factor
        self.bottleneck = bottleneck
        self.bottleneck_factor = bottleneck_factor

        # Set progress
        self.image_channels = in_channels
        self.num_classes = num_classes

        self.cur_layer = 1

        self.init_block = MSDFirstLayer(
            in_channels=self.image_channels,
            out_channels=init_block_out_channels,
            num_scales=num_scales,
            growth_factors=self.growth_factor)
        num_channels = init_block_out_channels

        feature_blocks = []
        classifiers = []
        in_scales = scales[0][0]
        for i in range(self.num_blocks):
            out_scales_list = scales[i]
            block, num_channels = self.create_block(
                in_scales=in_scales,
                out_scales_list=out_scales_list,
                in_channels=num_channels,
                num_steps=steps[i],
                num_scales=num_scales,
                growth=self.growth,
                growth_factor=self.growth_factor,
                bottleneck=self.bottleneck,
                bottleneck_factor=self.bottleneck_factor,
                reduction_rate=self.reduction_rate)
            feature_blocks.append(block)
            out_channels = num_channels * self.growth_factor[num_scales]
            classifiers.append(CifarClassifier(out_channels, self.num_classes))
            in_scales = out_scales_list[-1]
        self.feature_blocks = nn.ModuleList(feature_blocks)
        self.classifiers = nn.ModuleList(classifiers)

    def create_block(self,
                     in_scales,
                     out_scales_list,
                     in_channels,
                     num_steps,
                     num_scales,
                     growth,
                     growth_factor,
                     bottleneck,
                     bottleneck_factor,
                     reduction_rate):

        block = nn.Sequential()

        # Add regular layers
        for j in range(num_steps):
            out_scales = out_scales_list[j]

            self.cur_layer += 1

            # Add an MSD layer
            block.add_module('MSD_layer_{}'.format(self.cur_layer - 1), MSDLayer(
                in_channels=in_channels,
                out_channels=growth,
                in_scales=in_scales,
                out_scales=out_scales,
                orig_scales=num_scales,
                bottleneck=bottleneck,
                bottleneck_factor=bottleneck_factor,
                growth_factor=growth_factor))

            # Increase number of channel (as in densenet pattern)
            in_channels += growth

            # Add a transition layer if required
            if (in_scales > out_scales) and reduction_rate:

                # Calculate scales transition and add a Transition layer
                offset = num_scales - out_scales
                new_channels = int(math.floor(in_channels * reduction_rate))
                block.add_module('Transition', Transition(
                    in_channels=in_channels,
                    out_channels=new_channels,
                    out_scales=out_scales,
                    offset=offset,
                    growth_factor=growth_factor))
                in_channels = new_channels

            in_scales = out_scales

        return block, in_channels

    def forward(self, x, only_last=True):
        x = self.init_block(x)
        outs = []
        for i in range(self.num_blocks):
            x = self.feature_blocks[i](x)
            y = self.classifiers[i](x[-1])
            outs.append(y)
        if only_last:
            return outs[-1]
        else:
            return outs


def get_oth_msdnet_cifar10(in_channels,
                           num_classes):
    num_scales = 3
    num_blocks = 10
    base = 4
    step = 2
    step_mode = "even"
    reduction_rate = 0.5
    growth = 6  # [6, 12, 24]
    growth_factor = [1, 2, 4, 4]
    bottleneck = True
    bottleneck_factor = [1, 2, 4, 4]

    steps = [base]
    for i in range(num_blocks - 1):
        steps.append(step if step_mode == 'even' else step * i + 1)
    num_layers = sum(steps)

    cur_layer = 1
    scales = []
    for i in range(num_blocks):
        num_steps = steps[i]
        scales_i = []
        for _ in range(num_steps):
            interval = math.ceil(num_layers / num_scales)
            out_scales = int(num_scales - math.floor((cur_layer - 1) / interval))
            cur_layer += 1
            scales_i += [out_scales]
        scales += [scales_i]

    init_block_out_channels = 32

    return MSDNet(
        init_block_out_channels=init_block_out_channels,
        scales=scales,
        num_scales=num_scales,
        num_blocks=num_blocks,
        base=base,
        steps=steps,
        reduction_rate=reduction_rate,
        growth=growth,
        growth_factor=growth_factor,
        bottleneck=bottleneck,
        bottleneck_factor=bottleneck_factor,
        in_channels=in_channels,
        num_classes=num_classes)


def oth_msdnet_cifar10_2(in_channels=3, num_classes=10, pretrained=False):
    return get_oth_msdnet_cifar10(
        in_channels=in_channels,
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
