from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import math
from inspect import isfunction
import torch
import torch.nn as nn

__all__ = ['oth_msdnet_cifar10_2']


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
    """
    Creates the first layer of the MSD network, which takes
    an input tensor (image) and generates a list of size num_scales
    with deeper features with smaller (spatial) dimensions.

    :param in_channels: number of input channels to the first layer
    :param out_channels: number of output channels in the first scale
    :param num_scales: number of output scales in the first layer
    :param msd_growth_factor: ...
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_scales,
                 msd_growth_factor):
        super(MSDFirstLayer, self).__init__()
        self.num_scales = num_scales

        self.subnets = nn.ModuleList()
        for i in range(self.num_scales):
            stride = 1 if i == 0 else 2
            mid_channels = int(out_channels * msd_growth_factor[i])
            self.subnets.append(conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride,
                bias=True))
            in_channels = mid_channels

    def forward(self, x):
        out = []
        for i in range(self.num_scales):
            x = self.subnets[i](x)
            out.append(x)
        return out


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
    """
    Classifier of a cifar10/100 image.

    :param num_channels: Number of input channels to the classifier
    :param num_classes: Number of classes to classify
    """
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
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MSDNet(nn.Module):

    def __init__(self, args):
        """
        The main module for Multi Scale Dense Network.
        It holds the different blocks with layers and classifiers of the MSDNet layers

        :param args: Network argument
        """

        super(MSDNet, self).__init__()

        # Init arguments
        self.base = args["msd_base"]
        self.step = args["msd_step"]
        self.step_mode = args["msd_stepmode"]
        self.num_blocks = args["msd_blocks"]
        self.reduction_rate = args["reduction"]
        self.growth = args["msd_growth"]
        self.growth_factor = args["msd_growth_factor"]
        self.bottleneck = args["msd_bottleneck"]
        self.bottleneck_factor = args["msd_bottleneck_factor"]
        self.msd_growth_factor = args["msd_growth_factor"]

        # Set progress
        if args["data"] in ['cifar10', 'cifar100']:
            self.image_channels = 3
            self.num_channels = 32
            self.num_scales = 3
            self.num_classes = int(args["data"].strip('cifar'))
        else:
            raise NotImplementedError

        # Init MultiScale graph and fill with Blocks and Classifiers
        (self.num_layers, self.steps) = self.calc_steps()

        self.cur_layer = 1
        self.cur_transition_layer = 1
        self.subnets = nn.ModuleList(self.build_modules(self.num_channels))

        # initialize
        for m in self.subnets:
            self.init_weights(m)
            if hasattr(m,'__iter__'):
                for sub_m in m:
                    self.init_weights(sub_m)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def calc_steps(self):
        """Calculates the number of layers required in each
        Block and the total number of layers, according to
        the step and stepmod.

        :return: number of total layers and list of layers/steps per blocks
        """

        # Init steps array
        steps = [None]*self.num_blocks
        steps[0] = num_layers = self.base

        # Fill steps and num_layers
        for i in range(1, self.num_blocks):

            # Take even steps or calc next linear growth of a step
            steps[i] = (self.step_mode == 'even' and self.step) or self.step*(i-1)+1
            num_layers += steps[i]

        return num_layers, steps

    def build_modules(self, num_channels):
        """Builds all blocks and classifiers and add it
        into an array in the order of the format:
        [[block]*num_blocks [classifier]*num_blocks]
        where the i'th block corresponds to the (i+num_block) classifier.

        :param num_channels: number of input channels
        :return: An array with all blocks and classifiers
        """

        # Init the blocks & classifiers data structure
        modules = [None] * self.num_blocks * 2
        for i in range(0, self.num_blocks):

            # Add block
            modules[i], num_channels = self.create_block(num_channels, i)

            # Calculate the last scale (smallest) channels size
            channels_in_last_layer = num_channels * self.growth_factor[self.num_scales]

            # Add a classifier that belongs to the i'th block
            modules[i + self.num_blocks] = CifarClassifier(channels_in_last_layer, self.num_classes)
        return modules

    def create_block(self,
                     num_channels,
                     block_num):
        '''
        :param num_channels: number of input channels to the block
        :param block_num: the number of the block (among all blocks)
        :return: A sequential container with steps[block_num] MSD layers
        '''

        block = nn.Sequential()

        # Add the first layer if needed
        if block_num == 0:
            block.add_module('MSD_first', MSDFirstLayer(
                self.image_channels,
                num_channels,
                self.num_scales,
                self.msd_growth_factor))

        # Add regular layers
        current_channels = num_channels
        for _ in range(0, self.steps[block_num]):

            # Calculate in and out scales of the layer (use paper heuristics)
            interval = math.ceil(self.num_layers / self.num_scales)
            in_scales = int(self.num_scales - math.floor((max(0, self.cur_layer - 2)) / interval))
            out_scales = int(self.num_scales - math.floor((self.cur_layer - 1) / interval))

            self.cur_layer += 1

            # Add an MSD layer
            block.add_module('MSD_layer_{}'.format(self.cur_layer - 1), MSDLayer(
                in_channels=current_channels,
                out_channels=self.growth,
                in_scales=in_scales,
                out_scales=out_scales,
                orig_scales=self.num_scales,
                bottleneck=self.bottleneck,
                bottleneck_factor=self.bottleneck_factor,
                growth_factor=self.msd_growth_factor))

            # Increase number of channel (as in densenet pattern)
            current_channels += self.growth

            # Add a transition layer if required
            if (in_scales > out_scales) and self.reduction_rate:

                # Calculate scales transition and add a Transition layer
                offset = self.num_scales - out_scales
                new_channels = int(math.floor(current_channels * self.reduction_rate))
                block.add_module('Transition', Transition(
                    in_channels=current_channels,
                    out_channels=new_channels,
                    out_scales=out_scales,
                    offset=offset,
                    growth_factor=self.growth_factor))
                current_channels = new_channels

                # Increment counters
                self.cur_transition_layer += 1

        return block, current_channels

    def forward(self, x, all=False, progress=None):
        """
        Propagate Input image in all blocks of MSD layers and classifiers
        and return a list of classifications

        :param x: Input image / batch
        :return: a list of classification outputs
        """

        outputs = [None] * self.num_blocks
        cur_input = x
        for block_num in range(0, self.num_blocks):

            # Get the current block's output
            block = self.subnets[block_num]
            cur_input = block_output = block(cur_input)

            # Classify and add current output
            class_output = self.subnets[block_num+self.num_blocks](block_output[-1])
            outputs[block_num] = class_output

        if all:
            return outputs
        else:
            return outputs[-1]


def oth_msdnet_cifar10_2(in_channels=3, num_classes=10, pretrained=False):
    args = {
        "msd_blocks": 10,
        "msd_base": 4,
        "msd_step": 2,
        "msd_stepmode": "even",
        "growth": [6, 12, 24],
        "msd_share_weights": False,
        "reduction": 0.5,
        "msd_growth": 6,
        "msd_growth_factor": [1, 2, 4, 4],
        "msd_bottleneck": True,
        "msd_bottleneck_factor": [1, 2, 4, 4],
        "data": "cifar10",
    }
    return MSDNet(args)


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
