from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import math
import torch
import torch.nn as nn

__all__ = ['MSDNet']


class _DynamicInputDenseBlock(nn.Module):

    def __init__(self, conv_modules, debug):
        super(_DynamicInputDenseBlock, self).__init__()
        self.conv_modules = conv_modules
        self.debug = debug

    def forward(self, x):
        """
        Use the first element as raw input, and stream the rest of
        the inputs through the list of modules, then apply concatenation.
        expect x to be [identity, first input, second input, ..]
        and len(x) - len(self.conv_modules) = 1 for identity

        :param x: Input
        :return: Concatenation of the input with 1 or more module outputs
        """
        if self.debug:
            for i, t in enumerate(x):
                print("Current input size[{}]: {}".format(i,
                                                          t.size()))

        # Init output
        out = x[0]

        # Apply all given modules and return output
        for calc, m in enumerate(self.conv_modules):
            out = torch.cat([out, m(x[calc + 1])], 1)

            if self.debug:
                print("Working on input number: %s" % calc)
                print("Added: ", m(x[calc + 1]).size())
                print("Current out size {}".format(out.size()))

        return out


class MSDLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 in_scales, out_scales, orig_scales, args):
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
        super(MSDLayer, self).__init__()

        # Init vars
        self.current_channels = in_channels
        self.out_channels = out_channels
        self.in_scales = in_scales
        self.out_scales = out_scales
        self.orig_scales = orig_scales
        self.args = args
        self.bottleneck = args.msd_bottleneck
        self.bottleneck_factor = args.msd_bottleneck_factor
        self.growth_factor = self.args.msd_growth_factor
        self.debug = self.args.debug

        # Define Conv2d/GCN params
        self.use_gcn = args.msd_all_gcn
        self.conv_l, self.ks, self.pad = get_conv_params(self.use_gcn, args)

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
            in_channels1 = self.current_channels *\
                          self.growth_factor[self.dropped - 1]
            in_channels2 = self.current_channels *\
                           self.growth_factor[self.dropped]
            out_channels = self.out_channels *\
                           self.growth_factor[self.dropped]
            bn_width1 = self.bottleneck_factor[self.dropped - 1]
            bn_width2 = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_down_densenet(in_channels1,
                                                    in_channels2,
                                                    out_channels,
                                                    self.bottleneck,
                                                    bn_width1,
                                                    bn_width2))
        else:
            # Create a normal first scale
            in_channels = self.current_channels *\
                          self.growth_factor[self.dropped]
            out_channels = self.out_channels *\
                           self.growth_factor[self.dropped]
            bn_width = self.bottleneck_factor[self.dropped]
            subnets.append(self.build_densenet(in_channels,
                                               out_channels,
                                               self.bottleneck,
                                               bn_width))


        # Build second+ scales
        for scale in range(1, self.out_scales):
            in_channels1 = self.current_channels *\
                          self.growth_factor[self.dropped + scale - 1]
            in_channels2 = self.current_channels *\
                           self.growth_factor[self.dropped + scale]
            out_channels = self.out_channels *\
                           self.growth_factor[self.dropped + scale]
            bn_width1 = self.bottleneck_factor[self.dropped + scale - 1]
            bn_width2 = self.bottleneck_factor[self.dropped + scale]
            subnets.append(self.build_down_densenet(in_channels1,
                                                    in_channels2,
                                                    out_channels,
                                                    self.bottleneck,
                                                    bn_width1,
                                                    bn_width2))

        return subnets

    def build_down_densenet(self, in_channels1, in_channels2, out_channels,
                            bottleneck, bn_width1, bn_width2):
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
        conv_module1 = self.convolve(in_channels1, int(out_channels/2), 'down',
                                    bottleneck, bn_width1)
        conv_module2 = self.convolve(in_channels2, int(out_channels/2), 'normal',
                                    bottleneck, bn_width2)
        conv_modules = [conv_module1, conv_module2]
        return _DynamicInputDenseBlock(nn.ModuleList(conv_modules),
                                       self.debug)

    def build_densenet(self, in_channels, out_channels, bottleneck, bn_width):
        """
        Builds a scale sub-network for the first layer

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param bottleneck: A flag to perform a channel dimension bottleneck
        :param bn_width: The width of the bottleneck factor
        :return: A scale module
        """
        conv_module = self.convolve(in_channels, out_channels, 'normal',
                                    bottleneck, bn_width)
        return _DynamicInputDenseBlock(nn.ModuleList([conv_module]),
                                       self.debug)

    def convolve(self, in_channels, out_channels, conv_type,
                 bottleneck, bn_width=4):
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
            conv.add_module('Bottleneck_1x1', nn.Conv2d(in_channels,
                                                        tmp_channels,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0))
            conv.add_module('Bottleneck_BN', nn.BatchNorm2d(tmp_channels))
            conv.add_module('Bottleneck_ReLU', nn.ReLU(inplace=True))
        if conv_type == 'normal':
            conv.add_module('Spatial_forward', self.conv_l(tmp_channels,
                                                           out_channels,
                                                           kernel_size=self.ks,
                                                           stride=1,
                                                           padding=self.pad))
        elif conv_type == 'down':
            conv.add_module('Spatial_down', self.conv_l(tmp_channels, out_channels,
                                                        kernel_size=self.ks,
                                                        stride=2,
                                                        padding=self.pad))
        else: # Leaving an option to change the main conv type
            raise NotImplementedError

        conv.add_module('BN_out', nn.BatchNorm2d(out_channels))
        conv.add_module('ReLU_out', nn.ReLU(inplace=True))
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


class MSDFirstLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_scales, args):
        """
        Creates the first layer of the MSD network, which takes
        an input tensor (image) and generates a list of size num_scales
        with deeper features with smaller (spatial) dimensions.

        :param in_channels: number of input channels to the first layer
        :param out_channels: number of output channels in the first scale
        :param num_scales: number of output scales in the first layer
        :param args: other arguments
        """
        super(MSDFirstLayer, self).__init__()

        # Init params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales
        self.args = args
        self.use_gcn = args.msd_gcn
        self.conv_l, self.ks, self.pad = get_conv_params(self.use_gcn, args)
        if self.use_gcn:
            print('|          First layer with GCN           |')
        else:
            print('|         First layer without GCN         |')

        self.subnets = self.create_modules()

    def create_modules(self):

        # Create first scale features
        modules = nn.ModuleList()
        if 'cifar' in self.args.data:
            current_channels = int(self.out_channels *
                                   self.args.msd_growth_factor[0])

            current_m = nn.Sequential(
                self.conv_l(self.in_channels,
                       current_channels, kernel_size=self.ks,
                       stride=1, padding=self.pad),
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True)
            )
            modules.append(current_m)
        else:
            raise NotImplementedError

        # Create second scale features and down
        for scale in range(1, self.num_scales):

            # Calculate desired output channels
            out_channels = int(self.out_channels *
                               self.args.msd_growth_factor[scale])

            # Use a strided convolution to create next scale features
            current_m = nn.Sequential(
                self.conv_l(current_channels, out_channels,
                       kernel_size=self.ks,
                       stride=2, padding=self.pad),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            # Use the output channels size for the next scale
            current_channels = out_channels

            # Append module
            modules.append(current_m)

        return modules

    def forward(self, x):
        output = [None] * self.num_scales
        current_input = x
        for scale in range(0, self.num_scales):

            # Use upper scale as an input
            if scale > 0:
                current_input = output[scale-1]
            output[scale] = self.subnets[scale](current_input)
        return output


class Transition(nn.Sequential):

    def __init__(self, channels_in, channels_out,
                 out_scales, offset, growth_factor, args):
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

        super(Transition, self).__init__()
        self.args = args

        # Define a parallel stream for the different scales
        self.scales = nn.ModuleList()
        for i in range(0, out_scales):
            cur_in = channels_in * growth_factor[offset + i]
            cur_out = channels_out * growth_factor[offset + i]
            self.scales.append(self.conv1x1(cur_in, cur_out))

    def conv1x1(self, in_channels, out_channels):
        """
        Inner function to define the basic operation

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :return: A Sequential module to perform 1x1 convolution
        """
        scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        return scale

    def forward(self, x):
        """
        Propegate output through different scales.

        :param x: input to the transition layer
        :return: list of scales' outputs
        """
        if self.args.debug:
            print ("In transition forward!")

        output = []
        for scale, scale_net in enumerate(self.scales):
            if self.args.debug:
                print ("Size of x[{}]: {}".format(scale, x[scale].size()))
                print ("scale_net[0]: {}".format(scale_net[0]))
            output.append(scale_net(x[scale]))

        return output


class CifarClassifier(nn.Module):

    def __init__(self, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """

        super(CifarClassifier, self).__init__()
        self.inner_channels = 128

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )

        self.classifier = nn.Linear(self.inner_channels, num_classes)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """

        x = self.features(x)
        x = x.view(x.size(0), self.inner_channels)
        x = self.classifier(x)
        return x


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=1):
        """
        Global convolutional network module implementation

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of conv kernel
        :param stride: stride to use in the conv parts
        :param padding: padding to use in the conv parts
        :param share_weights: use shared weights for every side of GCN
        """
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                 padding=(padding, 0), stride=(stride, 1))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size),
                                 padding=(0, padding), stride=(1, stride))
        self.conv_r1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                                 padding=(0, padding), stride=(1, stride))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                                 padding=(padding, 0), stride=(stride, 1))


    def forward(self, x):

        if GCN.share_weights:

            # Prepare input and state
            self.conv_l1.shared = 2
            self.conv_l2.shared = 2
            xt = x.transpose(2,3)

            # Left convs
            xl = self.conv_l1(x)
            xl = self.conv_l2(xl)

            # Right convs
            xrt = self.conv_l1(xt)
            xrt = self.conv_l2(xrt)
            xr = xrt.transpose(2,3)
        else:

            # Left convs
            xl = self.conv_l1(x)
            xl = self.conv_l2(xl)

            # Right convs
            xr = self.conv_r1(x)
            xr = self.conv_r2(xr)

        return xl + xr


def get_conv_params(use_gcn, args):
    """
    Calculates and returns the convulotion parameters

    :param use_gcn: flag to use GCN or not
    :param args: user defined arguments
    :return: convolution type, kernel size and padding
    """

    if use_gcn:
        GCN.share_weights = args.msd_share_weights
        conv_l = GCN
        ks = args.msd_gcn_kernel
    else:
        conv_l = nn.Conv2d
        ks = args.msd_kernel
    pad = int(math.floor(ks / 2))
    return conv_l, ks, pad


class MSDNet(nn.Module):

    def __init__(self, args):
        """
        The main module for Multi Scale Dense Network.
        It holds the different blocks with layers and classifiers of the MSDNet layers

        :param args: Network argument
        """

        super(MSDNet, self).__init__()

        # Init arguments
        self.args = args
        self.base = self.args.msd_base
        self.step = self.args.msd_step
        self.step_mode = self.args.msd_stepmode
        self.msd_prune = self.args.msd_prune
        self.num_blocks = self.args.msd_blocks
        self.reduction_rate = self.args.reduction
        self.growth = self.args.msd_growth
        self.growth_factor = args.msd_growth_factor
        self.bottleneck = self.args.msd_bottleneck
        self.bottleneck_factor = args.msd_bottleneck_factor


        # Set progress
        if args.data in ['cifar10', 'cifar100']:
            self.image_channels = 3
            self.num_channels = 32
            self.num_scales = 3
            self.num_classes = int(args.data.strip('cifar'))
        else:
            raise NotImplementedError

        # Init MultiScale graph and fill with Blocks and Classifiers
        print('| MSDNet-Block {}-{}-{}'.format(self.num_blocks,
                                               self.step,
                                               self.args.data))
        (self.num_layers, self.steps) = self.calc_steps()

        print('Building network with the steps: {}'.format(self.steps))
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
            steps[i] = (self.step_mode == 'even' and self.step) or \
                        self.step*(i-1)+1
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
            print ('|-----------------Block {:0>2d}----------------|'.format(i+1))

            # Add block
            modules[i], num_channels = self.create_block(num_channels, i)

            # Calculate the last scale (smallest) channels size
            channels_in_last_layer = num_channels *\
                                     self.growth_factor[self.num_scales]

            # Add a classifier that belongs to the i'th block
            modules[i + self.num_blocks] = \
                CifarClassifier(channels_in_last_layer, self.num_classes)
        return modules

    def create_block(self, num_channels, block_num):
        '''
        :param num_channels: number of input channels to the block
        :param block_num: the number of the block (among all blocks)
        :return: A sequential container with steps[block_num] MSD layers
        '''

        block = nn.Sequential()

        # Add the first layer if needed
        if block_num == 0:
            block.add_module('MSD_first', MSDFirstLayer(self.image_channels,
                                                        num_channels,
                                                        self.num_scales,
                                                        self.args))

        # Add regular layers
        current_channels = num_channels
        for _ in range(0, self.steps[block_num]):

            # Calculate in and out scales of the layer (use paper heuristics)
            if self.msd_prune == 'max':
                interval = math.ceil(self.num_layers/
                                      self.num_scales)
                in_scales = int(self.num_scales - \
                            math.floor((max(0, self.cur_layer - 2))/interval))
                out_scales = int(self.num_scales - \
                             math.floor((self.cur_layer - 1)/interval))
            else:
                raise NotImplementedError

            self.print_layer(in_scales, out_scales)
            self.cur_layer += 1

            # Add an MSD layer
            block.add_module('MSD_layer_{}'.format(self.cur_layer - 1),
                             MSDLayer(current_channels,
                                      self.growth,
                                      in_scales,
                                      out_scales,
                                      self.num_scales,
                                      self.args))

            # Increase number of channel (as in densenet pattern)
            current_channels += self.growth

            # Add a transition layer if required
            if (self.msd_prune == 'max' and in_scales > out_scales and
                self.reduction_rate):

                # Calculate scales transition and add a Transition layer
                offset = self.num_scales - out_scales
                new_channels = int(math.floor(current_channels*
                                              self.reduction_rate))
                block.add_module('Transition', Transition(
                    current_channels, new_channels, out_scales,
                    offset, self.growth_factor, self.args))
                print('|      Transition layer {} was added!      |'.
                      format(self.cur_transition_layer))
                current_channels = new_channels

                # Increment counters
                self.cur_transition_layer += 1

            elif self.msd_prune != 'max':
                raise NotImplementedError

        return block, current_channels

    def print_layer(self, in_scales, out_scales):
        print('| Layer {:0>2d} input scales {} output scales {} |'.
              format(self.cur_layer, in_scales, out_scales))

    def forward(self, x, progress=None):
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
            if self.args.debug:
                print("")
                print("Forwarding to block %s:" % str(block_num + 1))
            block = self.subnets[block_num]
            cur_input = block_output = block(cur_input)

            # Classify and add current output
            if self.args.debug:
                print("- Getting %s block's output" % str(block_num + 1))
                for s, b in enumerate(block_output):
                    print("- Output size of this block's scale {}: ".format(s),
                          b.size())
            class_output = \
                self.subnets[block_num+self.num_blocks](block_output[-1])
            outputs[block_num] = class_output

        return outputs


