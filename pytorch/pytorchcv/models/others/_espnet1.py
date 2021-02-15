"""
    ESPNet for image segmentation, implemented in PyTorch.
    Original paper: 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation,'
    https://arxiv.org/abs/1803.06815.
"""

import torch
import torch.nn as nn
from common import NormActivation, conv1x1, conv3x3, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock,\
    Concurrent, DualPathSequential, InterpolationBlock


class HierarchicalConcurrent(nn.Sequential):
    """
    A container for hierarchical concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    exclude_first : bool, default False
        Whether to exclude the first branch in the intermediate sum.
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self,
                 exclude_first=False,
                 axis=1):
        super(HierarchicalConcurrent, self).__init__()
        self.exclude_first = exclude_first
        self.axis = axis

    def forward(self, x):
        out = []
        y_prev = None
        for i, module in enumerate(self._modules.values()):
            y = module(x)
            if y_prev is not None:
                y += y_prev
            out.append(y)
            if (not self.exclude_first) or (i > 0):
                y_prev = y
        out = torch.cat(tuple(out), dim=self.axis)
        return out


class ESPBlock(nn.Module):
    """
    ESPNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether to downsample image.
    residual : bool
        Whether to use residual connection.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample,
                 residual,
                 bn_eps):
        super(ESPBlock, self).__init__()
        self.residual = residual
        dilations = [1, 2, 4, 8, 16]
        num_branches = len(dilations)
        mid_channels = out_channels // num_branches
        extra_mid_channels = out_channels - (num_branches - 1) * mid_channels

        if downsample:
            self.reduce_conv = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2)
        else:
            self.reduce_conv = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)

        self.branches = HierarchicalConcurrent(exclude_first=True)
        for i in range(num_branches):
            out_channels_i = extra_mid_channels if i == 0 else mid_channels
            self.branches.add_module("branch{}".format(i + 1), conv3x3(
                in_channels=mid_channels,
                out_channels=out_channels_i,
                padding=dilations[i],
                dilation=dilations[i]))

        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        y = self.reduce_conv(x)
        y = self.branches(y)
        if self.residual:
            y = y + x
        y = self.norm_activ(y)
        return y


class ESPNetUnit(nn.Module):
    """
    ESPNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 bn_eps):
        super(ESPNetUnit, self).__init__()
        mid_channels = out_channels // 2

        self.down = ESPBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            downsample=True,
            residual=False,
            bn_eps=bn_eps)
        self.blocks = nn.Sequential()
        for i in range(layers - 1):
            self.blocks.add_module("block{}".format(i + 1), ESPBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                downsample=False,
                residual=True,
                bn_eps=bn_eps))

    def forward(self, x):
        x = self.down(x)
        y = self.blocks(x)
        x = torch.cat((y, x), dim=1)  # NB: This differs from the original implementation.
        return x


class ESPNetStage(nn.Module):
    """
    ESPNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    layers : int
        Number of layers in the unit.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 layers,
                 bn_eps):
        super(ESPNetStage, self).__init__()
        self.use_x = (x_channels > 0)
        self.use_unit = (layers > 0)

        if self.use_x:
            self.x_down = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1)

        if self.use_unit:
            self.unit = ESPNetUnit(
                in_channels=y_in_channels,
                out_channels=(y_out_channels - x_channels),
                layers=layers,
                bn_eps=bn_eps)

        self.norm_activ = NormActivation(
            in_channels=y_out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(y_out_channels)))

    def forward(self, y, x=None):
        if self.use_unit:
            y = self.unit(y)
        if self.use_x:
            x = self.x_down(x)
            y = torch.cat((y, x), dim=1)
        y = self.norm_activ(y)
        return y, x


class ESPCNet(nn.Module):
    """
    ESPNet-C model from .

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of int
        Dilations for each unit.
    se_reductions : list of int
        SE-block reduction value for each unit.
    cut_x : list of int
        Whether to concatenate with x-branch for each unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    num_classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 layers,
                 channels,
                 init_block_channels,
                 dilations,
                 se_reductions,
                 cut_x,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(ESPCNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=0)
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps))
        y_in_channels = init_block_channels

        for i, (layers_i, y_out_channels) in enumerate(zip(layers, channels)):
            self.features.add_module("stage{}".format(i + 1), CGStage(
                x_channels=in_channels if cut_x[i] == 1 else 0,
                y_in_channels=y_in_channels,
                y_out_channels=y_out_channels,
                layers=layers_i,
                dilation=dilations[i],
                se_reduction=se_reductions[i],
                bn_eps=bn_eps))
            y_in_channels = y_out_channels

        self.classifier = conv1x1(
            in_channels=y_in_channels,
            out_channels=num_classes)

        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        y = self.features(x, x)
        y = self.classifier(y)
        y = self.up(y, size=in_size)
        return y


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3, for input reinforcement, which establishes a direct link between 
    the input image and encoding stage, improving the flow of information.    
    '''
    def __init__(self,
                 samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(
                3,
                stride=2,
                padding=1))

    def forward(self, x):
        '''
        :param x: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            x = pool(x)
        return x


class ESPNetStage1(nn.Module):
    def __init__(self,
                 x_in_channels,
                 x_out_channels,
                 y_in_channels,
                 y_out_channels,
                 layers,
                 bn_eps):
        super().__init__()
        self.use_x = (x_out_channels > 0)

        in_channels = x_in_channels + y_in_channels
        out_channels = x_out_channels + y_out_channels
        y_mid_channels = y_out_channels // 2

        self.level3_0 = ESPBlock(
            in_channels=in_channels,
            out_channels=y_mid_channels,
            downsample=True,
            residual=False,
            bn_eps=bn_eps)
        self.level3 = nn.ModuleList()
        for i in range(0, layers):
            self.level3.append(ESPBlock(
                in_channels=y_mid_channels,
                out_channels=y_mid_channels,
                downsample=False,
                residual=True,
                bn_eps=bn_eps))
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, y, x=None):
        output2_0 = self.level3_0(y)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        y = torch.cat((output2_0, output2), dim=1)
        if self.use_x:
            y = torch.cat((y, x), dim=1)
        y = self.norm_activ(y)
        return y


class ESPNetC1(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
        :param num_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
    '''
    def __init__(self,
                 num_classes=19,
                 p=5,
                 q=3,
                 bn_eps=1e-03):
        super().__init__()
        self.level1 = conv3x3_block(
            in_channels=3,
            out_channels=16,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(16)))      # feature map size divided 2,                         1/2
        self.sample1 = InputProjectionA(samplingTimes=1)  #down-sample for input reinforcement, factor=2
        self.sample2 = InputProjectionA(samplingTimes=2)  #down-sample for input reinforcement, factor=4

        self.b1 = NormActivation(
            in_channels=(16 + 3),
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(16 + 3)))

        self.level2_ = ESPNetStage1(
            x_in_channels=3,
            x_out_channels=3,
            y_in_channels=16,
            y_out_channels=128,
            layers=p,
            bn_eps=bn_eps)

        # self.level2_0 = DilatedParllelResidualBlockB(
        #     in_channels=(16 + 3),
        #     out_channels=64,
        #     downsample=True,
        #     residual=False,
        #     bn_eps=bn_eps)  # Downsample Block, feature map size divided 2,    1/4
        # self.level2 = nn.ModuleList()
        # for i in range(0, p):
        #     self.level2.append(DilatedParllelResidualBlockB(
        #         in_channels=64,
        #         out_channels=64,
        #         downsample=False,
        #         residual=True,
        #         bn_eps=bn_eps))  # ESP block
        # self.b2 = NormActivation(
        #     in_channels=(128 + 3),
        #     bn_eps=bn_eps,
        #     activation=(lambda: nn.PReLU(128 + 3)))

        self.level3_ = ESPNetStage1(
            x_in_channels=3,
            x_out_channels=0,
            y_in_channels=128,
            y_out_channels=256,
            layers=q,
            bn_eps=bn_eps)

        # self.level3_0 = DownSamplerB(
        #     in_channels=128 + 3,
        #     out_channels=128,
        #     bn_eps=bn_eps)  # Downsample Block, feature map size divided 2,   1/8
        # self.level3 = nn.ModuleList()
        # for i in range(0, q):
        #     self.level3.append(DilatedParllelResidualBlockB(
        #         in_channels=128,
        #         out_channels=128,
        #         add=True,
        #         bn_eps=bn_eps))  # ESPblock
        # self.b3 = NormActivation(
        #     in_channels=256,
        #     bn_eps=bn_eps,
        #     activation=(lambda: nn.PReLU(256)))

        self.head = conv1x1(
            in_channels=256,
            out_channels=num_classes)
        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=False)

    def forward(self, x):
        output0 = self.level1(x)
        inp1 = self.sample1(x)
        inp2 = self.sample2(x)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))

        output1_cat = self.level2_(output0_cat, inp2)
        y = self.level3_(output1_cat)

        y = self.head(y)
        y = self.up(y)
        return y


class ESPNet(nn.Module):
    '''
    This class defines the ESPNet network
        :param num_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
    '''

    def __init__(self,
                 num_classes=19,
                 p=2,
                 q=3,
                 bn_eps=1e-03):
        super().__init__()
        self.encoder = ESPNetC1(
            num_classes=num_classes,
            p=p,
            q=q,
            bn_eps=bn_eps)
        # load the encoder modules
        self.en_modules = []
        for i, m in enumerate(self.encoder.children()):
            self.en_modules.append(m)

        # light-weight decoder
        self.level3_C = conv1x1(
            in_channels=(128 + 3),
            out_channels=num_classes)
        self.br = nn.BatchNorm2d(num_classes, eps=1e-03)
        self.conv = conv3x3_block(
            in_channels=(19 + num_classes),
            out_channels=num_classes,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(num_classes)))

        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(
            num_classes,
            num_classes,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False))
        self.combine_l2_l3 = nn.Sequential(
            NormActivation(
                in_channels=(2 * num_classes),
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(2 * num_classes))),
            ESPBlock(
                in_channels=(2 * num_classes),
                out_channels=num_classes,
                downsample=False,
                residual=False,
                bn_eps=bn_eps))

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(
                num_classes,
                num_classes,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False),
            NormActivation(
                in_channels=num_classes,
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(num_classes))))

        self.head = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False)

    def forward(self, x):
        output0 = self.en_modules[0](x)
        inp1 = self.en_modules[1](x)
        inp2 = self.en_modules[2](x)

        output0_cat = self.en_modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.en_modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.en_modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.en_modules[6](torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.en_modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.en_modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.en_modules[9](torch.cat([output2_0, output2], 1)) # concatenate for feature map width expansion

        output2_c = self.up_l3(self.br(self.en_modules[10](output2_cat))) #RUM

        output1_C = self.level3_C(output1_cat) # project to C-dimensional space
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1))) #RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.head(concat_features)

        return classifier


def espnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return ESPNet(num_classes=num_classes, **kwargs)


def espnetc_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return ESPNetC1(num_classes=num_classes, **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    # fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        # espnetc_cityscapes,
        espnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != espnet_cityscapes or weight_count == 201542)
        assert (model != espnetc_cityscapes or weight_count == 210889)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
