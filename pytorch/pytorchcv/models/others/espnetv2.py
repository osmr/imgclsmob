import math
from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['espnetv2_wd2', 'espnetv2_w1', 'espnetv2_w5d8', 'espnetv2_w3d2', 'espnetv2_w2']


class DualPathSequential(nn.Sequential):
    """
    A sequential container for modules with dual inputs/outputs.
    Modules will be executed in the order they are added.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first modules with single input/output.
    last_ordinals : int, default 0
        Number of the final modules with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a module.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal module.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda module, x1, x2: module(x1, x2)),
                 dual_path_scheme_ordinal=(lambda module, x1, x2: (module(x1), x2))):
        super(DualPathSequential, self).__init__()
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def forward(self, x1, x2=None):
        length = len(self._modules.values())
        for i, module in enumerate(self._modules.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False):
    """
    Convolution 3x3 layer.

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
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias)


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
                  padding=0,
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
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
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
        padding=padding,
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


class PreActivation(nn.Module):
    """
    PreResNet like pure pre-activation block without convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """

    def __init__(self,
                 in_channels):
        super(PreActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ESPBlock(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 branches=4,
                 rfield=7,
                 use_avg=False):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param branches: # of parallel branches
        :param rfield: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super(ESPBlock, self).__init__()
        assert (out_channels % branches == 0)
        self.stride = stride
        n = out_channels // branches
        n1 = out_channels - (branches - 1) * n
        assert n == n1

        self.proj_1x1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=n,
            groups=branches,
            activation=(lambda: nn.PReLU(n)))

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(branches):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= rfield else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(branches):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(conv3x3(
                in_channels=n,
                out_channels=n,
                stride=stride,
                padding=d_rate,
                dilation=d_rate,
                groups=n))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=branches,
            activation=None,
            activate=False)
        self.br_after_cat = PreActivation(in_channels=out_channels)
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = use_avg

    def forward(self, input, input2):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp( # learn linear combinations using group point-wise convolutions
            self.br_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1) # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded, input2

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded), input2


class DownSampler(nn.Module):
    '''
    Down-sampling fucntion that has three parallel branches:
    (1) avg pooling,
    (2) EESP block with stride of 2 and
    (3) efficient long-range connection with the input.
    The output feature maps of branches from (1) and (2) are concatenated and then additively fused with (3) to produce
    the final output.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 input_reinf=3,
                 branches=4,
                 rfield=9,
                 reinf=True):
        '''
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param branches: # of parallel branches
            :param rfield: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        '''
        super(DownSampler, self).__init__()
        nout_new = out_channels - in_channels
        self.eesp = ESPBlock(
            in_channels=in_channels,
            out_channels=nout_new,
            stride=2,
            branches=branches,
            rfield=rfield,
            use_avg=True)
        self.avg = nn.AvgPool2d(
            kernel_size=3,
            padding=1,
            stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                conv3x3_block(
                    in_channels=input_reinf,
                    out_channels=input_reinf,
                    activation=(lambda: nn.PReLU(input_reinf))
                ),
                conv1x1_block(
                    in_channels=input_reinf,
                    out_channels=out_channels,
                    activation=None,
                    activate=False)
            )
        self.activ = nn.PReLU(out_channels)

    def forward(self, input, input2_=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        avg_out = self.avg(input)
        eesp_out, _ = self.eesp(input, None)
        output = torch.cat([avg_out, eesp_out], 1)

        if input2_ is not None:
            input2 = input2_.clone()

            # assuming the input is a square image
            # Shortcut connection with the input image
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(
                    input2,
                    kernel_size=3,
                    padding=1,
                    stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.activ(output), input2_


class ESPFinalBlock(nn.Module):
    """
    ESPNetv2 specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    final_groups : int
        Number of groups in the last convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 final_groups):
        super(ESPFinalBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            activation=(lambda: nn.PReLU(in_channels)))
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=final_groups,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EESPNet(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 branches,
                 rfields,
                 dropout_rate=0.2,
                 in_channels=3,
                 num_classes=1000):
        '''
        :param num_classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super(EESPNet, self).__init__()

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=3)
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            activation=(lambda: nn.PReLU(init_block_channels))))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    unit = DownSampler(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        branches=branches,
                        rfield=rfields[i][j])
                else:
                    unit = ESPBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        branches=branches,
                        rfield=rfields[i][j])
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('final_block', ESPFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels,
            final_groups=branches))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))
        self.features.add_module("final_dropout", nn.Dropout(p=dropout_rate))

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
        x = self.features(x, x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_espnetv2(width_scale=1.0,
                 **kwargs):
    """
    Create PreResNet model with specific parameters.

    Parameters:
    ----------
    width_scale : float, default 1.0
        Scale factor for width of layers.
    """
    assert (width_scale <= 2.0)

    branches = 4
    layers = [1, 4, 8, 4]

    rfields_list = [13, 11, 9, 7, 5]
    rfields = [[rfields_list[i]] + [rfields_list[i + 1]] * (li - 1) for (i, li) in enumerate(layers)]

    base_channels = 32
    weighed_base_channels = math.ceil(float(math.floor(base_channels * width_scale)) / branches) * branches
    channels_per_layers = [weighed_base_channels * pow(2, i + 1) for i in range(len(layers))]

    init_block_channels = base_channels if weighed_base_channels > base_channels else weighed_base_channels
    final_block_channels = 1024 if width_scale <= 1.5 else 1280

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = EESPNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        branches=branches,
        rfields=rfields,
        **kwargs)

    return net


def espnetv2_wd2(pretrained=False, **kwargs):
    return get_espnetv2(width_scale=0.5)


def espnetv2_w1(pretrained=False, **kwargs):
    return get_espnetv2(width_scale=1.0)


def espnetv2_w5d8(pretrained=False, **kwargs):
    return get_espnetv2(width_scale=1.25)


def espnetv2_w3d2(pretrained=False, **kwargs):
    return get_espnetv2(width_scale=1.5)


def espnetv2_w2(pretrained=False, **kwargs):
    return get_espnetv2(width_scale=2.0)


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
        espnetv2_wd2,
        espnetv2_w1,
        espnetv2_w5d8,
        espnetv2_w3d2,
        espnetv2_w2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espnetv2_wd2 or weight_count == 1241332)
        assert (model != espnetv2_w1 or weight_count == 1670072)
        assert (model != espnetv2_w5d8 or weight_count == 1965440)
        assert (model != espnetv2_w3d2 or weight_count == 2314856)
        assert (model != espnetv2_w2 or weight_count == 3498136)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
