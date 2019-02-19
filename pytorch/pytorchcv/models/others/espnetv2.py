import math
from inspect import isfunction
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

__all__ = ['espnetv2_wd2', 'espnetv2_w1', 'espnetv2_w5d8', 'espnetv2_w3d2', 'espnetv2_w2']


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


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1):
        '''

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param dilation: optional dilation rate
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=dilation,
            groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


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


class EESP(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 k=4,
                 r_lim=7,
                 down_method='esp'):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super(EESP, self).__init__()
        assert down_method in ['avg', 'esp']
        assert (out_channels % k == 0)
        self.stride = stride
        n = out_channels // k
        n1 = out_channels - (k - 1) * n
        assert n == n1

        self.proj_1x1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=n,
            groups=k,
            activation=(lambda: nn.PReLU(n)))

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(
                n,
                n,
                kernel_size=3,
                stride=stride,
                groups=n,
                dilation=d_rate))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=k,
            activation=None,
            activate=False)
        self.br_after_cat = PreActivation(in_channels=out_channels)
        self.module_act = nn.PReLU(out_channels)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
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
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


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
                 k=4,
                 r_lim=9,
                 reinf=True):
        '''
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        '''
        super(DownSampler, self).__init__()
        nout_new = out_channels - in_channels
        self.eesp = EESP(
            in_channels=in_channels,
            out_channels=nout_new,
            stride=2,
            k=k,
            r_lim=r_lim,
            down_method='avg')
        self.avg = nn.AvgPool2d(
            kernel_size=3,
            padding=1,
            stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                conv3x3_block(
                    in_channels=config_inp_reinf,
                    out_channels=config_inp_reinf,
                    activation=(lambda: nn.PReLU(config_inp_reinf))
                ),
                conv1x1_block(
                    in_channels=config_inp_reinf,
                    out_channels=out_channels,
                    activation=None,
                    activate=False)
            )
        self.activ = nn.PReLU(out_channels)

    def forward(self, input, input2=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)

        if input2 is not None:
            #assuming the input is a square image
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

        return self.activ(output)


class EESPNet(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self,
                 channels,
                 K,
                 r_lim,
                 reps,
                 in_channels=3,
                 classes=1000):
        '''
        :param classes: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super(EESPNet, self).__init__()
        global config_inp_reinf

        config_inp_reinf = 3
        self.input_reinforcement = True # True for the shortcut connection with input

        # assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(in_channels, channels[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(channels[0], channels[1], k=K[0], r_lim=r_lim[0], reinf=self.input_reinforcement)  # out = 56

        self.level3_0 = DownSampler(channels[1], channels[2], k=K[1], r_lim=r_lim[1], reinf=self.input_reinforcement) # out = 28
        self.level3 = nn.ModuleList()
        for i in range(reps[1]):
            self.level3.append(EESP(channels[2], channels[2], stride=1, k=K[2], r_lim=r_lim[2]))

        self.level4_0 = DownSampler(channels[2], channels[3], k=K[2], r_lim=r_lim[2], reinf=self.input_reinforcement) #out = 14
        self.level4 = nn.ModuleList()
        for i in range(reps[2]):
            self.level4.append(EESP(channels[3], channels[3], stride=1, k=K[3], r_lim=r_lim[3]))

        self.level5_0 = DownSampler(channels[3], channels[4], k=K[3], r_lim=r_lim[3]) #7
        self.level5 = nn.ModuleList()
        for i in range(reps[3]):
            self.level5.append(EESP(channels[4], channels[4], stride=1, k=K[4], r_lim=r_lim[4]))

        # expand the feature maps using depth-wise convolution followed by group point-wise convolution
        self.level5.append(CBR(channels[4], channels[4], 3, 1, groups=channels[4]))
        self.level5.append(CBR(channels[4], channels[5], 1, 1, groups=K[4]))

        self.classifier = nn.Linear(channels[5], classes)
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_l1 = self.level1(input)  # 112
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        out_l3_0 = self.level3_0(out_l2, input)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, input)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        out_l5_0 = self.level5_0(out_l4)  # down-sample
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)

        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)

        return self.classifier(output_1x1)


def get_espnetv2(width_scale=1.0,
                 **kwargs):
    """
    Create PreResNet model with specific parameters.

    Parameters:
    ----------
    width_scale : float, default 1.0
        Scale factor for width of layers.
    """
    reps = [0, 3, 7, 3]  # how many times EESP blocks should be repeated at each spatial level.

    r_lim = [13, 11, 9, 7, 5]  # receptive field at each spatial level
    K = [4] * len(r_lim)  # No. of parallel branches at different levels

    base = 32  # base configuration
    config_len = 5
    channels = [base] * config_len
    base_s = 0
    for i in range(config_len):
        if i == 0:
            base_s = int(base * width_scale)
            base_s = math.ceil(base_s / K[0]) * K[0]
            channels[i] = base if base_s > base else base_s
        else:
            channels[i] = base_s * pow(2, i)
    if width_scale <= 1.5:
        channels.append(1024)
    elif width_scale <= 2.0:
        channels.append(1280)
    else:
        ValueError('Configuration not supported')

    assert len(K) == len(r_lim), 'Length of branching factor array and receptive field array should be the same.'

    net = EESPNet(
        channels=channels,
        K=K,
        r_lim=r_lim,
        reps=reps,
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
