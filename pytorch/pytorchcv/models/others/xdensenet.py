"""
    X-DenseNet, implemented in PyTorch.
    Original paper: 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.
"""

__all__ = ['XDenseNet', 'xdensenet121', 'xdensenet161', 'xdensenet169', 'xdensenet201', 'XDenseUnit',
           'XDensTransitionBlock']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from pytorch.pytorchcv.models.common import pre_conv1x1_block, pre_conv3x3_block
from pytorch.pytorchcv.models.preresnet import PreResInitBlock, PreResActivation


class MulExpander(torch.autograd.Function):

    def __init__(self, mask):
        super(MulExpander, self).__init__()
        self.mask = mask

    def forward(self, weight):
        extend_weights = weight.clone()
        extend_weights.mul_(self.mask.data)
        return extend_weights

    def backward(self, grad_output):
        grad_weight = grad_output.clone()
        grad_weight.mul_(self.mask.data)
        return grad_weight


class execute2DConvolution(nn.Module):
    def __init__(self,
                 mask,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1):
        super(execute2DConvolution, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.mask = mask

    def forward(self, dataIn, weightIn):
        fpWeights = MulExpander(self.mask)(weightIn)
        return torch.nn.functional.conv2d(
            dataIn,
            fpWeights,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)


class ExpanderConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 expandSize,
                 stride=1,
                 padding=0,
                 inDil=1,
                 groups=1,
                 mode='random'):
        super(ExpanderConv2d, self).__init__()

        # Initialize all parameters that the convolution function needs to know
        self.kernel_size = kernel_size
        self.conStride = stride
        self.conPad = padding
        self.outPad = 0
        self.conDil = inDil
        self.conTrans = False
        self.conGroups = groups

        n = kernel_size * kernel_size * out_channels
        # initialize the weights and the bias as well as the
        self.fpWeight = torch.nn.Parameter(
            data=torch.Tensor(out_channels, in_channels, kernel_size, kernel_size),
            requires_grad=True)
        nn.init.kaiming_normal(self.fpWeight.data, mode='fan_out')

        self.mask = torch.zeros(out_channels, (in_channels), 1, 1)
        #print(inWCout,inWCin,expandSize)
        if in_channels > out_channels:
            for i in range(out_channels):
                x = torch.randperm(in_channels)
                for j in range(expandSize):
                    self.mask[i][x[j]][0][0] = 1
        else:
            for i in range(in_channels):
                x = torch.randperm(out_channels)
                for j in range(expandSize):
                    self.mask[x[j]][i][0][0] = 1

        self.mask = self.mask.repeat(1, 1, kernel_size, kernel_size)
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False

    def forward(self, dataInput):
        return execute2DConvolution(self.mask, self.conStride, self.conPad,self.conDil, self.conGroups)(dataInput, self.fpWeight)


class XDenseUnit(nn.Module):
    """
    X-DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : bool
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate):
        super(XDenseUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=inc_channels)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.cat((identity, x), dim=1)
        return x


class XDensTransitionBlock(nn.Module):
    """
    X-DenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(XDensTransitionBlock, self).__init__()
        self.conv = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class XDenseNet(nn.Module):
    """
    X-DenseNet model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(XDenseNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", PreResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            if i != 0:
                stage.add_module("trans{}".format(i + 1), XDensTransitionBlock(
                    in_channels=in_channels,
                    out_channels=(in_channels // 2)))
                in_channels = in_channels // 2
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), XDenseUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("post_activ", PreResActivation(in_channels=in_channels))
        self.features.add_module("final_pool", nn.AvgPool2d(
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


def get_xdensenet(blocks,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join('~', '.torch', 'models'),
                  **kwargs):
    """
    Create X-DenseNet model with specific parameters.

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

    if blocks == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif blocks == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif blocks == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif blocks == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported X-DenseNet version with number of layers {}".format(blocks))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = XDenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def xdensenet121(**kwargs):
    """
    X-DenseNet-121 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=121, model_name="xdensenet121", **kwargs)


def xdensenet161(**kwargs):
    """
    X-DenseNet-161 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=161, model_name="xdensenet161", **kwargs)


def xdensenet169(**kwargs):
    """
    X-DenseNet-169 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=169, model_name="xdensenet169", **kwargs)


def xdensenet201(**kwargs):
    """
    X-DenseNet-201 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=201, model_name="xdensenet201", **kwargs)


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
        xdensenet121,
        xdensenet161,
        xdensenet169,
        xdensenet201,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xdensenet121 or weight_count == 7978856)
        assert (model != xdensenet161 or weight_count == 28681000)
        assert (model != xdensenet169 or weight_count == 14149480)
        assert (model != xdensenet201 or weight_count == 20013928)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
