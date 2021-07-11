__all__ = ['inceptionresnetv1']

import torch
from torch import nn
from common import conv1x1, ConvBlock, conv1x1_block, conv3x3_block, Concurrent


class MaxPoolBranch(nn.Module):
    """
    InceptionResNetV2 specific max pooling branch block.
    """
    def __init__(self):
        super(MaxPoolBranch, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.pool(x)
        return x


class Conv1x1Branch(nn.Module):
    """
    InceptionResNetV2 specific convolutional 1x1 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps):
        super(Conv1x1Branch, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_eps=bn_eps)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvSeqBranch(nn.Module):
    """
    InceptionResNetV2 specific convolutional sequence branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 bn_eps):
        super(ConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.conv_list.add_module("conv{}".format(i + 1), ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                bn_eps=bn_eps))
            in_channels = out_channels

    def forward(self, x):
        x = self.conv_list(x)
        return x


class InceptionAUnit(nn.Module):
    """
    InceptionResNetV1 type Inception-A unit.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 bn_eps):
        super(InceptionAUnit, self).__init__()
        self.scale = 0.17
        in_channels = 256

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=32,
            bn_eps=bn_eps))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 32),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            bn_eps=bn_eps))
        self.branches.add_module("branch3", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(32, 32, 32),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            bn_eps=bn_eps))
        self.conv = conv1x1(
            in_channels=96,
            out_channels=in_channels,
            bias=True)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class InceptionBUnit(nn.Module):
    """
    InceptionResNetV1 type Inception-B unit.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 bn_eps):
        super(InceptionBUnit, self).__init__()
        self.scale = 0.10
        in_channels = 896

        self.branches = Concurrent()
        self.branches.add_module("branch1", Conv1x1Branch(
            in_channels=in_channels,
            out_channels=128,
            bn_eps=bn_eps))
        self.branches.add_module("branch2", ConvSeqBranch(
            in_channels=in_channels,
            out_channels_list=(128, 128, 128),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            bn_eps=bn_eps))
        self.conv = conv1x1(
            in_channels=256,
            out_channels=in_channels,
            bias=True)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class Block8(nn.Module):

    def __init__(self,
                 scale,
                 noReLU,
                 bn_eps):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = conv1x1_block(
            in_channels=1792,
            out_channels=192,
            stride=1,
            padding=0)

        self.branch1 = nn.Sequential(
            conv1x1_block(
                in_channels=1792,
                out_channels=192,
                stride=1,
                padding=0),
            ConvBlock(in_channels=192, out_channels=192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBlock(in_channels=192, out_channels=192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(in_channels=384, out_channels=1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = ConvBlock(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=0)

        self.branch1 = nn.Sequential(
            conv1x1_block(
                in_channels=256,
                out_channels=192,
                stride=1,
                padding=0),
            ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=192, out_channels=256, kernel_size=3, stride=2, padding=0)
        )

        self.branch2 = MaxPoolBranch()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            conv1x1_block(
                in_channels=896,
                out_channels=256,
                stride=1,
                padding=0),
            ConvBlock(in_channels=256, out_channels=384, kernel_size=3, stride=2, padding=0)
        )

        self.branch1 = nn.Sequential(
            ConvBlock(
                in_channels=896,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            ConvBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=0)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=896, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0)
        )

        self.branch3 = MaxPoolBranch()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self,
                 bn_eps=1e-5,
                 num_classes=1000,
                 dropout_prob=0.6):
        super().__init__()

        # Set simple attributes
        self.num_classes = num_classes

        # Define layers
        self.conv2d_1a = ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv2d_2a = ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2d_2b = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = MaxPoolBranch()
        self.conv2d_3b = ConvBlock(in_channels=64, out_channels=80, kernel_size=1, stride=1, padding=0)
        self.conv2d_4a = ConvBlock(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=0)
        self.conv2d_4b = ConvBlock(in_channels=192, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.repeat_1 = nn.Sequential(
            InceptionAUnit(bn_eps=bn_eps),
            InceptionAUnit(bn_eps=bn_eps),
            InceptionAUnit(bn_eps=bn_eps),
            InceptionAUnit(bn_eps=bn_eps),
            InceptionAUnit(bn_eps=bn_eps),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
            InceptionBUnit(bn_eps=bn_eps),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20, noReLU=False, bn_eps=bn_eps),
            Block8(scale=0.20, noReLU=False, bn_eps=bn_eps),
            Block8(scale=0.20, noReLU=False, bn_eps=bn_eps),
            Block8(scale=0.20, noReLU=False, bn_eps=bn_eps),
            Block8(scale=0.20, noReLU=False, bn_eps=bn_eps),
        )
        self.block8 = Block8(scale=1.0, noReLU=True, bn_eps=bn_eps)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout_prob)

        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        self.logits = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        x = self.logits(x)
        return x


def inceptionresnetv1(pretrained=False, **kwargs):
    return InceptionResnetV1(bn_eps=1e-3, **kwargs)


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
        inceptionresnetv1,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionresnetv1 or weight_count == 23995624)

        x = torch.randn(1, 3, 299, 299)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
