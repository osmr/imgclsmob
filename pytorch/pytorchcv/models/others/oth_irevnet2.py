"""
    i-RevNet, implemented in PyTorch.
    Original paper: 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.
"""

__all__ = ['oth_irevnet301']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PreConvBlock(nn.Module):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

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
    bias : bool, default False
        Whether the layer uses a bias vector.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
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
                 bias=False,
                 return_preact=False,
                 activate=True):
        super(PreConvBlock, self).__init__()
        self.return_preact = return_preact
        self.activate = activate

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

    def forward(self, x):
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv3x3_block(in_channels,
                      out_channels,
                      stride=1,
                      padding=1,
                      dilation=1,
                      return_preact=False,
                      activate=True):
    """
    3x3 version of the pre-activated convolution block.

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
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_preact=return_preact,
        activate=activate)


class RevPostActivation(nn.Module):
    """
    iRevNet post-activation block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(RevPostActivation, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features=in_channels,
            momentum=0.9)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


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


class RevDualPathSequential(DualPathSequential):
    """
    An invertible sequential container for modules with dual inputs/outputs.
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
                 dual_path_scheme_ordinal=(lambda module, x1, x2: (module(x1), x2)),
                 last_noninvertible=0):
        super(RevDualPathSequential, self).__init__(
            return_two=return_two,
            first_ordinals=first_ordinals,
            last_ordinals=last_ordinals,
            dual_path_scheme=dual_path_scheme,
            dual_path_scheme_ordinal=dual_path_scheme_ordinal)
        self.last_noninvertible = last_noninvertible

    def inverse(self, x1, x2=None):
        length = len(self._modules.values())
        for i, module in enumerate(reversed(self._modules.values())):
            if i < self.last_noninvertible:
                pass
            elif (i < self.last_ordinals) or (i >= length - self.first_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module.inverse, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module.inverse, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class RevInjectivePad(nn.Module):
    def __init__(self, pad_size):
        super(RevInjectivePad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class RevDownscale(nn.Module):
    def __init__(self, scale):
        super(RevDownscale, self).__init__()
        self.scale = scale

    def forward(self, x):
        batch, x_channels, x_height, x_width = x.size()
        y_channels = x_channels * self.scale * self.scale
        assert (x_height % self.scale == 0)
        y_height = x_height // self.scale

        y = x.permute(0, 2, 3, 1)
        d2_split_seq = y.split(split_size=self.scale, dim=2)
        d2_split_seq = [t.contiguous().view(batch, y_height, y_channels) for t in d2_split_seq]
        y = torch.stack(d2_split_seq, dim=1)
        y = y.permute(0, 3, 2, 1)
        return y.contiguous()

    def inverse(self, y):
        scale_sqr = self.scale * self.scale
        batch, y_channels, y_height, y_width = y.size()
        assert (y_channels % scale_sqr == 0)
        x_channels = y_channels // scale_sqr
        x_height = y_height * self.scale
        x_width = y_width * self.scale

        x = y.permute(0, 2, 3, 1)
        x = x.contiguous().view(batch, y_height, y_width, scale_sqr, x_channels)
        d3_split_seq = x.split(split_size=self.scale, dim=3)
        d3_split_seq = [t.contiguous().view(batch, y_height, x_width, x_channels) for t in d3_split_seq]
        x = torch.stack(d3_split_seq, dim=0)
        x = x.transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch, x_height, x_width, x_channels)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class RevBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 preactivate):
        super(RevBottleneck, self).__init__()
        mid_channels = out_channels // 4

        if preactivate:
            self.conv1 = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride)
        else:
            self.conv1 = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.conv3 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class RevUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 preactivate=True):
        super(RevUnit, self).__init__()
        if not preactivate:
            in_channels = in_channels // 2

        padding = 2 * (out_channels - in_channels)
        self.do_padding = (padding != 0) and (stride == 1)
        self.do_downscale = (stride != 1)

        if self.do_padding:
            self.inj_pad = RevInjectivePad(padding)
        self.bottleneck_block = RevBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            preactivate=preactivate)
        if self.do_downscale:
            self.psi = RevDownscale(stride)

    def forward(self, x1, x2):
        if self.do_padding:
            x = torch.cat((x1, x2), dim=1)
            x = self.inj_pad.forward(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
        fx2 = self.bottleneck_block(x2)
        if self.do_downscale:
            x1 = self.psi(x1)
            x2 = self.psi(x2)
        y1 = fx2 + x1
        return x2, y1

    def inverse(self, x2, y1):
        if self.do_downscale:
            x2 = self.psi.inverse(x2)
        fx2 = - self.bottleneck_block(x2)
        x1 = fx2 + y1
        if self.do_downscale:
            x1 = self.psi.inverse(x1)
        if self.do_padding:
            x = torch.cat((x1, x2), dim=1)
            x = self.inj_pad.inverse(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
        return x1, x2


class iRevSplitBlock(nn.Module):
    """
    iRevNet split block.
    """
    def __init__(self):
        super(iRevSplitBlock, self).__init__()

    def forward(self, x, _):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        return x1, x2

    def inverse(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x, None


class iRevMergeBlock(nn.Module):
    """
    iRevNet merge block.
    """
    def __init__(self):
        super(iRevMergeBlock, self).__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x, x

    def inverse(self, x, _):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        return x1, x2


class iRevNet(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(iRevNet, self).__init__()
        assert (in_channels > 0)
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = RevDualPathSequential(
            first_ordinals=1,
            last_ordinals=2,
            last_noninvertible=2)
        self.features.add_module("init_block", RevDownscale(scale=2))
        in_channels = init_block_channels
        self.features.add_module("init_split", iRevSplitBlock())
        for i, channels_per_stage in enumerate(channels):
            stage = RevDualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                preactivate = not ((i == 0) and (j == 0))
                stage.add_module("unit{}".format(j + 1), RevUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    preactivate=preactivate))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        in_channels = final_block_channels
        self.features.add_module("final_merge", iRevMergeBlock())
        self.features.add_module("final_postactiv", RevPostActivation(in_channels=in_channels))
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

    def forward(self, x, return_out_bij=False):
        x, out_bij = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        if return_out_bij:
            return x, out_bij
        else:
            return x

    def inverse(self, out_bij):
        x, _ = self.features.inverse(out_bij)
        return x


def get_irevnet(blocks,
                model_name=None,
                pretrained=False,
                root=os.path.join('~', '.torch', 'models'),
                **kwargs):
    """
    Create i-RevNet model with specific parameters.

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
    if blocks == 301:
        layers = [6, 16, 72, 6]
    else:
        raise ValueError("Unsupported i-RevNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 1 == blocks)

    channels_per_layers = [24, 96, 384, 1536]
    init_block_channels = 12
    final_block_channels = 3072

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = iRevNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)

    return net


def oth_irevnet301(pretrained=False, **kwargs):
    return get_irevnet(blocks=301)


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
        oth_irevnet301,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_irevnet301 or weight_count == 125120356)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))
        y, out_bij = net(x, return_out_bij=True)
        x_ = net.inverse(out_bij)
        assert (tuple(x_.size()) == (1, 3, 224, 224))


if __name__ == "__main__":
    _test()
