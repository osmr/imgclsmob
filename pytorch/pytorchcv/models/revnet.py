"""
    RevNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'The Reversible Residual Network: Backpropagation Without Storing Activations,'
    https://arxiv.org/abs/1707.04585.
"""

__all__ = ['RevNet', 'revnet38', 'revnet110', 'revnet164']

import os
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, pre_conv1x1_block, pre_conv3x3_block


use_context_mans = int(
    torch.__version__[0]) * 100 + int(torch.__version__[2]) - (1 if 'a' in torch.__version__ else 0) > 3


@contextmanager
def set_grad_enabled(grad_mode):
    if not use_context_mans:
        yield
    else:
        with torch.set_grad_enabled(grad_mode) as c:
            yield [c]


class ReversibleBlockFunction(torch.autograd.Function):
    """
    RevNet reversible block function.
    """

    @staticmethod
    def forward(ctx, x, fm, gm, *params):

        with torch.no_grad():
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
            x1 = x1.contiguous()
            x2 = x2.contiguous()

            y1 = x1 + fm(x2)
            y2 = x2 + gm(y1)

            y = torch.cat((y1, y2), dim=1)

            x1.set_()
            x2.set_()
            y1.set_()
            y2.set_()
            del x1, x2, y1, y2

        ctx.save_for_backward(x, y)
        ctx.fm = fm
        ctx.gm = gm

        return y

    @staticmethod
    def backward(ctx, grad_y):
        fm = ctx.fm
        gm = ctx.gm

        x, y = ctx.saved_variables
        y1, y2 = torch.chunk(y, chunks=2, dim=1)
        y1 = y1.contiguous()
        y2 = y2.contiguous()

        with torch.no_grad():
            y1_z = Variable(y1.data, requires_grad=True)
            x2 = y2 - gm(y1_z)
            x1 = y1 - fm(x2)

        with set_grad_enabled(True):
            x1_ = Variable(x1.data, requires_grad=True)
            x2_ = Variable(x2.data, requires_grad=True)
            y1_ = x1_ + fm.forward(x2_)
            y2_ = x2_ + gm(y1_)
            y = torch.cat((y1_, y2_), dim=1)

            dd = torch.autograd.grad(y, (x1_, x2_) + tuple(gm.parameters()) + tuple(fm.parameters()), grad_y)

            gm_params_len = len([p for p in gm.parameters()])
            gm_params_grads = dd[2:2 + gm_params_len]
            fm_params_grads = dd[2 + gm_params_len:]
            grad_x = torch.cat((dd[0], dd[1]), dim=1)

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_

        x.data.set_(torch.cat((x1, x2), dim=1).data.contiguous())

        return (grad_x, None, None) + fm_params_grads + gm_params_grads


class ReversibleBlock(nn.Module):
    """
    RevNet reversible block.

    Parameters:
    ----------
    fm : nn.Module
        Fm-function.
    gm : nn.Module
        Gm-function.
    """
    def __init__(self,
                 fm,
                 gm):
        super(ReversibleBlock, self).__init__()
        self.gm = gm
        self.fm = fm
        self.rev_funct = ReversibleBlockFunction.apply

    def forward(self, x):
        assert (x.shape[1] % 2 == 0)

        params = [w for w in self.fm.parameters()] + [w for w in self.gm.parameters()]
        y = self.rev_funct(x, self.fm, self.gm, *params)

        x.data.set_()

        return y

    def inverse(self, y):
        assert (y.shape[1] % 2 == 0)

        y1, y2 = torch.chunk(y, chunks=2, dim=1)
        y1 = y1.contiguous()
        y2 = y2.contiguous()

        x2 = y2 - self.gm(y1)
        x1 = y1 - self.fm(x2)

        x = torch.cat((x1, x2), dim=1)
        return x


class RevResBlock(nn.Module):
    """
    Simple RevNet block for residual path in RevNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 preactivate):
        super(RevResBlock, self).__init__()
        if preactivate:
            self.conv1 = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        else:
            self.conv1 = conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        self.conv2 = pre_conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RevResBottleneck(nn.Module):
    """
    RevNet bottleneck block for residual path in RevNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 preactivate,
                 bottleneck_factor=4):
        super(RevResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        if preactivate:
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
        else:
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = pre_conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class RevUnit(nn.Module):
    """
    RevNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck,
                 preactivate):
        super(RevUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        body_class = RevResBottleneck if bottleneck else RevResBlock

        if (not self.resize_identity) and (stride == 1):
            assert (in_channels % 2 == 0)
            assert (out_channels % 2 == 0)
            in_channels2 = in_channels // 2
            out_channels2 = out_channels // 2
            gm = body_class(
                in_channels=in_channels2,
                out_channels=out_channels2,
                stride=1,
                preactivate=preactivate)
            fm = body_class(
                in_channels=in_channels2,
                out_channels=out_channels2,
                stride=1,
                preactivate=preactivate)
            self.body = ReversibleBlock(gm, fm)
        else:
            self.body = body_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                preactivate=preactivate)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
            x = self.body(x)
            x = x + identity
        else:
            x = self.body(x)
        return x


class RevPostActivation(nn.Module):
    """
    RevNet specific post-activation block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(RevPostActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class RevNet(nn.Module):
    """
    RevNet model from 'The Reversible Residual Network: Backpropagation Without Storing Activations,'
    https://arxiv.org/abs/1707.04585.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
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
                 bottleneck,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(RevNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                preactivate = (j != 0) or (i != 0)
                stage.add_module("unit{}".format(j + 1), RevUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    preactivate=preactivate))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_postactiv", RevPostActivation(in_channels=in_channels))
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=56,
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


def get_revnet(blocks,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create RevNet model with specific parameters.

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

    if blocks == 38:
        layers = [3, 3, 3]
        channels_per_layers = [32, 64, 112]
        bottleneck = False
    elif blocks == 110:
        layers = [9, 9, 9]
        channels_per_layers = [32, 64, 128]
        bottleneck = False
    elif blocks == 164:
        layers = [9, 9, 9]
        channels_per_layers = [128, 256, 512]
        bottleneck = True
    else:
        raise ValueError("Unsupported RevNet with number of blocks: {}".format(blocks))

    init_block_channels = 32

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = RevNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
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


def revnet38(**kwargs):
    """
    RevNet-38 model from 'The Reversible Residual Network: Backpropagation Without Storing Activations,'
    https://arxiv.org/abs/1707.04585.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_revnet(blocks=38, model_name="revnet38", **kwargs)


def revnet110(**kwargs):
    """
    RevNet-110 model from 'The Reversible Residual Network: Backpropagation Without Storing Activations,'
    https://arxiv.org/abs/1707.04585.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_revnet(blocks=110, model_name="revnet110", **kwargs)


def revnet164(**kwargs):
    """
    RevNet-164 model from 'The Reversible Residual Network: Backpropagation Without Storing Activations,'
    https://arxiv.org/abs/1707.04585.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_revnet(blocks=164, model_name="revnet164", **kwargs)


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
        revnet38,
        revnet110,
        revnet164,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != revnet38 or weight_count == 685864)
        assert (model != revnet110 or weight_count == 1982600)
        assert (model != revnet164 or weight_count == 2491656)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
