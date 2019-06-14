"""
    ResAttNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.
"""

__all__ = ['ResAttNet', 'resattnet56', 'resattnet92', 'resattnet128', 'resattnet164', 'resattnet200', 'resattnet236',
           'resattnet452']

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .common import conv1x1, conv7x7_block, pre_conv1x1_block, pre_conv3x3_block, Hourglass


class PreResBottleneck(nn.Module):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            return_preact=True)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = pre_conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x, x_pre_activ


class ResBlock(nn.Module):
    """
    Residual block with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(ResBlock, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = PreResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class InterpolationBlock(nn.Module):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : float
        Multiplier for spatial size.
    """
    def __init__(self,
                 scale_factor):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            input=x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True)


class DoubleSkipBlock(nn.Module):
    """
    Double skip connection block.

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
        super(DoubleSkipBlock, self).__init__()
        self.skip1 = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = x + self.skip1(x)
        return x


class ResBlockSequence(nn.Module):
    """
    Sequence of residual blocks with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of sequence.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length):
        super(ResBlockSequence, self).__init__()
        self.blocks = nn.Sequential()
        for i in range(length):
            self.blocks.add_module('block{}'.format(i + 1), ResBlock(
                in_channels=in_channels,
                out_channels=out_channels))

    def forward(self, x):
        x = self.blocks(x)
        return x


class DownAttBlock(nn.Module):
    """
    Down sub-block for hourglass of attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of residual blocks list.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length):
        super(DownAttBlock, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)
        self.res_blocks = ResBlockSequence(
            in_channels=in_channels,
            out_channels=out_channels,
            length=length)

    def forward(self, x):
        x = self.pool(x)
        x = self.res_blocks(x)
        return x


class UpAttBlock(nn.Module):
    """
    Up sub-block for hourglass of attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of residual blocks list.
    scale_factor : float
        Multiplier for spatial size.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length,
                 scale_factor):
        super(UpAttBlock, self).__init__()
        self.res_blocks = ResBlockSequence(
            in_channels=in_channels,
            out_channels=out_channels,
            length=length)
        self.upsample = InterpolationBlock(scale_factor)

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


class MiddleAttBlock(nn.Module):
    """
    Middle sub-block for attention block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    """
    def __init__(self,
                 channels):
        super(MiddleAttBlock, self).__init__()
        self.conv1 = pre_conv1x1_block(
            in_channels=channels,
            out_channels=channels)
        self.conv2 = pre_conv1x1_block(
            in_channels=channels,
            out_channels=channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class AttBlock(nn.Module):
    """
    Attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hourglass_depth : int
        Depth of hourglass block.
    att_scales : list of int
        Attention block specific scales.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hourglass_depth,
                 att_scales):
        super(AttBlock, self).__init__()
        assert (len(att_scales) == 3)
        scale_factor = 2
        scale_p, scale_t, scale_r = att_scales

        self.init_blocks = ResBlockSequence(
            in_channels=in_channels,
            out_channels=out_channels,
            length=scale_p)

        down_seq = nn.Sequential()
        up_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i in range(hourglass_depth):
            down_seq.add_module('down{}'.format(i + 1), DownAttBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                length=scale_r))
            up_seq.add_module('up{}'.format(i + 1), UpAttBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                length=scale_r,
                scale_factor=scale_factor))
            if i == 0:
                skip_seq.add_module('skip1', ResBlockSequence(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    length=scale_t))
            else:
                skip_seq.add_module('skip{}'.format(i + 1), DoubleSkipBlock(
                    in_channels=in_channels,
                    out_channels=out_channels))
        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            return_first_skip=True)

        self.middle_block = MiddleAttBlock(channels=out_channels)
        self.final_block = ResBlock(
            in_channels=in_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.init_blocks(x)
        x, y = self.hg(x)
        x = self.middle_block(x)
        x = (1 + x) * y
        x = self.final_block(x)
        return x


class ResAttInitBlock(nn.Module):
    """
    ResAttNet specific initial block.

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
        super(ResAttInitBlock, self).__init__()
        self.conv = conv7x7_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PreActivation(nn.Module):
    """
    Pre-activation block without convolution layer. It's used by itself as the final block in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PreActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ResAttNet(nn.Module):
    """
    ResAttNet model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    attentions : list of list of int
        Whether to use a attention unit or residual one.
    att_scales : list of int
        Attention block specific scales.
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
                 attentions,
                 att_scales,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ResAttNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResAttInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            hourglass_depth = len(channels) - 1 - i
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 1 if (i == 0) or (j != 0) else 2
                if attentions[i][j]:
                    stage.add_module("unit{}".format(j + 1), AttBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        hourglass_depth=hourglass_depth,
                        att_scales=att_scales))
                else:
                    stage.add_module("unit{}".format(j + 1), ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('post_activ', PreActivation(in_channels=in_channels))
        self.features.add_module('final_pool', nn.AvgPool2d(
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


def get_resattnet(blocks,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create ResAttNet model with specific parameters.

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
    if blocks == 56:
        att_layers = [1, 1, 1]
        att_scales = [1, 2, 1]
    elif blocks == 92:
        att_layers = [1, 2, 3]
        att_scales = [1, 2, 1]
    elif blocks == 128:
        att_layers = [2, 3, 4]
        att_scales = [1, 2, 1]
    elif blocks == 164:
        att_layers = [3, 4, 5]
        att_scales = [1, 2, 1]
    elif blocks == 200:
        att_layers = [4, 5, 6]
        att_scales = [1, 2, 1]
    elif blocks == 236:
        att_layers = [5, 6, 7]
        att_scales = [1, 2, 1]
    elif blocks == 452:
        att_layers = [5, 6, 7]
        att_scales = [2, 4, 3]
    else:
        raise ValueError("Unsupported ResAttNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    layers = att_layers + [2]
    channels = [[ci] * (li + 1) for (ci, li) in zip(channels_per_layers, layers)]
    attentions = [[0] + [1] * li for li in att_layers] + [[0] * 3]

    net = ResAttNet(
        channels=channels,
        init_block_channels=init_block_channels,
        attentions=attentions,
        att_scales=att_scales,
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


def resattnet56(**kwargs):
    """
    ResAttNet-56 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=56, model_name="resattnet56", **kwargs)


def resattnet92(**kwargs):
    """
    ResAttNet-92 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=92, model_name="resattnet92", **kwargs)


def resattnet128(**kwargs):
    """
    ResAttNet-128 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=128, model_name="resattnet128", **kwargs)


def resattnet164(**kwargs):
    """
    ResAttNet-164 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=164, model_name="resattnet164", **kwargs)


def resattnet200(**kwargs):
    """
    ResAttNet-200 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=200, model_name="resattnet200", **kwargs)


def resattnet236(**kwargs):
    """
    ResAttNet-236 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=236, model_name="resattnet236", **kwargs)


def resattnet452(**kwargs):
    """
    ResAttNet-452 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=452, model_name="resattnet452", **kwargs)


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
        resattnet56,
        resattnet92,
        resattnet128,
        resattnet164,
        resattnet200,
        resattnet236,
        resattnet452,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resattnet56 or weight_count == 31810728)
        assert (model != resattnet92 or weight_count == 52466344)
        assert (model != resattnet128 or weight_count == 65294504)
        assert (model != resattnet164 or weight_count == 78122664)
        assert (model != resattnet200 or weight_count == 90950824)
        assert (model != resattnet236 or weight_count == 103778984)
        assert (model != resattnet452 or weight_count == 182285224)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
