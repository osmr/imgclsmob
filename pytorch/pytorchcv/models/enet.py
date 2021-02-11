"""
    ENet for image segmentation, implemented in PyTorch.
    Original paper: 'ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1606.02147.
"""

__all__ = ['ENet', 'enet_cityscapes', 'ENetMixDownBlock']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv3x3, ConvBlock, AsymConvBlock, DeconvBlock, NormActivation, conv1x1_block


class ENetMaxDownBlock(nn.Module):
    """
    ENet specific max-pooling downscale block.

    Parameters:
    ----------
    ext_channels : int
        Number of extra channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 ext_channels,
                 kernel_size,
                 padding):
        super(ENetMaxDownBlock, self).__init__()
        self.ext_channels = ext_channels

        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            return_indices=True)

    def forward(self, x):
        x, max_indices = self.pool(x)
        branch, _, height, width = x.size()
        pad = torch.zeros(branch, self.ext_channels, height, width, dtype=x.dtype, device=x.device)
        x = torch.cat((x, pad), dim=1)
        return x, max_indices


class ENetUpBlock(nn.Module):
    """
    ENet upscale block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias):
        super(ENetUpBlock, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            activation=None)
        self.unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, max_indices):
        x = self.conv(x)
        x = self.unpool(x, max_indices)
        return x


class ENetUnit(nn.Module):
    """
    ENet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_asym_convs : bool
        Whether to use asymmetric convolution blocks.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bias : bool
        Whether the layer uses a bias vector.
    activation : function or str or None
        Activation function or name of activation function.
    downs : bool
        Whether to downscale or upscale.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 dilation,
                 use_asym_conv,
                 dropout_rate,
                 bias,
                 activation,
                 down,
                 bottleneck_factor=4):
        super(ENetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels)
        self.down = down
        mid_channels = in_channels // bottleneck_factor

        if not self.resize_identity:
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bias=bias,
                activation=activation)
            if use_asym_conv:
                self.conv2 = AsymConvBlock(
                    channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    lw_activation=activation,
                    rw_activation=activation)
            else:
                self.conv2 = ConvBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    activation=activation)
        elif self.down:
            self.identity_block = ENetMaxDownBlock(
                ext_channels=(out_channels - in_channels),
                kernel_size=kernel_size,
                padding=padding)
            self.conv1 = ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=1,
                bias=bias,
                activation=activation)
            self.conv2 = ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                bias=bias,
                activation=activation)
        else:
            self.identity_block = ENetUpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias)
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bias=bias,
                activation=activation)
            self.conv2 = DeconvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                out_padding=1,
                dilation=dilation,
                bias=bias,
                activation=activation)
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=bias,
            activation=activation)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.activ = activation()

    def forward(self, x, max_indices=None):
        if not self.resize_identity:
            identity = x
        elif self.down:
            identity, max_indices = self.identity_block(x)
        else:
            identity = self.identity_block(x, max_indices)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = x + identity
        x = self.activ(x)

        if self.resize_identity and self.down:
            return x, max_indices
        else:
            return x


class ENetStage(nn.Module):
    """
    ENet stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_sizes : list of int
        Kernel sizes.
    paddings : list of int
        Padding values.
    dilations : list of int
        Dilation values.
    use_asym_convs : list of int
        Whether to use asymmetric convolution blocks.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bias : bool
        Whether the layer uses a bias vector.
    activation : function or str or None
        Activation function or name of activation function.
    downs : bool
        Whether to downscale or upscale.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 paddings,
                 dilations,
                 use_asym_convs,
                 dropout_rate,
                 bias,
                 activation,
                 down):
        super(ENetStage, self).__init__()
        self.down = down

        units = nn.Sequential()
        for i, kernel_size in enumerate(kernel_sizes):
            unit = ENetUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=paddings[i],
                dilation=dilations[i],
                use_asym_conv=(use_asym_convs[i] == 1),
                dropout_rate=dropout_rate,
                bias=bias,
                activation=activation,
                down=down)
            if i == 0:
                self.scale_unit = unit
            else:
                units.add_module("unit{}".format(i + 1), unit)
            in_channels = out_channels
        self.units = units

    def forward(self, x, max_indices=None):
        if self.down:
            x, max_indices = self.scale_unit(x)
        else:
            x = self.scale_unit(x, max_indices)

        x = self.units(x)

        if self.down:
            return x, max_indices
        else:
            return x


class ENetMixDownBlock(nn.Module):
    """
    ENet specific mixed downscale block, used as an initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    correct_size_mistmatch : bool, default False
        Whether to correct downscaled sizes of images.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 correct_size_mismatch=False):
        super(ENetMixDownBlock, self).__init__()
        self.correct_size_mismatch = correct_size_mismatch

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)
        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=(out_channels - in_channels),
            stride=2,
            bias=bias)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=activation)

    def forward(self, x):
        y1 = self.pool(x)
        y2 = self.conv(x)

        if self.correct_size_mismatch:
            diff_h = y2.size()[2] - y1.size()[2]
            diff_w = y2.size()[3] - y1.size()[3]
            y1 = F.pad(y1, pad=(diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))

        x = torch.cat((y2, y1), dim=1)
        x = self.norm_activ(x)
        return x


class ENet(nn.Module):
    """
    ENet model from 'ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1606.02147.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for the first unit of each stage.
    init_block_channels : int
        Number of output channels for the initial unit.
    kernel_sizes : list of list of int
        Kernel sizes for each unit.
    paddings : list of list of int
        Padding values for each unit.
    dilations : list of list of int
        Dilation values for each unit.
    use_asym_convs : list of list of int
        Whether to use asymmetric convolution blocks for each unit.
    dropout_rates : list of float
        Parameter of dropout layer for each stage.
    downs : list of int
        Whether to downscale or upscale in each stage.
    correct_size_mistmatch : bool
        Whether to correct downscaled sizes of images in encoder.
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
                 channels,
                 init_block_channels,
                 kernel_sizes,
                 paddings,
                 dilations,
                 use_asym_convs,
                 dropout_rates,
                 downs,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(ENet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        bias = False
        encoder_activation = (lambda: nn.PReLU(1))
        decoder_activation = (lambda: nn.ReLU(inplace=True))

        self.stem = ENetMixDownBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bias=bias,
            bn_eps=bn_eps,
            activation=encoder_activation,
            correct_size_mismatch=correct_size_mismatch)
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            setattr(self, "stage{}".format(i + 1), ENetStage(
                in_channels=in_channels,
                out_channels=channels_per_stage,
                kernel_sizes=kernel_sizes[i],
                paddings=paddings[i],
                dilations=dilations[i],
                use_asym_convs=use_asym_convs[i],
                dropout_rate=dropout_rates[i],
                bias=bias,
                activation=(encoder_activation if downs[i] == 1 else decoder_activation),
                down=(downs[i] == 1)))
            in_channels = channels_per_stage

        self.head = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x, max_indices1 = self.stage1(x)
        x, max_indices2 = self.stage2(x)
        x = self.stage3(x, max_indices2)
        x = self.stage4(x, max_indices1)
        x = self.head(x)
        return x


def get_enet(model_name=None,
             pretrained=False,
             root=os.path.join("~", ".torch", "models"),
             **kwargs):
    """
    Create ENet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [64, 128, 64, 16]
    kernel_sizes = [[3, 3, 3, 3, 3], [3, 3, 3, 5, 3, 3, 3, 5, 3, 3, 3, 5, 3, 3, 3, 5, 3], [3, 3, 3], [3, 3]]
    paddings = [[1, 1, 1, 1, 1], [1, 1, 2, 2, 4, 1, 8, 2, 16, 1, 2, 2, 4, 1, 8, 2, 16], [1, 1, 1], [1, 1]]
    dilations = [[1, 1, 1, 1, 1], [1, 1, 2, 1, 4, 1, 8, 1, 16, 1, 2, 1, 4, 1, 8, 1, 16], [1, 1, 1], [1, 1]]
    use_asym_convs = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0], [0, 0]]
    dropout_rates = [0.01, 0.1, 0.1, 0.1]
    downs = [1, 1, 0, 0]
    init_block_channels = 16

    net = ENet(
        channels=channels,
        init_block_channels=init_block_channels,
        kernel_sizes=kernel_sizes,
        paddings=paddings,
        dilations=dilations,
        use_asym_convs=use_asym_convs,
        dropout_rates=dropout_rates,
        downs=downs,
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


def enet_cityscapes(num_classes=19, **kwargs):
    """
    ENet model for Cityscapes from 'ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation,'
    https://arxiv.org/abs/1606.02147.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_enet(num_classes=num_classes, model_name="enet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        enet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != enet_cityscapes or weight_count == 358060)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
