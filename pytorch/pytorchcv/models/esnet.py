"""
    ESNet for image segmentation, implemented in PyTorch.
    Original paper: 'ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1906.09826.
"""

__all__ = ['ESNet', 'esnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import AsymConvBlock, deconv3x3_block, Concurrent
from .enet import ENetMixDownBlock
from .erfnet import FCU


class PFCUBranch(nn.Module):
    """
    Parallel factorized convolution unit's branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    dilation : int
        Dilation value for convolution layer.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 dilation,
                 dropout_rate,
                 bn_eps):
        super(PFCUBranch, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=dilation,
            dilation=dilation,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class PFCU(nn.Module):
    """
    Parallel factorized convolution unit.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 dropout_rate,
                 bn_eps):
        super(PFCU, self).__init__()
        dilations = [2, 5, 9]
        padding = (kernel_size - 1) // 2

        self.conv1 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps)
        self.branches = Concurrent(merge_type="sum")
        for i, dilation in enumerate(dilations):
            self.branches.add_module("branch{}".format(i + 1), PFCUBranch(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bn_eps=bn_eps))
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.branches(x)

        x = x + identity
        x = self.activ(x)
        return x


class ESNet(nn.Module):
    """
    ESNet model from 'ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1906.09826.

    Parameters:
    ----------
    layers : list of list of int
        Number of layers in each stage of encoder and decoder.
    channels : list of list of int
        Number of output channels for each in encoder and decoder.
    kernel_sizes : list of list of int
        Kernel size for each in encoder and decoder.
    dropout_rates : list of list of int
        Dropout rates for each unit in encoder and decoder.
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
                 layers,
                 channels,
                 kernel_sizes,
                 dropout_rates,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(ESNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        self.encoder = nn.Sequential()
        for i, layers_per_stage in enumerate(layers[0]):
            out_channels = channels[0][i]
            kernel_size = kernel_sizes[0][i]
            dropout_rate = dropout_rates[0][i]
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), ENetMixDownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=True,
                        bn_eps=bn_eps,
                        correct_size_mismatch=correct_size_mismatch))
                    in_channels = out_channels
                elif i != len(layers[0]) - 1:
                    stage.add_module("unit{}".format(j + 1), FCU(
                        channels=in_channels,
                        kernel_size=kernel_size,
                        dilation=1,
                        dropout_rate=dropout_rate,
                        bn_eps=bn_eps))
                else:
                    stage.add_module("unit{}".format(j + 1), PFCU(
                        channels=in_channels,
                        kernel_size=kernel_size,
                        dropout_rate=dropout_rate,
                        bn_eps=bn_eps))
            self.encoder.add_module("stage{}".format(i + 1), stage)

        self.decoder = nn.Sequential()
        for i, layers_per_stage in enumerate(layers[1]):
            out_channels = channels[1][i]
            kernel_size = kernel_sizes[1][i]
            stage = nn.Sequential()
            for j in range(layers_per_stage):
                if j == 0:
                    stage.add_module("unit{}".format(j + 1), deconv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        bias=True,
                        bn_eps=bn_eps))
                    in_channels = out_channels
                else:
                    stage.add_module("unit{}".format(j + 1), FCU(
                        channels=in_channels,
                        kernel_size=kernel_size,
                        dilation=1,
                        dropout_rate=0,
                        bn_eps=bn_eps))
            self.decoder.add_module("stage{}".format(i + 1), stage)

        self.head = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x


def get_esnet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create ESNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    layers = [[4, 3, 4], [3, 3]]
    channels = [[16, 64, 128], [64, 16]]
    kernel_sizes = [[3, 5, 3], [5, 3]]
    dropout_rates = [[0.03, 0.03, 0.3], [0, 0]]
    bn_eps = 1e-3

    net = ESNet(
        layers=layers,
        channels=channels,
        kernel_sizes=kernel_sizes,
        dropout_rates=dropout_rates,
        bn_eps=bn_eps,
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


def esnet_cityscapes(num_classes=19, **kwargs):
    """
    ESNet model for Cityscapes from 'ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1906.09826.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_esnet(num_classes=num_classes, model_name="esnet_cityscapes", **kwargs)


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
    correct_size_mismatch = False
    in_size = (1024, 2048)
    classes = 19

    models = [
        esnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size,
                    correct_size_mismatch=correct_size_mismatch)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != esnet_cityscapes or weight_count == 1660607)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
