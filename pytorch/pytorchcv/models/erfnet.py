"""
    ERFNet for image segmentation, implemented in PyTorch.
    Original paper: 'ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation,'
    http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf.
"""

__all__ = ['ERFNet', 'erfnet_cityscapes', 'FCU']

import os
import torch
import torch.nn as nn
from .common import deconv3x3_block, AsymConvBlock
from .enet import ENetMixDownBlock


class FCU(nn.Module):
    """
    Factorized convolution unit.

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
        super(FCU, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        padding1 = (kernel_size - 1) // 2
        padding2 = padding1 * dilation

        self.conv1 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding1,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps)
        self.conv2 = AsymConvBlock(
            channels=channels,
            kernel_size=kernel_size,
            padding=padding2,
            dilation=dilation,
            bias=True,
            lw_use_bn=False,
            bn_eps=bn_eps,
            rw_activation=None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)

        x = x + identity
        x = self.activ(x)
        return x


class ERFNet(nn.Module):
    """
    ERFNet model from 'ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation,'
    http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for the first unit of each stage.
    dilations : list of list of int
        Dilation values for each unit.
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
                 dilations,
                 dropout_rates,
                 downs,
                 correct_size_mismatch=False,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(ERFNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        bias = True

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        enc_idx = 0
        dec_idx = 0
        for i, out_channels in enumerate(channels):
            dilations_per_stage = dilations[i]
            dropout_rates_per_stage = dropout_rates[i]
            is_down = downs[i]
            stage = nn.Sequential()
            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    if is_down:
                        unit = ENetMixDownBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            bias=bias,
                            bn_eps=bn_eps,
                            correct_size_mismatch=correct_size_mismatch)
                    else:
                        unit = deconv3x3_block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            stride=2,
                            bias=bias,
                            bn_eps=bn_eps)
                else:
                    unit = FCU(
                        channels=in_channels,
                        kernel_size=3,
                        dilation=dilation,
                        dropout_rate=dropout_rates_per_stage[j],
                        bn_eps=bn_eps)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            if is_down:
                enc_idx += 1
                self.encoder.add_module("stage{}".format(enc_idx), stage)
            else:
                dec_idx += 1
                self.decoder.add_module("stage{}".format(dec_idx), stage)

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


def get_erfnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create ERFNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    downs = [1, 1, 1, 0, 0]
    channels = [16, 64, 128, 64, 16]
    dilations = [[1], [1, 1, 1, 1, 1, 1], [1, 2, 4, 8, 16, 2, 4, 8, 16], [1, 1, 1], [1, 1, 1]]
    dropout_rates = [[0.0], [0.03, 0.03, 0.03, 0.03, 0.03, 0.03], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                     [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    net = ERFNet(
        channels=channels,
        dilations=dilations,
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


def erfnet_cityscapes(num_classes=19, **kwargs):
    """
    ERFNet model for Cityscapes from 'ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic
    Segmentation,' http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_erfnet(num_classes=num_classes, model_name="erfnet_cityscapes", **kwargs)


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
        erfnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != erfnet_cityscapes or weight_count == 2064191)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
