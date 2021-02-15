"""
    ESPNet for image segmentation, implemented in PyTorch.
    Original paper: 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation,'
    https://arxiv.org/abs/1803.06815.
"""

__all__ = ['ESPNet', 'espnet_cityscapes']

import os
import torch
import torch.nn as nn
from common import conv1x1, conv3x3_block, NormActivation, DeconvBlock
from espcnet import ESPCNet, ESPBlock


class ESPFinalBlock(nn.Module):
    """
    ESPNet final block.

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
        super(ESPFinalBlock, self).__init__()
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))
        self.deconv = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


class ESPNet(ESPCNet):
    """
    ESPNet model from 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation,'
    https://arxiv.org/abs/1803.06815.

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    cut_x : list of int
        Whether to concatenate with x-branch for each unit.
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
                 init_block_channels,
                 cut_x,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(ESPNet, self).__init__(
            layers=layers,
            channels=channels,
            init_block_channels=init_block_channels,
            cut_x=cut_x,
            bn_eps=bn_eps,
            aux=aux,
            fixed_size=fixed_size,
            in_channels=in_channels,
            in_size=in_size,
            num_classes=num_classes)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        self.skip1 = nn.BatchNorm2d(
            num_features=num_classes,
            eps=bn_eps)
        self.skip2 = conv1x1(
            in_channels=channels[1],
            out_channels=num_classes)

        self.up1 = nn.Sequential(nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False))

        self.up2 = nn.Sequential()
        self.up2.add_module("block1", NormActivation(
            in_channels=(2 * num_classes),
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(2 * num_classes))))
        self.up2.add_module("block2", ESPBlock(
                in_channels=(2 * num_classes),
                out_channels=num_classes,
                downsample=False,
                residual=False,
                bn_eps=bn_eps))
        self.up2.add_module("block3", DeconvBlock(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=2,
            stride=2,
            padding=0,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(num_classes))))

        self.decoder_head = ESPFinalBlock(
            in_channels=(channels[0] + num_classes),
            out_channels=num_classes,
            bn_eps=bn_eps)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        y0 = self.features.init_block(x)
        y1, x = self.features.stage1(y0, x)
        y2, x = self.features.stage2(y1, x)
        y3, x = self.features.stage3(y2, x)
        yh = self.head(y3)

        v1 = self.skip1(yh)
        z1 = self.up1(v1)
        v2 = self.skip2(y2)
        z2 = torch.cat((v2, z1), dim=1)
        z2 = self.up2(z2)
        z = torch.cat((z2, y1), dim=1)
        z = self.decoder_head(z)
        return z


def get_espnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create ESPNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 16
    layers = [0, 3, 4]
    channels = [19, 131, 256]
    cut_x = [1, 1, 0]
    bn_eps = 1e-3

    net = ESPNet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        cut_x=cut_x,
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


def espnet_cityscapes(num_classes=19, **kwargs):
    """
    ESPNet model for Cityscapes from 'ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic
    Segmentation,' https://arxiv.org/abs/1803.06815.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_espnet(num_classes=num_classes, model_name="espnet_cityscapes", **kwargs)


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
        espnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != espnet_cityscapes or weight_count == 201542)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
