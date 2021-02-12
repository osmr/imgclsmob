"""
    SegNet for image segmentation, implemented in PyTorch.
    Original paper: 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,'
    https://arxiv.org/abs/1511.00561.
"""

__all__ = ['SegNet', 'segnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv3x3, conv3x3_block, DualPathSequential


class SegNet(nn.Module):
    """
    SegNet model from 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,'
    https://arxiv.org/abs/1511.00561.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each stage in encoder and decoder.
    layers : list of list of int
        Number of layers for each stage in encoder and decoder.
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
                 layers,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(SegNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        bias = True

        for i, out_channels in enumerate(channels[0]):
            stage = nn.Sequential()
            for j in range(layers[0][i]):
                if j == layers[0][i] - 1:
                    unit = nn.MaxPool2d(
                        kernel_size=2,
                        stride=2,
                        return_indices=True)
                else:
                    unit = conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=bias)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            setattr(self, "down_stage{}".format(i + 1), stage)

        for i, channels_per_stage in enumerate(channels[1]):
            stage = DualPathSequential(
                return_two=False,
                last_ordinals=(layers[1][i] - 1),
                dual_path_scheme=(lambda module, x1, x2: (module(x1, x2), x2)))
            for j in range(layers[1][i]):
                if j == layers[1][i] - 1:
                    out_channels = channels_per_stage
                else:
                    out_channels = in_channels
                if j == 0:
                    unit = nn.MaxUnpool2d(
                        kernel_size=2,
                        stride=2)
                else:
                    unit = conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=bias)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            setattr(self, "up_stage{}".format(i + 1), stage)

        self.head = conv3x3(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=bias)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x, max_indices1 = self.down_stage1(x)
        x, max_indices2 = self.down_stage2(x)
        x, max_indices3 = self.down_stage3(x)
        x, max_indices4 = self.down_stage4(x)
        x, max_indices5 = self.down_stage5(x)

        x = self.up_stage1(x, max_indices5)
        x = self.up_stage2(x, max_indices4)
        x = self.up_stage3(x, max_indices3)
        x = self.up_stage4(x, max_indices2)
        x = self.up_stage5(x, max_indices1)

        x = self.head(x)
        return x


def get_segnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create SegNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [[64, 128, 256, 512, 512], [512, 256, 128, 64, 64]]
    layers = [[3, 3, 4, 4, 4], [4, 4, 4, 3, 2]]

    net = SegNet(
        channels=channels,
        layers=layers,
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


def segnet_cityscapes(num_classes=19, **kwargs):
    """
    SegNet model for Cityscapes from 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,'
    https://arxiv.org/abs/1511.00561.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_segnet(num_classes=num_classes, model_name="segnet_cityscapes", **kwargs)


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
        segnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != segnet_cityscapes or weight_count == 29453971)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
