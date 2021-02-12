"""
    SQNet for image segmentation, implemented in PyTorch.
    Original paper: 'Speeding up Semantic Segmentation for Autonomous Driving,'
    https://openreview.net/pdf?id=S1uHiFyyg.
"""

__all__ = ['SQNet', 'sqnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1_block, conv3x3_block, deconv3x3_block, Concurrent, Hourglass


class FireBlock(nn.Module):
    """
    SQNet specific encoder block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias,
                 use_bn,
                 activation):
        super(FireBlock, self).__init__()
        squeeze_channels = out_channels // 8
        expand_channels = out_channels // 2

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            bias=bias,
            use_bn=use_bn,
            activation=activation)
        self.branches = Concurrent(merge_type="cat")
        self.branches.add_module("branch1", conv1x1_block(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            bias=bias,
            use_bn=use_bn,
            activation=None))
        self.branches.add_module("branch2", conv3x3_block(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            bias=bias,
            use_bn=use_bn,
            activation=None))
        self.activ = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.branches(x)
        x = self.activ(x)
        return x


class ParallelDilatedConv(nn.Module):
    """
    SQNet specific decoder block (parallel dilated convolution).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias,
                 use_bn,
                 activation):
        super(ParallelDilatedConv, self).__init__()
        dilations = [1, 2, 3, 4]

        self.branches = Concurrent(merge_type="sum")
        for i, dilation in enumerate(dilations):
            self.branches.add_module("branch{}".format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=dilation,
                dilation=dilation,
                bias=bias,
                use_bn=use_bn,
                activation=activation))

    def forward(self, x):
        x = self.branches(x)
        return x


class SQNetUpStage(nn.Module):
    """
    SQNet upscale stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    use_parallel_conv : bool
        Whether to use parallel dilated convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias,
                 use_bn,
                 activation,
                 use_parallel_conv):
        super(SQNetUpStage, self).__init__()

        if use_parallel_conv:
            self.conv = ParallelDilatedConv(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=bias,
                use_bn=use_bn,
                activation=activation)
        else:
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=bias,
                use_bn=use_bn,
                activation=activation)
        self.deconv = deconv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bias=bias,
            use_bn=use_bn,
            activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x


class SQNet(nn.Module):
    """
    SQNet model from 'Speeding up Semantic Segmentation for Autonomous Driving,'
    https://openreview.net/pdf?id=S1uHiFyyg.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each stage in encoder and decoder.
    init_block_channels : int
        Number of output channels for the initial unit.
    layers : list of int
        Number of layers for each stage in encoder.
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
                 layers,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(SQNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size
        bias = True
        use_bn = False
        activation = (lambda: nn.ELU(inplace=True))

        self.stem = conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bias=bias,
            use_bn=use_bn,
            activation=activation)
        in_channels = init_block_channels

        down_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[0]):
            skip_seq.add_module("skip{}".format(i + 1), conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bias=bias,
                use_bn=use_bn,
                activation=activation))
            stage = nn.Sequential()
            stage.add_module("unit1", nn.MaxPool2d(
                kernel_size=2,
                stride=2))
            for j in range(layers[i]):
                stage.add_module("unit{}".format(j + 2), FireBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bias=bias,
                    use_bn=use_bn,
                    activation=activation))
                in_channels = out_channels
            down_seq.add_module("down{}".format(i + 1), stage)

        in_channels = in_channels // 2

        up_seq = nn.Sequential()
        for i, out_channels in enumerate(channels[1]):
            use_parallel_conv = True if i == 0 else False
            up_seq.add_module("up{}".format(i + 1), SQNetUpStage(
                in_channels=(2 * in_channels),
                out_channels=out_channels,
                bias=bias,
                use_bn=use_bn,
                activation=activation,
                use_parallel_conv=use_parallel_conv))
            in_channels = out_channels
        up_seq = up_seq[::-1]

        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            merge_type="cat")

        self.head = SQNetUpStage(
            in_channels=(2 * in_channels),
            out_channels=num_classes,
            bias=bias,
            use_bn=use_bn,
            activation=activation,
            use_parallel_conv=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.hg(x)
        x = self.head(x)
        return x


def get_sqnet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create SQNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [[128, 256, 512], [256, 128, 96]]
    init_block_channels = 96
    layers = [2, 2, 3]

    net = SQNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def sqnet_cityscapes(num_classes=19, **kwargs):
    """
    SQNet model for Cityscapes from 'Speeding up Semantic Segmentation for Autonomous Driving,'
    https://openreview.net/pdf?id=S1uHiFyyg.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_sqnet(num_classes=num_classes, model_name="sqnet_cityscapes", **kwargs)


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
        sqnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != sqnet_cityscapes or weight_count == 16262771)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
