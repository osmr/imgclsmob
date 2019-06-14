"""
    DiracNetV2 for ImageNet-1K, implemented in Gluon.
    Original paper: 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.
"""

__all__ = ['DiracNetV2', 'diracnet18v2', 'diracnet34v2']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class DiracConv(HybridBlock):
    """
    DiracNetV2 specific convolution block with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 **kwargs):
        super(DiracConv, self).__init__(**kwargs)
        with self.name_scope():
            self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=True,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.activ(x)
        x = self.conv(x)
        return x


def dirac_conv3x3(in_channels,
                  out_channels):
    """
    3x3 version of the DiracNetV2 specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DiracConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1)


class DiracInitBlock(HybridBlock):
    """
    DiracNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(DiracInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=7,
                strides=2,
                padding=3,
                use_bias=True,
                in_channels=in_channels)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DiracNetV2(HybridBlock):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(DiracNetV2, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(DiracInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(dirac_conv3x3(
                            in_channels=in_channels,
                            out_channels=out_channels))
                        in_channels = out_channels
                    if i != len(channels) - 1:
                        stage.add(nn.MaxPool2D(
                            pool_size=2,
                            strides=2,
                            padding=0))
                self.features.add(stage)
            self.features.add(nn.Activation("relu"))
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_diracnetv2(blocks,
                   model_name=None,
                   pretrained=False,
                   ctx=cpu(),
                   root=os.path.join("~", ".mxnet", "models"),
                   **kwargs):
    """
    Create DiracNetV2 model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if blocks == 18:
        layers = [4, 4, 4, 4]
    elif blocks == 34:
        layers = [6, 8, 12, 6]
    else:
        raise ValueError("Unsupported DiracNetV2 with number of blocks: {}".format(blocks))

    channels_per_layers = [64, 128, 256, 512]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    init_block_channels = 64

    net = DiracNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def diracnet18v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=18, model_name="diracnet18v2", **kwargs)


def diracnet34v2(**kwargs):
    """
    DiracNetV2 model from 'DiracNets: Training Very Deep Neural Networks Without Skip-Connections,'
    https://arxiv.org/abs/1706.00388.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diracnetv2(blocks=34, model_name="diracnet34v2", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        diracnet18v2,
        diracnet34v2,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diracnet18v2 or weight_count == 11511784)
        assert (model != diracnet34v2 or weight_count == 21616232)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
