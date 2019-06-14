"""
    Xception for ImageNet-1K, implemented in Gluon.
    Original paper: 'Xception: Deep Learning with Depthwise Separable Convolutions,' https://arxiv.org/abs/1610.02357.
"""

__all__ = ['Xception', 'xception']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block


class DwsConv(HybridBlock):
    """
    Depthwise separable convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 **kwargs):
        super(DwsConv, self).__init__(**kwargs)
        with self.name_scope():
            self.dw_conv = nn.Conv2D(
                channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                groups=in_channels,
                use_bias=False,
                in_channels=in_channels)
            self.pw_conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=1,
                use_bias=False,
                in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DwsConvBlock(HybridBlock):
    """
    Depthwise separable convolution block with batchnorm and ReLU pre-activation.

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
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 bn_use_global_stats,
                 activate,
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        self.activate = activate

        with self.name_scope():
            if self.activate:
                self.activ = nn.Activation("relu")
            self.conv = DwsConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        if self.activate:
            x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def dws_conv3x3_block(in_channels,
                      out_channels,
                      bn_use_global_stats,
                      activate):
    """
    3x3 version of the depthwise separable convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activate : bool
        Whether activate the convolution block.
    """
    return DwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1,
        bn_use_global_stats=bn_use_global_stats,
        activate=activate)


class XceptionUnit(HybridBlock):
    """
    Xception unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the downsample polling.
    reps : int
        Number of repetitions.
    start_with_relu : bool, default True
        Whether start with ReLU activation.
    grow_first : bool, default True
        Whether start from growing.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 reps,
                 start_with_relu=True,
                 grow_first=True,
                 bn_use_global_stats=False,
                 **kwargs):
        super(XceptionUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)

            self.body = nn.HybridSequential(prefix="")
            for i in range(reps):
                if (grow_first and (i == 0)) or ((not grow_first) and (i == reps - 1)):
                    in_channels_i = in_channels
                    out_channels_i = out_channels
                else:
                    if grow_first:
                        in_channels_i = out_channels
                        out_channels_i = out_channels
                    else:
                        in_channels_i = in_channels
                        out_channels_i = in_channels
                activate = start_with_relu if (i == 0) else True
                self.body.add(dws_conv3x3_block(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i,
                    bn_use_global_stats=bn_use_global_stats,
                    activate=activate))
            if strides != 1:
                self.body.add(nn.MaxPool2D(
                    pool_size=3,
                    strides=strides,
                    padding=1))

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = F.identity(x)
        x = self.body(x)
        x = x + identity
        return x


class XceptionInitBlock(HybridBlock):
    """
    Xception specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(XceptionInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=32,
                strides=2,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=32,
                out_channels=64,
                strides=1,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class XceptionFinalBlock(HybridBlock):
    """
    Xception specific final block.

    Parameters:
    ----------
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_use_global_stats,
                 **kwargs):
        super(XceptionFinalBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = dws_conv3x3_block(
                in_channels=1024,
                out_channels=1536,
                bn_use_global_stats=bn_use_global_stats,
                activate=False)
            self.conv2 = dws_conv3x3_block(
                in_channels=1536,
                out_channels=2048,
                bn_use_global_stats=bn_use_global_stats,
                activate=True)
            self.activ = nn.Activation("relu")
            self.pool = nn.AvgPool2D(
                pool_size=10,
                strides=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


class Xception(HybridBlock):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 **kwargs):
        super(Xception, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(XceptionInitBlock(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = 64
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(XceptionUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=(2 if (j == 0) else 1),
                            reps=(2 if (j == 0) else 3),
                            start_with_relu=((i != 0) or (j != 0)),
                            grow_first=((i != len(channels) - 1) or (j != len(channels_per_stage) - 1)),
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(XceptionFinalBlock(bn_use_global_stats=bn_use_global_stats))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=2048))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_xception(model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    """
    Create Xception model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    channels = [[128], [256], [728] * 9, [1024]]

    net = Xception(
        channels=channels,
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


def xception(**kwargs):
    """
    Xception model from 'Xception: Deep Learning with Depthwise Separable Convolutions,'
    https://arxiv.org/abs/1610.02357.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_xception(model_name="xception", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        xception,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xception or weight_count == 22855952)

        x = mx.nd.zeros((1, 3, 299, 299), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
