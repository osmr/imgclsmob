"""
    ShaResNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.
"""

__all__ = ['ShaResNet', 'sharesnet18', 'sharesnet34', 'sharesnet50', 'sharesnet50b', 'sharesnet101', 'sharesnet101b',
           'sharesnet152', 'sharesnet152b']

import os
from inspect import isfunction
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import ReLU6, conv1x1_block, conv3x3_block
from .resnet import ResInitBlock


class ShaConvBlock(HybridBlock):
    """
    Shared convolution block with Batch normalization and ReLU/ReLU6 activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation("relu")
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    shared_conv : HybridBlock, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 activate=True,
                 shared_conv=None,
                 **kwargs):
        super(ShaConvBlock, self).__init__(**kwargs)
        self.activate = activate

        with self.name_scope():
            if shared_conv is None:
                self.conv = nn.Conv2D(
                    channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    use_bias=use_bias,
                    in_channels=in_channels)
            else:
                self.conv = shared_conv
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            if self.activate:
                assert (activation is not None)
                if isfunction(activation):
                    self.activ = activation()
                elif isinstance(activation, str):
                    if activation == "relu6":
                        self.activ = ReLU6()
                    else:
                        self.activ = nn.Activation(activation)
                else:
                    self.activ = activation

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def sha_conv3x3_block(in_channels,
                      out_channels,
                      strides=1,
                      padding=1,
                      dilation=1,
                      groups=1,
                      use_bias=False,
                      bn_use_global_stats=False,
                      activation=(lambda: nn.Activation("relu")),
                      activate=True,
                      shared_conv=None,
                      **kwargs):
    """
    3x3 version of the shared convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation("relu")
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    shared_conv : HybridBlock, default None
        Shared convolution layer.
    """
    return ShaConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        activate=activate,
        shared_conv=shared_conv,
        **kwargs)


class ShaResBlock(HybridBlock):
    """
    Simple ShaResNet block for residual path in ShaResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    shared_conv : HybridBlock, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 shared_conv=None,
                 **kwargs):
        super(ShaResBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = sha_conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None,
                activate=False,
                shared_conv=shared_conv)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ShaResBottleneck(HybridBlock):
    """
    ShaResNet bottleneck block for residual path in ShaResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    shared_conv : HybridBlock, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 conv1_stride=False,
                 bottleneck_factor=4,
                 shared_conv=None,
                 **kwargs):
        super(ShaResBottleneck, self).__init__(**kwargs)
        assert (conv1_stride or not ((strides > 1) and (shared_conv is not None)))
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=(strides if conv1_stride else 1),
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = sha_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if conv1_stride else strides),
                bn_use_global_stats=bn_use_global_stats,
                shared_conv=shared_conv)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class ShaResUnit(HybridBlock):
    """
    ShaResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    shared_conv : HybridBlock, default None
        Shared convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 bottleneck,
                 conv1_stride,
                 shared_conv=None,
                 **kwargs):
        super(ShaResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ShaResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    conv1_stride=conv1_stride,
                    shared_conv=shared_conv)
            else:
                self.body = ShaResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    shared_conv=shared_conv)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class ShaResNet(HybridBlock):
    """
    ShaResNet model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bottleneck,
                 conv1_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(ShaResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                shared_conv = None
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        unit = ShaResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride,
                            shared_conv=shared_conv)
                        if (shared_conv is None) and not (bottleneck and not conv1_stride and strides > 1):
                            shared_conv = unit.body.conv2.conv
                        stage.add(unit)
                        in_channels = out_channels
                self.features.add(stage)
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


def get_sharesnet(blocks,
                  conv1_stride=True,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create ShaResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
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
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ShaResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = ShaResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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


def sharesnet18(**kwargs):
    """
    ShaResNet-18 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=18, model_name="sharesnet18", **kwargs)


def sharesnet34(**kwargs):
    """
    ShaResNet-34 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=34, model_name="sharesnet34", **kwargs)


def sharesnet50(**kwargs):
    """
    ShaResNet-50 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=50, model_name="sharesnet50", **kwargs)


def sharesnet50b(**kwargs):
    """
    ShaResNet-50b model with stride at the second convolution in bottleneck block from 'ShaResNet: reducing residual
    network parameter number by sharing weights,' https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=50, conv1_stride=False, model_name="sharesnet50b", **kwargs)


def sharesnet101(**kwargs):
    """
    ShaResNet-101 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=101, model_name="sharesnet101", **kwargs)


def sharesnet101b(**kwargs):
    """
    ShaResNet-101b model with stride at the second convolution in bottleneck block from 'ShaResNet: reducing residual
    network parameter number by sharing weights,' https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=101, conv1_stride=False, model_name="sharesnet101b", **kwargs)


def sharesnet152(**kwargs):
    """
    ShaResNet-152 model from 'ShaResNet: reducing residual network parameter number by sharing weights,'
    https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=152, model_name="sharesnet152", **kwargs)


def sharesnet152b(**kwargs):
    """
    ShaResNet-152b model with stride at the second convolution in bottleneck block from 'ShaResNet: reducing residual
    network parameter number by sharing weights,' https://arxiv.org/abs/1702.08782.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_sharesnet(blocks=152, conv1_stride=False, model_name="sharesnet152b", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        sharesnet18,
        sharesnet34,
        sharesnet50,
        sharesnet50b,
        sharesnet101,
        sharesnet101b,
        sharesnet152,
        sharesnet152b,
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
        assert (model != sharesnet18 or weight_count == 8556072)
        assert (model != sharesnet34 or weight_count == 13613864)
        assert (model != sharesnet50 or weight_count == 17373224)
        assert (model != sharesnet50b or weight_count == 20469800)
        assert (model != sharesnet101 or weight_count == 26338344)
        assert (model != sharesnet101b or weight_count == 29434920)
        assert (model != sharesnet152 or weight_count == 33724456)
        assert (model != sharesnet152b or weight_count == 36821032)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
