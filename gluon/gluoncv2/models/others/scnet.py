"""
    SCNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.
"""

__all__ = ['SCNet', 'scnet50b', 'scnet101b']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from gluon.gluoncv2.models.common import conv1x1_block, conv3x3_block, InterpolationBlock
from gluon.gluoncv2.models.resnet import ResBlock, ResBottleneck
from gluon.gluoncv2.models.senet import SEInitBlock


class ConvDownBlock(HybridBlock):
    """
    Convolutional downscale block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    groups : int, default 1
        Number of groups.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    pool_size: int or list/tuple of 2 ints, default 2
        Size of the average pooling windows.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding,
                 dilation=1,
                 groups=1,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 pool_size=2,
                 **kwargs):
        super(ConvDownBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=pool_size,
                strides=pool_size)
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class SCConv(HybridBlock):
    """
    Self-calibrated convolutional block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    groups : int, default 1
        Number of groups.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    pool_size: int or list/tuple of 2 ints, default 2
        Size of the average pooling windows.
    in_size : tuple of 2 int, default None
        Spatial size of output image for the upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 pool_size=2,
                 in_size=None,
                 **kwargs):
        super(SCConv, self).__init__(**kwargs)
        self.in_size = in_size

        with self.name_scope():
            self.down = ConvDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                pool_size=pool_size)
            self.up = InterpolationBlock(scale_factor=None)
            self.sigmoid = nn.Activation("sigmoid")
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)
            self.conv2 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        y = self.sigmoid(x + self.up(self.down(x), in_size))
        y = y * self.conv1(x)
        y = self.conv2(y)
        return y


class SCBottleneck(HybridBlock):
    """
    SCNet bottleneck block for residual path in SCNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    pool_size: int or list/tuple of 2 ints, default 4
        Size of the average pooling windows.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 conv1_stride=False,
                 bottleneck_factor=4,
                 pool_size=4,
                 **kwargs):
        super(SCBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.k1 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

            self.conv2 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.scconv = SCConv(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                padding=padding,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                pool_size=pool_size)

            self.conv3 = conv1x1_block(
                in_channels=(2 * mid_channels),
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)

    def hybrid_forward(self, F, x):
        y1 = self.conv1(x)
        y1 = self.k1(y1)

        y2 = self.conv2(x)
        y2 = self.scconv(y2)


        x = self.conv3(x)
        return x


class ResAUnit(HybridBlock):
    """
    ResNet(A) unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 padding=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 bottleneck=True,
                 conv1_stride=False,
                 **kwargs):
        super(ResAUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off,
                    conv1_stride=conv1_stride)
            else:
                self.body = ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)
            if self.resize_identity:
                self.identity_block = ResADownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_block(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class SCNet(HybridBlock):
    """
    ResNet(A) with average downsampling model from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385.

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
    dilated : bool, default False
        Whether to use dilation.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
                 bn_cudnn_off=False,
                 dilated=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(SCNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(SEInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        if dilated:
                            strides = 2 if ((j == 0) and (i != 0) and (i < 2)) else 1
                            dilation = (2 ** max(0, i - 1 - int(j == 0)))
                        else:
                            strides = 2 if (j == 0) and (i != 0) else 1
                            dilation = 1
                        stage.add(ResAUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            padding=dilation,
                            dilation=dilation,
                            bn_use_global_stats=bn_use_global_stats,
                            bn_cudnn_off=bn_cudnn_off,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_resneta(blocks,
                bottleneck=None,
                conv1_stride=True,
                width_scale=1.0,
                model_name=None,
                pretrained=False,
                ctx=cpu(),
                root=os.path.join("~", ".mxnet", "models"),
                **kwargs):
    """
    Create ResNet(A) with average downsampling model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet(A) with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = SCNet(
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


def scnet50b(**kwargs):
    """
    ResNet(A)-50 with average downsampling model with stride at the second convolution in bottleneck block
    from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=50, conv1_stride=False, model_name="resneta50b", **kwargs)


def scnet101b(**kwargs):
    """
    ResNet(A)-101 with average downsampling model with stride at the second convolution in bottleneck
    block from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resneta(blocks=101, conv1_stride=False, model_name="resneta101b", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        scnet50b,
        scnet101b,
    ]

    for model in models:

        net = model(
            pretrained=pretrained)

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
        assert (model != scnet50b or weight_count == 25576264)
        assert (model != scnet101b or weight_count == 44568392)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
