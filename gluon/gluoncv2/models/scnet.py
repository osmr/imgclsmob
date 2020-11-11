"""
    SCNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.
"""

__all__ = ['SCNet', 'scnet50', 'scnet101', 'scneta50', 'scneta101']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block, InterpolationBlock
from .resnet import ResInitBlock
from .senet import SEInitBlock
from .resnesta import ResNeStADownBlock


class ScDownBlock(HybridBlock):
    """
    SCNet specific convolutional downscale block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    pool_size: int or list/tuple of 2 ints, default 2
        Size of the average pooling windows.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_size=2,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(ScDownBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=pool_size,
                strides=pool_size)
            self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ScConv(HybridBlock):
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
    scale_factor : int
        Scale factor.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    in_size : tuple of 2 int, default None
        Spatial size of output image for the upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 scale_factor,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_size=None,
                 **kwargs):
        super(ScConv, self).__init__(**kwargs)
        self.in_size = in_size

        with self.name_scope():
            self.down = ScDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                pool_size=scale_factor,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.up = InterpolationBlock(scale_factor=scale_factor, bilinear=False)
            self.sigmoid = nn.Activation("sigmoid")
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)
            self.conv2 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        w = self.sigmoid(x + self.up(self.down(x), in_size))
        x = self.conv1(x) * w
        x = self.conv2(x)
        return x


class ScBottleneck(HybridBlock):
    """
    SCNet specific bottleneck block for residual path in SCNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    scale_factor : int, default 4
        Scale factor.
    avg_downsample : bool, default False
        Whether to use average downsampling.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    in_size : tuple of 2 int, default None
        Spatial size of output image for the upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck_factor=4,
                 scale_factor=4,
                 avg_downsample=False,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_size=None,
                 **kwargs):
        super(ScBottleneck, self).__init__(**kwargs)
        self.avg_resize = (strides > 1) and avg_downsample
        mid_channels = out_channels // bottleneck_factor // 2

        with self.name_scope():
            self.conv1a = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2a = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if self.avg_resize else strides),
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

            self.conv1b = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2b = ScConv(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=(1 if self.avg_resize else strides),
                scale_factor=scale_factor,
                in_size=in_size)

            if self.avg_resize:
                self.pool = nn.AvgPool2D(
                    pool_size=3,
                    strides=strides,
                    padding=1)

            self.conv3 = conv1x1_block(
                in_channels=(2 * mid_channels),
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)

    def hybrid_forward(self, F, x):
        y = self.conv1a(x)
        y = self.conv2a(y)

        z = self.conv1b(x)
        z = self.conv2b(z)

        if self.avg_resize:
            y = self.pool(y)
            z = self.pool(z)

        x = F.concat(y, z, dim=1)

        x = self.conv3(x)
        return x


class ScUnit(HybridBlock):
    """
    SCNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    avg_downsample : bool, default False
        Whether to use average downsampling.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    in_size : tuple of 2 int, default None
        Spatial size of output image for the upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 avg_downsample=False,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_size=None,
                 **kwargs):
        super(ScUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = ScBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                avg_downsample=avg_downsample,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                in_size=in_size)
            if self.resize_identity:
                if avg_downsample:
                    self.identity_block = ResNeStADownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=strides,
                        bn_use_global_stats=bn_use_global_stats,
                        bn_cudnn_off=bn_cudnn_off)
                else:
                    self.identity_block = conv1x1_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=strides,
                        bn_use_global_stats=bn_use_global_stats,
                        bn_cudnn_off=bn_cudnn_off,
                        activation=None)
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
    SCNet model from 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    se_init_block : bool, default False
        SENet-like initial block.
    avg_downsample : bool, default False
        Whether to use average downsampling.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
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
                 se_init_block=False,
                 avg_downsample=False,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(SCNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            init_block_class = SEInitBlock if se_init_block else ResInitBlock
            self.features.add(init_block_class(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = init_block_channels
            in_size = (in_size[0] // 4, in_size[1] // 4)
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(ScUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            avg_downsample=avg_downsample,
                            bn_use_global_stats=bn_use_global_stats,
                            bn_cudnn_off=bn_cudnn_off,
                            in_size=in_size))
                        in_channels = out_channels
                        if strides > 1:
                            in_size = (in_size[0] // 2, in_size[1] // 2) if fixed_size else None
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


def get_scnet(blocks,
              width_scale=1.0,
              se_init_block=False,
              avg_downsample=False,
              init_block_channels_scale=1,
              model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    """
    Create SCNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    se_init_block : bool, default False
        SENet-like initial block.
    avg_downsample : bool, default False
        Whether to use average downsampling.
    init_block_channels_scale : int, default 1
        Scale factor for number of output channels in the initial unit.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if blocks == 14:
        layers = [1, 1, 1, 1]
    elif blocks == 26:
        layers = [2, 2, 2, 2]
    elif blocks == 38:
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
        raise ValueError("Unsupported SCNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    init_block_channels *= init_block_channels_scale

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
        se_init_block=se_init_block,
        avg_downsample=avg_downsample,
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


def scnet50(**kwargs):
    """
    SCNet-50 model from 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
     http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_scnet(blocks=50, model_name="scnet50", **kwargs)


def scnet101(**kwargs):
    """
    SCNet-101 model from 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_scnet(blocks=101, model_name="scnet101", **kwargs)


def scneta50(**kwargs):
    """
    SCNet(A)-50 with average downsampling model from 'Improving Convolutional Networks with Self-Calibrated
    Convolutions,' http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_scnet(blocks=50, se_init_block=True, avg_downsample=True, model_name="scneta50", **kwargs)


def scneta101(**kwargs):
    """
    SCNet(A)-101 with average downsampling model from 'Improving Convolutional Networks with Self-Calibrated
    Convolutions,' http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_scnet(blocks=101, se_init_block=True, avg_downsample=True, init_block_channels_scale=2,
                     model_name="scneta101", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    fixed_size = True
    pretrained = False

    models = [
        scnet50,
        scnet101,
        scneta50,
        scneta101,
    ]

    for model in models:

        net = model(pretrained=pretrained, fixed_size=fixed_size)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != scnet50 or weight_count == 25564584)
        assert (model != scnet101 or weight_count == 44565416)
        assert (model != scneta50 or weight_count == 25583816)
        assert (model != scneta101 or weight_count == 44689192)

        batch = 1
        x = mx.nd.random.normal(shape=(batch, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (batch, 1000))


if __name__ == "__main__":
    _test()
