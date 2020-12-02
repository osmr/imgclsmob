"""
    RegNetV for ImageNet-1K, implemented in Gluon.
    Original paper: 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.
"""

__all__ = ['RegNetV', 'regnetv002', 'regnetv004', 'regnetv006', 'regnetv008', 'regnetv016', 'regnetv032', 'regnetv040',
           'regnetv064', 'regnetv080', 'regnetv120', 'regnetv160', 'regnetv320']

import os
import numpy as np
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block, dwsconv3x3_block


class DownBlock(HybridBlock):
    """
    ResNet(A)-like downsample block for the identity branch of a residual unit.

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
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DownBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.pool = nn.AvgPool2D(
                pool_size=strides,
                strides=strides,
                ceil_mode=True,
                count_include_pad=False)
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class RegNetVUnit(HybridBlock):
    """
    RegNetV unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    downscale : bool
        Whether to downscale tensor.
    dw_use_bn : bool
        Whether to use BatchNorm layer (depthwise convolution block).
    dw_activation : function or str or None
        Activation function after the depthwise convolution block.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 downscale,
                 dw_use_bn,
                 dw_activation,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(RegNetVUnit, self).__init__(**kwargs)
        self.downscale = downscale

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            if self.downscale:
                self.pool = nn.AvgPool2D(
                    pool_size=3,
                    strides=2,
                    padding=1)
            self.conv2 = dwsconv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                dw_use_bn=dw_use_bn,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                dw_activation=dw_activation,
                pw_activation=None)
            if self.downscale:
                self.identity_block = DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=2,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.downscale:
            identity = self.identity_block(x)
        else:
            identity = x
        x = self.conv1(x)
        if self.downscale:
            x = self.pool(x)
        x = self.conv2(x)
        x = x + identity
        x = self.activ(x)
        return x


class RegNetVInitBlock(HybridBlock):
    """
    RegNetV specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(RegNetVInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x


class RegNetV(HybridBlock):
    """
    RegNet model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    dw_activation : function or str or None, default nn.Activation('relu')
        Activation function after the depthwise convolution block.
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
                 dw_use_bn=True,
                 dw_activation=(lambda: nn.Activation("relu")),
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(RegNetV, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(RegNetVInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        downscale = (j == 0)
                        stage.add(RegNetVUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            downscale=downscale,
                            dw_use_bn=dw_use_bn,
                            dw_activation=dw_activation,
                            bn_use_global_stats=bn_use_global_stats,
                            bn_cudnn_off=bn_cudnn_off))
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


def get_regnet(channels_init,
               channels_slope,
               channels_mult,
               depth,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create RegNet model with specific parameters.

    Parameters:
    ----------
    channels_init : float
        Initial value for channels/widths.
    channels_slope : float
        Slope value for channels/widths.
    width_mult : float
        Width multiplier value.
    depth : int
        Depth value.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    divisor = 8
    assert (channels_slope >= 0) and (channels_init > 0) and (channels_mult > 1) and (channels_init % divisor == 0)

    # Generate continuous per-block channels/widths:
    channels_cont = np.arange(depth) * channels_slope + channels_init

    # Generate quantized per-block channels/widths:
    channels_exps = np.round(np.log(channels_cont / channels_init) / np.log(channels_mult))
    channels = channels_init * np.power(channels_mult, channels_exps)
    channels = (np.round(channels / divisor) * divisor).astype(np.int)

    # Generate per stage channels/widths and layers/depths:
    channels_per_stage, layers = np.unique(channels, return_counts=True)

    channels = [[ci] * li for (ci, li) in zip(channels_per_stage, layers)]

    init_block_channels = 32

    dws_simplified = True
    if dws_simplified:
        dw_use_bn = False
        dw_activation = None
    else:
        dw_use_bn = True
        dw_activation = (lambda: nn.Activation("relu"))

    net = RegNetV(
        channels=channels,
        init_block_channels=init_block_channels,
        dw_use_bn=dw_use_bn,
        dw_activation=dw_activation,
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


def regnetv002(**kwargs):
    """
    RegNetV-200MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=24, channels_slope=36.44, channels_mult=2.49, depth=13,
                      model_name="regnetv002", **kwargs)


def regnetv004(**kwargs):
    """
    RegNetV-400MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=24, channels_slope=24.48, channels_mult=2.54, depth=22,
                      model_name="regnetv004", **kwargs)


def regnetv006(**kwargs):
    """
    RegNetV-600MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=48, channels_slope=36.97, channels_mult=2.24, depth=16,
                      model_name="regnetv006", **kwargs)


def regnetv008(**kwargs):
    """
    RegNetV-800MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=56, channels_slope=35.73, channels_mult=2.28, depth=16,
                      model_name="regnetv008", **kwargs)


def regnetv016(**kwargs):
    """
    RegNetV-1.6GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=80, channels_slope=34.01, channels_mult=2.25, depth=18,
                      model_name="regnetv016", **kwargs)


def regnetv032(**kwargs):
    """
    RegNetV-3.2GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=88, channels_slope=26.31, channels_mult=2.25, depth=25,
                      model_name="regnetv032", **kwargs)


def regnetv040(**kwargs):
    """
    RegNetV-4.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=96, channels_slope=38.65, channels_mult=2.43, depth=23,
                      model_name="regnetv040", **kwargs)


def regnetv064(**kwargs):
    """
    RegNetV-6.4GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=184, channels_slope=60.83, channels_mult=2.07, depth=17,
                      model_name="regnetv064", **kwargs)


def regnetv080(**kwargs):
    """
    RegNetV-8.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=80, channels_slope=49.56, channels_mult=2.88, depth=23,
                      model_name="regnetv080", **kwargs)


def regnetv120(**kwargs):
    """
    RegNetV-12GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=168, channels_slope=73.36, channels_mult=2.37, depth=19,
                      model_name="regnetv120", **kwargs)


def regnetv160(**kwargs):
    """
    RegNetV-16GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=216, channels_slope=55.59, channels_mult=2.1, depth=22,
                      model_name="regnetv160", **kwargs)


def regnetv320(**kwargs):
    """
    RegNetV-32GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=320, channels_slope=69.86, channels_mult=2.0, depth=23,
                      model_name="regnetv320", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    dws_simplified = True
    pretrained = False

    models = [
        regnetv002,
        regnetv004,
        regnetv006,
        regnetv008,
        regnetv016,
        regnetv032,
        regnetv040,
        regnetv064,
        regnetv080,
        regnetv120,
        regnetv160,
        regnetv320,
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
        if dws_simplified:
            assert (model != regnetv002 or weight_count == 2476840)
            assert (model != regnetv004 or weight_count == 4467080)
            assert (model != regnetv006 or weight_count == 5242936)
            assert (model != regnetv008 or weight_count == 6353000)
            assert (model != regnetv016 or weight_count == 7824440)
            assert (model != regnetv032 or weight_count == 11540536)
            assert (model != regnetv040 or weight_count == 18323824)
            assert (model != regnetv064 or weight_count == 20854680)
            assert (model != regnetv080 or weight_count == 21930224)
            assert (model != regnetv120 or weight_count == 32833720)
            assert (model != regnetv160 or weight_count == 36213360)
            assert (model != regnetv320 or weight_count == 64659576)
        else:
            assert (model != regnetv002 or weight_count == 2479160)
            assert (model != regnetv004 or weight_count == 4474712)
            assert (model != regnetv006 or weight_count == 5249352)
            assert (model != regnetv008 or weight_count == 6360344)
            assert (model != regnetv016 or weight_count == 7833768)
            assert (model != regnetv032 or weight_count == 11556520)
            assert (model != regnetv040 or weight_count == 18343728)
            assert (model != regnetv064 or weight_count == 20873384)
            assert (model != regnetv080 or weight_count == 21952400)
            assert (model != regnetv120 or weight_count == 32859432)
            assert (model != regnetv160 or weight_count == 36244240)
            assert (model != regnetv320 or weight_count == 64704008)
        batch = 14
        size = 224
        x = mx.nd.zeros((batch, 3, size, size), ctx=ctx)
        y = net(x)
        assert (y.shape == (batch, 1000))


if __name__ == "__main__":
    _test()
