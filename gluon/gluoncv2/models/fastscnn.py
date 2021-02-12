"""
    Fast-SCNN for image segmentation, implemented in Gluon.
    Original paper: 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502.
"""

__all__ = ['FastSCNN', 'fastscnn_cityscapes']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import Identity
from .common import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwsconv3x3_block, Concurrent,\
    InterpolationBlock


class Stem(HybridBlock):
    """
    Fast-SCNN specific stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : tuple/list of 3 int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(Stem, self).__init__(**kwargs)
        assert (len(channels) == 3)

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=channels[0],
                strides=2,
                padding=0,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2 = dwsconv3x3_block(
                in_channels=channels[0],
                out_channels=channels[1],
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv3 = dwsconv3x3_block(
                in_channels=channels[1],
                out_channels=channels[2],
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LinearBottleneck(HybridBlock):
    """
    Fast-SCNN specific Linear Bottleneck layer from MobileNetV2.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
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
        super(LinearBottleneck, self).__init__(**kwargs)
        self.residual = (in_channels == out_channels) and (strides == 1)
        mid_channels = in_channels * 6

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class FeatureExtractor(HybridBlock):
    """
    Fast-SCNN specific feature extractor/encoder.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : list of list of int
        Number of output channels for each unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != len(channels) - 1) else 1
                        stage.add(LinearBottleneck(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bn_cudnn_off=bn_cudnn_off))
                        in_channels = out_channels
                self.features.add(stage)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x


class PoolingBranch(HybridBlock):
    """
    Fast-SCNN specific pooling branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    down_size : int
        Spatial size of downscaled image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 down_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(PoolingBranch, self).__init__(**kwargs)
        self.in_size = in_size
        self.down_size = down_size

        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=in_size)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.in_size is not None else x.shape[2:]
        x = F.contrib.AdaptiveAvgPooling2D(x, output_size=self.down_size)
        x = self.conv(x)
        x = self.up(x, in_size)
        return x


class FastPyramidPooling(HybridBlock):
    """
    Fast-SCNN specific fast pyramid pooling block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of 2 int or None
        Spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(FastPyramidPooling, self).__init__(**kwargs)
        down_sizes = [1, 2, 3, 6]
        mid_channels = in_channels // 4

        with self.name_scope():
            self.branches = Concurrent()
            self.branches.add(Identity())
            for down_size in down_sizes:
                self.branches.add(PoolingBranch(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    in_size=in_size,
                    down_size=down_size,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
            self.conv = conv1x1_block(
                in_channels=(in_channels * 2),
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        x = self.conv(x)
        return x


class FeatureFusion(HybridBlock):
    """
    Fast-SCNN specific feature fusion block.

    Parameters:
    ----------
    x_in_channels : int
        Number of high resolution (x) input channels.
    y_in_channels : int
        Number of low resolution (y) input channels.
    out_channels : int
        Number of output channels.
    x_in_size : tuple of 2 int or None
        Spatial size of high resolution (x) input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 x_in_channels,
                 y_in_channels,
                 out_channels,
                 x_in_size,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(FeatureFusion, self).__init__(**kwargs)
        self.x_in_size = x_in_size

        with self.name_scope():
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=x_in_size)
            self.low_dw_conv = dwconv3x3_block(
                in_channels=y_in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.low_pw_conv = conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)
            self.high_conv = conv1x1_block(
                in_channels=x_in_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off,
                activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x, y):
        x_in_size = self.x_in_size if self.x_in_size is not None else x.shape[2:]
        y = self.up(y, x_in_size)
        y = self.low_dw_conv(y)
        y = self.low_pw_conv(y)
        x = self.high_conv(x)
        out = x + y
        return self.activ(out)


class Head(HybridBlock):
    """
    Fast-SCNN head (classifier) block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 classes,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(Head, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv2 = dwsconv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.dropout = nn.Dropout(rate=0.1)
            self.conv3 = conv1x1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class AuxHead(HybridBlock):
    """
    Fast-SCNN auxiliary (after stem) head (classifier) block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    classes : int
        Number of classification classes.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 classes,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(AuxHead, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.dropout = nn.Dropout(rate=0.1)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=classes,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class FastSCNN(HybridBlock):
    """
    Fast-SCNN from 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502.

    Parameters:
    ----------
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 1024)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 aux=False,
                 fixed_size=True,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_channels=3,
                 in_size=(1024, 1024),
                 classes=19,
                 **kwargs):
        super(FastSCNN, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size

        with self.name_scope():
            steam_channels = [32, 48, 64]
            self.stem = Stem(
                in_channels=in_channels,
                channels=steam_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            in_channels = steam_channels[-1]
            feature_channels = [[64, 64, 64], [96, 96, 96], [128, 128, 128]]
            self.features = FeatureExtractor(
                in_channels=in_channels,
                channels=feature_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            pool_out_size = (in_size[0] // 32, in_size[1] // 32) if fixed_size else None
            self.pool = FastPyramidPooling(
                in_channels=feature_channels[-1][-1],
                out_channels=feature_channels[-1][-1],
                in_size=pool_out_size,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            fusion_out_size = (in_size[0] // 8, in_size[1] // 8) if fixed_size else None
            fusion_out_channels = 128
            self.fusion = FeatureFusion(
                x_in_channels=steam_channels[-1],
                y_in_channels=feature_channels[-1][-1],
                out_channels=fusion_out_channels,
                x_in_size=fusion_out_size,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.head = Head(
                in_channels=fusion_out_channels,
                classes=classes,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.up = InterpolationBlock(
                scale_factor=None,
                out_size=in_size)

            if self.aux:
                self.aux_head = AuxHead(
                    in_channels=64,
                    mid_channels=64,
                    classes=classes,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x = self.stem(x)
        y = self.features(x)
        y = self.pool(y)
        y = self.fusion(x, y)
        y = self.head(y)
        y = self.up(y, in_size)

        if self.aux:
            x = self.aux_head(x)
            x = self.up(x, in_size)
            return y, x
        return y


def get_fastscnn(model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    """
    Create Fast-SCNN model with specific parameters.

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
    net = FastSCNN(
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx,
            ignore_extra=True)

    return net


def fastscnn_cityscapes(classes=19, aux=True, **kwargs):
    """
    Fast-SCNN model for Cityscapes from 'Fast-SCNN: Fast Semantic Segmentation Network,'
    https://arxiv.org/abs/1902.04502.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_fastscnn(classes=classes, aux=aux, model_name="fastscnn_cityscapes", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    # in_size = (1024, 1024)
    in_size = (1024, 2048)
    aux = True
    pretrained = False
    fixed_size = False

    models = [
        (fastscnn_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size, aux=aux)

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
        if aux:
            assert (model != fastscnn_cityscapes or weight_count == 1176278)
        else:
            assert (model != fastscnn_cityscapes or weight_count == 1138051)

        x = mx.nd.zeros((1, 3, in_size[0], in_size[1]), ctx=ctx)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
