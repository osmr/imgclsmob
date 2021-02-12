"""
    Fast-SCNN for image segmentation, implemented in TensorFlow.
    Original paper: 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502.
"""

__all__ = ['FastSCNN', 'fastscnn_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwsconv3x3_block, Concurrent,\
    InterpolationBlock, SimpleSequential, Identity, get_im_size, is_channels_first


class Stem(nn.Layer):
    """
    Fast-SCNN specific stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : tuple/list of 3 int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 data_format="channels_last",
                 **kwargs):
        super(Stem, self).__init__(**kwargs)
        assert (len(channels) == 3)

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=channels[0],
            strides=2,
            padding=0,
            data_format=data_format,
            name="conv1")
        self.conv2 = dwsconv3x3_block(
            in_channels=channels[0],
            out_channels=channels[1],
            strides=2,
            data_format=data_format,
            name="conv2")
        self.conv3 = dwsconv3x3_block(
            in_channels=channels[1],
            out_channels=channels[2],
            strides=2,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class LinearBottleneck(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 data_format="channels_last",
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.residual = (in_channels == out_channels) and (strides == 1)
        mid_channels = in_channels * 6

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = dwconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        if self.residual:
            identity = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        if self.residual:
            x = x + identity
        return x


class FeatureExtractor(nn.Layer):
    """
    Fast-SCNN specific feature extractor/encoder.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : list of list of int
        Number of output channels for each unit.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 data_format="channels_last",
                 **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        self.features = SimpleSequential(name="features")
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != len(channels) - 1) else 1
                stage.add(LinearBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)

    def call(self, x, training=None):
        x = self.features(x, training=training)
        return x


class PoolingBranch(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 down_size,
                 data_format="channels_last",
                 **kwargs):
        super(PoolingBranch, self).__init__(**kwargs)
        self.in_size = in_size
        self.down_size = down_size
        self.data_format = data_format

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")
        self.up = InterpolationBlock(
            scale_factor=None,
            out_size=in_size,
            data_format=data_format,
            name="up")

    def call(self, x, training=None):
        in_size = self.in_size if self.in_size is not None else get_im_size(x, data_format=self.data_format)
        x = nn.AveragePooling2D(pool_size=(in_size[0] // self.down_size, in_size[1] // self.down_size), strides=1,
                                data_format=self.data_format, name="pool")(x)
        x = self.conv(x, training=training)
        x = self.up(x, in_size)
        return x


class FastPyramidPooling(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(FastPyramidPooling, self).__init__(**kwargs)
        down_sizes = [1, 2, 3, 6]
        mid_channels = in_channels // 4

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(Identity(name="branch1"))
        for i, down_size in enumerate(down_sizes):
            self.branches.add(PoolingBranch(
                in_channels=in_channels,
                out_channels=mid_channels,
                in_size=in_size,
                down_size=down_size,
                data_format=data_format,
                name="branch{}".format(i + 2)))
        self.conv = conv1x1_block(
            in_channels=(in_channels * 2),
            out_channels=out_channels,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        return x


class FeatureFusion(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 x_in_channels,
                 y_in_channels,
                 out_channels,
                 x_in_size,
                 data_format="channels_last",
                 **kwargs):
        super(FeatureFusion, self).__init__(**kwargs)
        self.x_in_size = x_in_size
        self.data_format = data_format

        self.up = InterpolationBlock(
            scale_factor=None,
            out_size=x_in_size,
            data_format=data_format,
            name="up")
        self.low_dw_conv = dwconv3x3_block(
            in_channels=y_in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="low_dw_conv")
        self.low_pw_conv = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bias=True,
            activation=None,
            data_format=data_format,
            name="low_pw_conv")
        self.high_conv = conv1x1_block(
            in_channels=x_in_channels,
            out_channels=out_channels,
            use_bias=True,
            activation=None,
            data_format=data_format,
            name="high_conv")
        self.activ = nn.ReLU()

    def call(self, x, y, training=None):
        x_in_size = self.x_in_size if self.x_in_size is not None else get_im_size(x, data_format=self.data_format)
        y = self.up(y, x_in_size)
        y = self.low_dw_conv(y, training=training)
        y = self.low_pw_conv(y, training=training)
        x = self.high_conv(x, training=training)
        out = x + y
        return self.activ(out)


class Head(nn.Layer):
    """
    Fast-SCNN head (classifier) block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 classes,
                 data_format="channels_last",
                 **kwargs):
        super(Head, self).__init__(**kwargs)
        self.conv1 = dwsconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = dwsconv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            data_format=data_format,
            name="conv2")
        self.dropout = nn.Dropout(
            rate=0.1,
            name="dropout")
        self.conv3 = conv1x1(
            in_channels=in_channels,
            out_channels=classes,
            use_bias=True,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.dropout(x, training=training)
        x = self.conv3(x)
        return x


class AuxHead(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 classes,
                 data_format="channels_last",
                 **kwargs):
        super(AuxHead, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.dropout = nn.Dropout(
            rate=0.1,
            name="dropout")
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=classes,
            use_bias=True,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        return x


class FastSCNN(tf.keras.Model):
    """
    Fast-SCNN from 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502.

    Parameters:
    ----------
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 1024)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(1024, 1024),
                 classes=19,
                 data_format="channels_last",
                 **kwargs):
        super(FastSCNN, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 32 == 0) and (in_size[1] % 32 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size
        self.data_format = data_format

        steam_channels = [32, 48, 64]
        self.stem = Stem(
            in_channels=in_channels,
            channels=steam_channels,
            data_format=data_format,
            name="stem")
        in_channels = steam_channels[-1]
        feature_channels = [[64, 64, 64], [96, 96, 96], [128, 128, 128]]
        self.features = FeatureExtractor(
            in_channels=in_channels,
            channels=feature_channels,
            data_format=data_format,
            name="features")
        pool_out_size = (in_size[0] // 32, in_size[1] // 32) if fixed_size else None
        self.pool = FastPyramidPooling(
            in_channels=feature_channels[-1][-1],
            out_channels=feature_channels[-1][-1],
            in_size=pool_out_size,
            data_format=data_format,
            name="pool")
        fusion_out_size = (in_size[0] // 8, in_size[1] // 8) if fixed_size else None
        fusion_out_channels = 128
        self.fusion = FeatureFusion(
            x_in_channels=steam_channels[-1],
            y_in_channels=feature_channels[-1][-1],
            out_channels=fusion_out_channels,
            x_in_size=fusion_out_size,
            data_format=data_format,
            name="fusion")
        self.head = Head(
            in_channels=fusion_out_channels,
            classes=classes,
            data_format=data_format,
            name="head")
        self.up = InterpolationBlock(
            scale_factor=None,
            out_size=in_size,
            data_format=data_format,
            name="up")

        if self.aux:
            self.aux_head = AuxHead(
                in_channels=64,
                mid_channels=64,
                classes=classes,
                data_format=data_format,
                name="aux_head")

    def call(self, x, training=None):
        in_size = self.in_size if self.fixed_size else get_im_size(x, data_format=self.data_format)
        x = self.stem(x, training=training)
        y = self.features(x, training=training)
        y = self.pool(y, training=training)
        y = self.fusion(x, y, training=training)
        y = self.head(y, training=training)
        y = self.up(y, in_size)

        if self.aux:
            x = self.aux_head(x, training=training)
            x = self.up(x, in_size)
            return y, x
        return y


def get_fastscnn(model_name=None,
                 pretrained=False,
                 root=os.path.join("~", ".tensorflow", "models"),
                 **kwargs):
    """
    Create Fast-SCNN model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = FastSCNN(
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            by_name=True,
            skip_mismatch=True)

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
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_fastscnn(classes=classes, aux=aux, model_name="fastscnn_cityscapes", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (1024, 2048)
    aux = True
    fixed_size = False
    pretrained = True

    models = [
        (fastscnn_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux, fixed_size=fixed_size, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape[0] == x.shape[0])
        if is_channels_first(data_format):
            assert ((y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and (y.shape[3] == x.shape[3]))
        else:
            assert ((y.shape[3] == classes) and (y.shape[1] == x.shape[1]) and (y.shape[2] == x.shape[2]))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != fastscnn_cityscapes or weight_count == 1176278)
        else:
            assert (model != fastscnn_cityscapes or weight_count == 1138051)


if __name__ == "__main__":
    _test()
