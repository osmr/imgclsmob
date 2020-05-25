"""
    BiSeNet for CelebAMask-HQ, implemented in TensorFlow.
    Original paper: 'BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1808.00897.
"""

__all__ = ['BiSeNet', 'bisenet_resnet18_celebamaskhq']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, InterpolationBlock, MultiOutputSequential, get_channel_axis,\
    get_im_size, is_channels_first
from .resnet import resnet18


class PyramidPoolingZeroBranch(nn.Layer):
    """
    Pyramid pooling zero branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    in_size : tuple of 2 int
        Spatial size of output image for the upsampling operation.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 data_format="channels_last",
                 **kwargs):
        super(PyramidPoolingZeroBranch, self).__init__(**kwargs)
        self.in_size = in_size
        self.data_format = data_format

        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")
        self.up = InterpolationBlock(
            scale_factor=None,
            interpolation="bilinear",
            data_format=data_format,
            name="up")

    def call(self, x, training=None):
        in_size = self.in_size if self.in_size is not None else get_im_size(x, data_format=self.data_format)
        x = self.pool(x)
        axis = -1 if is_channels_first(self.data_format) else 1
        x = tf.expand_dims(tf.expand_dims(x, axis=axis), axis=axis)
        x = self.conv(x, training=training)
        x = self.up(x, size=in_size)
        return x


class AttentionRefinementBlock(nn.Layer):
    """
    Attention refinement block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(AttentionRefinementBlock, self).__init__(**kwargs)
        self.data_format = data_format

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv1")
        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.conv2 = conv1x1_block(
            in_channels=out_channels,
            out_channels=out_channels,
            activation="sigmoid",
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        w = self.pool(x)
        axis = -1 if is_channels_first(self.data_format) else 1
        w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        w = self.conv2(w, training=training)
        x = x * w
        return x


class PyramidPoolingMainBranch(nn.Layer):
    """
    Pyramid pooling main branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    scale_factor : float
        Multiplier for spatial size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor,
                 data_format="channels_last",
                 **kwargs):
        super(PyramidPoolingMainBranch, self).__init__(**kwargs)
        self.att = AttentionRefinementBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="att")
        self.up = InterpolationBlock(
            scale_factor=scale_factor,
            interpolation="bilinear",
            data_format=data_format,
            name="up")
        self.conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")

    def call(self, x, y, training=None):
        x = self.att(x, training=training)
        x = x + y
        x = self.up(x)
        x = self.conv(x, training=training)
        return x


class FeatureFusion(nn.Layer):
    """
    Feature fusion block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    reduction : int, default 4
        Squeeze reduction value.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=4,
                 data_format="channels_last",
                 **kwargs):
        super(FeatureFusion, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = out_channels // reduction

        self.conv_merge = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv_merge")
        self.pool = nn.GlobalAveragePooling2D(
            data_format=data_format,
            name="pool")
        self.conv1 = conv1x1(
            in_channels=out_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.activ = nn.ReLU()
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv2")
        self.sigmoid = tf.nn.sigmoid

    def call(self, x, y, training=None):
        x = tf.concat([x, y], axis=get_channel_axis(self.data_format))
        x = self.conv_merge(x, training=training)
        w = self.pool(x)
        axis = -1 if is_channels_first(self.data_format) else 1
        w = tf.expand_dims(tf.expand_dims(w, axis=axis), axis=axis)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x_att = x * w
        x = x + x_att
        return x


class PyramidPooling(nn.Layer):
    """
    Pyramid Pooling module.

    Parameters:
    ----------
    x16_in_channels : int
        Number of input channels for x16.
    x32_in_channels : int
        Number of input channels for x32.
    y_out_channels : int
        Number of output channels for y-outputs.
    y32_out_size : tuple of 2 int
        Spatial size of the y32 tensor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 x16_in_channels,
                 x32_in_channels,
                 y_out_channels,
                 y32_out_size,
                 data_format="channels_last",
                 **kwargs):
        super(PyramidPooling, self).__init__(**kwargs)
        z_out_channels = 2 * y_out_channels

        self.pool32 = PyramidPoolingZeroBranch(
            in_channels=x32_in_channels,
            out_channels=y_out_channels,
            in_size=y32_out_size,
            data_format=data_format,
            name="pool32")
        self.pool16 = PyramidPoolingMainBranch(
            in_channels=x32_in_channels,
            out_channels=y_out_channels,
            scale_factor=2,
            data_format=data_format,
            name="pool16")
        self.pool8 = PyramidPoolingMainBranch(
            in_channels=x16_in_channels,
            out_channels=y_out_channels,
            scale_factor=2,
            data_format=data_format,
            name="pool8")
        self.fusion = FeatureFusion(
            in_channels=z_out_channels,
            out_channels=z_out_channels,
            data_format=data_format,
            name="fusion")

    def call(self, x8, x16, x32, training=None):
        y32 = self.pool32(x32, training=training)
        y16 = self.pool16(x32, y32, training=training)
        y8 = self.pool8(x16, y16, training=training)
        z8 = self.fusion(x8, y8, training=training)
        return z8, y8, y16


class BiSeHead(nn.Layer):
    """
    BiSeNet head (final) block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(BiSeHead, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x)
        return x


class BiSeNet(tf.keras.Model):
    """
    BiSeNet model from 'BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1808.00897.

    Parameters:
    ----------
    backbone : func -> nn.Sequential
        Feature extractor.
    aux : bool, default True
        Whether to output an auxiliary results.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (640, 480)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 backbone,
                 aux=True,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(640, 480),
                 classes=19,
                 data_format="channels_last",
                 **kwargs):
        super(BiSeNet, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        self.aux = aux
        self.fixed_size = fixed_size

        self.backbone, backbone_out_channels = backbone(
            data_format=data_format,
            name="backbone")

        y_out_channels = backbone_out_channels[0]
        z_out_channels = 2 * y_out_channels
        y32_out_size = (self.in_size[0] // 32, self.in_size[1] // 32) if fixed_size else None
        self.pool = PyramidPooling(
            x16_in_channels=backbone_out_channels[1],
            x32_in_channels=backbone_out_channels[2],
            y_out_channels=y_out_channels,
            y32_out_size=y32_out_size,
            data_format=data_format,
            name="pool")
        self.head_z8 = BiSeHead(
            in_channels=z_out_channels,
            mid_channels=z_out_channels,
            out_channels=classes,
            data_format=data_format,
            name="head_z8")
        self.up8 = InterpolationBlock(
            scale_factor=(8 if fixed_size else None),
            data_format=data_format,
            name="up8")

        if self.aux:
            mid_channels = y_out_channels // 2
            self.head_y8 = BiSeHead(
                in_channels=y_out_channels,
                mid_channels=mid_channels,
                out_channels=classes,
                data_format=data_format,
                name="head_y8")
            self.head_y16 = BiSeHead(
                in_channels=y_out_channels,
                mid_channels=mid_channels,
                out_channels=classes,
                data_format=data_format,
                name="head_y16")
            self.up16 = InterpolationBlock(
                scale_factor=(16 if fixed_size else None),
                data_format=data_format,
                name="up16")

    def call(self, x, training=None):
        assert is_channels_first(self.data_format) or ((x.shape[1] % 32 == 0) and (x.shape[2] % 32 == 0))
        assert (not is_channels_first(self.data_format)) or ((x.shape[2] % 32 == 0) and (x.shape[3] % 32 == 0))

        x8, x16, x32 = self.backbone(x, training=training)
        z8, y8, y16 = self.pool(x8, x16, x32, training=training)

        z8 = self.head_z8(z8, training=training)
        z8 = self.up8(z8)

        if self.aux:
            y8 = self.head_y8(y8, training=training)
            y16 = self.head_y16(y16, training=training)
            y8 = self.up8(y8)
            y16 = self.up16(y16)
            return z8, y8, y16
        else:
            return z8


def get_bisenet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create BiSeNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = BiSeNet(
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def bisenet_resnet18_celebamaskhq(pretrained_backbone=False, classes=19, **kwargs):
    """
    BiSeNet model on the base of ResNet-18 for face segmentation on CelebAMask-HQ from 'BiSeNet: Bilateral Segmentation
    Network for Real-time Semantic Segmentation,' https://arxiv.org/abs/1808.00897.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    def backbone(**bb_kwargs):
        features_raw = resnet18(pretrained=pretrained_backbone, **bb_kwargs).features
        features_raw._layers.pop()
        features = MultiOutputSequential(return_last=False, name="backbone")
        features.add(features_raw._layers[0])
        for i, stage in enumerate(features_raw._layers[1:]):
            if i != 0:
                stage.do_output = True
            features.add(stage)
        out_channels = [128, 256, 512]
        return features, out_channels
    return get_bisenet(backbone=backbone, classes=classes, model_name="bisenet_resnet18_celebamaskhq", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (640, 480)
    aux = True
    pretrained = False

    models = [
        bisenet_resnet18_celebamaskhq,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape[0] == x.shape[0])
        if is_channels_first(data_format):
            assert ((y.shape[1] == 19) and (y.shape[2] == x.shape[2]) and (y.shape[3] == x.shape[3]))
        else:
            assert ((y.shape[3] == 19) and (y.shape[1] == x.shape[1]) and (y.shape[2] == x.shape[2]))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != bisenet_resnet18_celebamaskhq or weight_count == 13300416)
        else:
            assert (model != bisenet_resnet18_celebamaskhq or weight_count == 13150272)


if __name__ == "__main__":
    _test()
