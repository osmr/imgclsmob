"""
    ICNet for image segmentation, implemented in TensorFlow.
    Original paper: 'ICNet for Real-Time Semantic Segmentation on High-Resolution Images,'
    https://arxiv.org/abs/1704.08545.
"""

__all__ = ['ICNet', 'icnet_resnetd50b_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, InterpolationBlock, MultiOutputSequential, is_channels_first
from .pspnet import PyramidPooling
from .resnetd import resnetd50b


class ICInitBlock(nn.Layer):
    """
    ICNet specific initial block.

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
        super(ICInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=2,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class PSPBlock(nn.Layer):
    """
    ICNet specific PSPNet reduced head block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    upscale_out_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    bottleneck_factor : int
        Bottleneck factor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 upscale_out_size,
                 bottleneck_factor,
                 data_format="channels_last",
                 **kwargs):
        super(PSPBlock, self).__init__(**kwargs)
        assert (in_channels % bottleneck_factor == 0)
        mid_channels = in_channels // bottleneck_factor

        self.pool = PyramidPooling(
            in_channels=in_channels,
            upscale_out_size=upscale_out_size,
            data_format=data_format,
            name="pool")
        self.conv = conv3x3_block(
            in_channels=4096,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv")
        self.dropout = nn.Dropout(
            rate=0.1,
            name="dropout")

    def call(self, x, training=None):
        x = self.pool(x, training=training)
        x = self.conv(x, training=training)
        x = self.dropout(x, training=training)
        return x


class CFFBlock(nn.Layer):
    """
    Cascade Feature Fusion block.

    Parameters:
    ----------
    in_channels_low : int
        Number of input channels (low input).
    in_channels_high : int
        Number of input channels (low high).
    out_channels : int
        Number of output channels.
    classes : int
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels_low,
                 in_channels_high,
                 out_channels,
                 classes,
                 data_format="channels_last",
                 **kwargs):
        super(CFFBlock, self).__init__(**kwargs)
        self.up = InterpolationBlock(
            scale_factor=2,
            data_format=data_format,
            name="up")
        self.conv_low = conv3x3_block(
            in_channels=in_channels_low,
            out_channels=out_channels,
            padding=2,
            dilation=2,
            activation=None,
            data_format=data_format,
            name="conv_low")
        self.conv_hign = conv1x1_block(
            in_channels=in_channels_high,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv_hign")
        self.activ = nn.ReLU()
        self.conv_cls = conv1x1(
            in_channels=out_channels,
            out_channels=classes,
            data_format=data_format,
            name="conv_cls")

    def call(self, xl, xh, training=None):
        xl = self.up(xl)
        xl = self.conv_low(xl, training=training)
        xh = self.conv_hign(xh, training=training)
        x = xl + xh
        x = self.activ(x)
        x_cls = self.conv_cls(xl)
        return x, x_cls


class ICHeadBlock(nn.Layer):
    """
    ICNet head block.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 classes,
                 data_format="channels_last",
                 **kwargs):
        super(ICHeadBlock, self).__init__(**kwargs)
        self.cff_12 = CFFBlock(
            in_channels_low=128,
            in_channels_high=64,
            out_channels=128,
            classes=classes,
            data_format=data_format,
            name="cff_12")
        self.cff_24 = CFFBlock(
            in_channels_low=256,
            in_channels_high=256,
            out_channels=128,
            classes=classes,
            data_format=data_format,
            name="cff_24")
        self.up_x2 = InterpolationBlock(
            scale_factor=2,
            data_format=data_format,
            name="up_x2")
        self.up_x8 = InterpolationBlock(
            scale_factor=4,
            data_format=data_format,
            name="up_x8")
        self.conv_cls = conv1x1(
            in_channels=128,
            out_channels=classes,
            data_format=data_format,
            name="conv_cls")

    def call(self, x1, x2, x4, training=None):
        outputs = []

        x_cff_24, x_24_cls = self.cff_24(x4, x2, training=training)
        outputs.append(x_24_cls)

        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x1, training=training)
        outputs.append(x_12_cls)

        up_x2 = self.up_x2(x_cff_12)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)

        up_x8 = self.up_x8(up_x2)
        outputs.append(up_x8)

        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return tuple(outputs)


class ICNet(tf.keras.Model):
    """
    ICNet model from 'ICNet for Real-Time Semantic Segmentation on High-Resolution Images,'
    https://arxiv.org/abs/1704.08545.

    Parameters:
    ----------
    backbones : tuple of nn.Sequential
        Feature extractors.
    backbones_out_channels : tuple of int
        Number of output channels form each feature extractor.
    classes : tuple of int
        Number of output channels for each branch.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    classes : int, default 21
        Number of segmentation classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 backbones,
                 backbones_out_channels,
                 channels,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 classes=21,
                 data_format="channels_last",
                 **kwargs):
        super(ICNet, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size
        self.data_format = data_format
        psp_pool_out_size = (self.in_size[0] // 32, self.in_size[1] // 32) if fixed_size else None
        psp_head_out_channels = 512

        self.branch1 = ICInitBlock(
            in_channels=in_channels,
            out_channels=channels[0],
            data_format=data_format,
            name="branch1")

        self.branch2 = MultiOutputSequential(name="branch2")
        self.branch2.add(InterpolationBlock(
            scale_factor=2,
            up=False,
            data_format=data_format,
            name="down1"))
        backbones[0].do_output = True
        self.branch2.add(backbones[0])

        self.branch2.add(InterpolationBlock(
            scale_factor=2,
            up=False,
            data_format=data_format,
            name="down2"))
        self.branch2.add(backbones[1])
        self.branch2.add(PSPBlock(
            in_channels=backbones_out_channels[1],
            upscale_out_size=psp_pool_out_size,
            bottleneck_factor=4,
            data_format=data_format,
            name="psp"))
        self.branch2.add(conv1x1_block(
            in_channels=psp_head_out_channels,
            out_channels=channels[2],
            data_format=data_format,
            name="final_block"))

        self.conv_y2 = conv1x1_block(
            in_channels=backbones_out_channels[0],
            out_channels=channels[1],
            data_format=data_format,
            name="conv_y2")

        self.final_block = ICHeadBlock(
            classes=classes,
            data_format=data_format,
            name="final_block")

    def call(self, x, training=None):
        y1 = self.branch1(x, training=training)
        y3, y2 = self.branch2(x, training=training)
        y2 = self.conv_y2(y2, training=training)
        x = self.final_block(y1, y2, y3, training=training)
        if self.aux:
            return x
        else:
            return x[0]


def get_icnet(backbones,
              backbones_out_channels,
              classes,
              aux=False,
              model_name=None,
              data_format="channels_last",
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create ICNet model with specific parameters.

    Parameters:
    ----------
    backbones : tuple of nn.Sequential
        Feature extractors.
    backbones_out_channels : tuple of int
        Number of output channels form each feature extractor.
    classes : int
        Number of segmentation classes.
    aux : bool, default False
        Whether to output an auxiliary result.
    model_name : str or None, default None
        Model name for loading pretrained model.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    channels = (64, 256, 256)

    backbones[0].multi_output = False
    backbones[1].multi_output = False

    net = ICNet(
        backbones=backbones,
        backbones_out_channels=backbones_out_channels,
        channels=channels,
        classes=classes,
        aux=aux,
        data_format=data_format,
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
                local_model_store_dir_path=root),
            by_name=True,
            skip_mismatch=True)

    return net


def icnet_resnetd50b_cityscapes(pretrained_backbone=False, classes=19, aux=True, data_format="channels_last", **kwargs):
    """
    ICNet model on the base of ResNet(D)-50b for Cityscapes from 'ICNet for Real-Time Semantic Segmentation on
    High-Resolution Images,' https://arxiv.org/abs/1704.08545.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone1 = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=None,
                           data_format=data_format).features
    for i in range(len(backbone1) - 3):
        backbone1.children.pop()
    backbone2 = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=None,
                           data_format=data_format).features
    backbone2.children.pop()
    for i in range(3):
        backbone2.children.pop(0)
    backbones = (backbone1, backbone2)
    backbones_out_channels = (512, 2048)
    return get_icnet(backbones=backbones, backbones_out_channels=backbones_out_channels, classes=classes,
                     aux=aux, model_name="icnet_resnetd50b_cityscapes", data_format=data_format, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (480, 480)
    aux = False
    fixed_size = False
    pretrained = False

    models = [
        (icnet_resnetd50b_cityscapes, 19),
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
        assert (model != icnet_resnetd50b_cityscapes or weight_count == 47489184)


if __name__ == "__main__":
    _test()
