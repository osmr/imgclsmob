"""
    SCNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
    http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.
"""

__all__ = ['SCNet', 'scnet50', 'scnet101', 'scneta50', 'scneta101']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, AvgPool2d, InterpolationBlock, SimpleSequential, get_channel_axis,\
    get_im_size, is_channels_first
from .resnet import ResInitBlock
from .senet import SEInitBlock
from .resnesta import ResNeStADownBlock


class ScDownBlock(nn.Layer):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_size=2,
                 data_format="channels_last",
                 **kwargs):
        super(ScDownBlock, self).__init__(**kwargs)
        self.pool = AvgPool2d(
            pool_size=pool_size,
            strides=pool_size,
            data_format=data_format,
            name="pool")
        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.pool(x)
        x = self.conv(x, training=training)
        return x


class ScConv(nn.Layer):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 scale_factor,
                 data_format="channels_last",
                 **kwargs):
        super(ScConv, self).__init__(**kwargs)
        self.data_format = data_format

        self.down = ScDownBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            pool_size=scale_factor,
            data_format=data_format,
            name="down")
        self.up = InterpolationBlock(
            scale_factor=scale_factor,
            interpolation="nearest",
            data_format=data_format,
            name="up")
        self.sigmoid = tf.nn.sigmoid
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=in_channels,
            activation=None,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        in_size = get_im_size(x, data_format=self.data_format)
        w = self.sigmoid(x + self.up(self.down(x, training=training), size=in_size))
        x = self.conv1(x, training=training) * w
        x = self.conv2(x, training=training)
        return x


class ScBottleneck(nn.Layer):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck_factor=4,
                 scale_factor=4,
                 avg_downsample=False,
                 data_format="channels_last",
                 **kwargs):
        super(ScBottleneck, self).__init__(**kwargs)
        self.data_format = data_format
        self.avg_resize = (strides > 1) and avg_downsample
        mid_channels = out_channels // bottleneck_factor // 2

        self.conv1a = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1a")
        self.conv2a = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=(1 if self.avg_resize else strides),
            data_format=data_format,
            name="conv2a")

        self.conv1b = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1b")
        self.conv2b = ScConv(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=(1 if self.avg_resize else strides),
            scale_factor=scale_factor,
            data_format=data_format,
            name="conv2b")

        if self.avg_resize:
            self.pool = AvgPool2d(
                pool_size=3,
                strides=strides,
                padding=1,
                data_format=data_format,
                name="pool")

        self.conv3 = conv1x1_block(
            in_channels=(2 * mid_channels),
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        y = self.conv1a(x, training=training)
        y = self.conv2a(y, training=training)

        z = self.conv1b(x, training=training)
        z = self.conv2b(z, training=training)

        if self.avg_resize:
            y = self.pool(y)
            z = self.pool(z)

        x = tf.concat([y, z], axis=get_channel_axis(self.data_format))

        x = self.conv3(x)
        return x


class ScUnit(nn.Layer):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 avg_downsample=False,
                 data_format="channels_last",
                 **kwargs):
        super(ScUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = ScBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            avg_downsample=avg_downsample,
            data_format=data_format,
            name="body")
        if self.resize_identity:
            if avg_downsample:
                self.identity_block = ResNeStADownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    data_format=data_format,
                    name="identity_block")
            else:
                self.identity_block = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    activation=None,
                    data_format=data_format,
                    name="identity_block")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_block(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class SCNet(tf.keras.Model):
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
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 se_init_block=False,
                 avg_downsample=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(SCNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        init_block_class = SEInitBlock if se_init_block else ResInitBlock
        self.features.add(init_block_class(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(ScUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    avg_downsample=avg_downsample,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.GlobalAvgPool2D(
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        return x


def get_scnet(blocks,
              width_scale=1.0,
              se_init_block=False,
              avg_downsample=False,
              init_block_channels_scale=1,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
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
    root : str, default '~/.tensorflow/models'
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
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def scnet50(**kwargs):
    """
    SCNet-50 model from 'Improving Convolutional Networks with Self-Calibrated Convolutions,'
     http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_scnet(blocks=101, se_init_block=True, avg_downsample=True, init_block_channels_scale=2,
                     model_name="scneta101", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        scnet50,
        scnet101,
        scneta50,
        scneta101,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != scnet50 or weight_count == 25564584)
        assert (model != scnet101 or weight_count == 44565416)
        assert (model != scneta50 or weight_count == 25583816)
        assert (model != scneta101 or weight_count == 44689192)


if __name__ == "__main__":
    _test()
