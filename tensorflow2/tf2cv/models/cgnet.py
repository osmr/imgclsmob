"""
    CGNet for image segmentation, implemented in TensorFlow.
    Original paper: 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.
"""

__all__ = ['CGNet', 'cgnet_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import NormActivation, conv1x1, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock, Concurrent,\
    DualPathSequential, InterpolationBlock, SimpleSequential, is_channels_first, get_im_size, PReLU2, AvgPool2d,\
    get_channel_axis


class CGBlock(nn.Layer):
    """
    CGNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilation : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    down : bool
        Whether to downsample.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 se_reduction,
                 down,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(CGBlock, self).__init__(**kwargs)
        self.down = down
        if self.down:
            mid1_channels = out_channels
            mid2_channels = 2 * out_channels
        else:
            mid1_channels = out_channels // 2
            mid2_channels = out_channels

        if self.down:
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_eps=bn_eps,
                activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
                data_format=data_format,
                name="conv1")
        else:
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid1_channels,
                bn_eps=bn_eps,
                activation=(lambda: PReLU2(mid1_channels, data_format=data_format, name="activ")),
                data_format=data_format,
                name="conv1")

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(depthwise_conv3x3(
            channels=mid1_channels,
            data_format=data_format,
            name="branches1"))
        self.branches.add(depthwise_conv3x3(
            channels=mid1_channels,
            padding=dilation,
            dilation=dilation,
            data_format=data_format,
            name="branches2"))

        self.norm_activ = NormActivation(
            in_channels=mid2_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(mid2_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="norm_activ")

        if self.down:
            self.conv2 = conv1x1(
                in_channels=mid2_channels,
                out_channels=out_channels,
                data_format=data_format,
                name="conv2")

        self.se = SEBlock(
            channels=out_channels,
            reduction=se_reduction,
            use_conv=False,
            data_format=data_format,
            name="se")

    def call(self, x, training=None):
        if not self.down:
            identity = x
        x = self.conv1(x, training=training)
        x = self.branches(x, training=training)
        x = self.norm_activ(x, training=training)
        if self.down:
            x = self.conv2(x, training=training)
        x = self.se(x, training=training)
        if not self.down:
            x += identity
        return x


class CGUnit(nn.Layer):
    """
    CGNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    dilation : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(CGUnit, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        mid_channels = out_channels // 2

        self.down = CGBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            dilation=dilation,
            se_reduction=se_reduction,
            down=True,
            bn_eps=bn_eps,
            data_format=data_format,
            name="down")
        self.blocks = SimpleSequential(name="blocks")
        for i in range(layers - 1):
            self.blocks.add(CGBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                dilation=dilation,
                se_reduction=se_reduction,
                down=False,
                bn_eps=bn_eps,
                data_format=data_format,
                name="block{}".format(i + 1)))

    def call(self, x, training=None):
        x = self.down(x, training=training)
        y = self.blocks(x, training=training)
        x = tf.concat([y, x], axis=self.axis)  # NB: This differs from the original implementation.
        return x


class CGStage(nn.Layer):
    """
    CGNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    layers : int
        Number of layers in the unit.
    dilation : int
        Dilation for blocks.
    se_reduction : int
        SE-block reduction value for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(CGStage, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)
        self.use_x = (x_channels > 0)
        self.use_unit = (layers > 0)

        if self.use_x:
            self.x_down = AvgPool2d(
                pool_size=3,
                strides=2,
                padding=1,
                data_format=data_format,
                name="x_down")

        if self.use_unit:
            self.unit = CGUnit(
                in_channels=y_in_channels,
                out_channels=(y_out_channels - x_channels),
                layers=layers,
                dilation=dilation,
                se_reduction=se_reduction,
                bn_eps=bn_eps,
                data_format=data_format,
                name="unit")

        self.norm_activ = NormActivation(
            in_channels=y_out_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(y_out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="norm_activ")

    def call(self, y, x=None, training=None):
        if self.use_unit:
            y = self.unit(y, training=training)
        if self.use_x:
            x = self.x_down(x)
            y = tf.concat([y, x], axis=self.axis)
        y = self.norm_activ(y, training=training)
        return y, x


class CGInitBlock(nn.Layer):
    """
    CGNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps,
                 data_format="channels_last",
                 **kwargs):
        super(CGInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=2,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: PReLU2(out_channels, data_format=data_format, name="activ")),
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class CGNet(tf.keras.Model):
    """
    CGNet model from 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of int
        Dilations for each unit.
    se_reductions : list of int
        SE-block reduction value for each unit.
    cut_x : list of int
        Whether to concatenate with x-branch for each unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    classes : int, default 19
        Number of segmentation classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 layers,
                 channels,
                 init_block_channels,
                 dilations,
                 se_reductions,
                 cut_x,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 classes=19,
                 data_format="channels_last",
                 **kwargs):
        super(CGNet, self).__init__(**kwargs)
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.fixed_size = fixed_size
        self.data_format = data_format

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=0,
            name="features")
        self.features.add(CGInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps,
            data_format=data_format,
            name="init_block"))
        y_in_channels = init_block_channels

        for i, (layers_i, y_out_channels) in enumerate(zip(layers, channels)):
            self.features.add(CGStage(
                x_channels=in_channels if cut_x[i] == 1 else 0,
                y_in_channels=y_in_channels,
                y_out_channels=y_out_channels,
                layers=layers_i,
                dilation=dilations[i],
                se_reduction=se_reductions[i],
                bn_eps=bn_eps,
                data_format=data_format,
                name="stage{}".format(i + 1)))
            y_in_channels = y_out_channels

        self.classifier = conv1x1(
            in_channels=y_in_channels,
            out_channels=classes,
            data_format=data_format,
            name="classifier")

        self.up = InterpolationBlock(
            scale_factor=8,
            data_format=data_format,
            name="up")

    def call(self, x, training=None):
        in_size = self.in_size if self.fixed_size else get_im_size(x, data_format=self.data_format)
        y = self.features(x, x, training=training)
        y = self.classifier(y)
        y = self.up(y, size=in_size)
        return y


def get_cgnet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create CGNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [0, 3, 21]
    channels = [35, 131, 256]
    dilations = [0, 2, 4]
    se_reductions = [0, 8, 16]
    cut_x = [1, 1, 0]
    bn_eps = 1e-3

    net = CGNet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
        se_reductions=se_reductions,
        cut_x=cut_x,
        bn_eps=bn_eps,
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


def cgnet_cityscapes(classes=19, **kwargs):
    """
    CGNet model for Cityscapes from 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.

    Parameters:
    ----------
    classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_cgnet(classes=classes, model_name="cgnet_cityscapes", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False
    in_size = (1024, 2048)
    classes = 19

    models = [
        cgnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, data_format=data_format)

        batch = 4
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, classes, in_size[0], in_size[1]) if is_channels_first(data_format)
                else tuple(y.shape.as_list()) == (batch, in_size[0], in_size[1], classes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cgnet_cityscapes or weight_count == 496306)


if __name__ == "__main__":
    _test()
