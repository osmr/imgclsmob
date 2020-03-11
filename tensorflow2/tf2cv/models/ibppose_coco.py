"""
    IBPPose for COCO Keypoint, implemented in TensorFlow.
    Original paper: 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.
"""

__all__ = ['IbpPose', 'ibppose_coco']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import get_activation_layer, MaxPool2d, conv1x1_block, conv3x3_block, conv7x7_block, SEBlock, Hourglass,\
    InterpolationBlock, SimpleSequential, is_channels_first, get_channel_axis


class IbpResBottleneck(nn.Layer):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_bias=False,
                 bottleneck_factor=2,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(IbpResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_bias=use_bias,
            activation=activation,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            use_bias=use_bias,
            activation=activation,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            activation=None,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class IbpResUnit(nn.Layer):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 use_bias=False,
                 bottleneck_factor=2,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(IbpResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = IbpResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            use_bias=use_bias,
            bottleneck_factor=bottleneck_factor,
            activation=activation,
            data_format=data_format,
            name="body")
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                use_bias=use_bias,
                activation=None,
                data_format=data_format,
                name="identity_conv")
        self.activ = get_activation_layer(activation)

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class IbpBackbone(nn.Layer):
    """
    IBPPose backbone.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation,
                 data_format="channels_last",
                 **kwargs):
        super(IbpBackbone, self).__init__(**kwargs)
        self.data_format = data_format
        dilations = (3, 3, 4, 4, 5, 5)
        mid1_channels = out_channels // 4
        mid2_channels = out_channels // 2

        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid1_channels,
            strides=2,
            activation=activation,
            data_format=data_format,
            name="conv1")
        self.res1 = IbpResUnit(
            in_channels=mid1_channels,
            out_channels=mid2_channels,
            activation=activation,
            data_format=data_format,
            name="res1")
        self.pool = MaxPool2d(
            pool_size=2,
            strides=2,
            data_format=data_format,
            name="pool")
        self.res2 = IbpResUnit(
            in_channels=mid2_channels,
            out_channels=mid2_channels,
            activation=activation,
            data_format=data_format,
            name="res2")
        self.dilation_branch = SimpleSequential(name="dilation_branch")
        for i, dilation in enumerate(dilations):
            self.dilation_branch.add(conv3x3_block(
                in_channels=mid2_channels,
                out_channels=mid2_channels,
                padding=dilation,
                dilation=dilation,
                activation=activation,
                data_format=data_format,
                name="block{}".format(i + 1)))

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.res1(x, training=training)
        x = self.pool(x, training=training)
        x = self.res2(x, training=training)
        y = self.dilation_branch(x, training=training)
        x = tf.concat([x, y], axis=get_channel_axis(self.data_format))
        return x


class IbpDownBlock(nn.Layer):
    """
    IBPPose down block for the hourglass.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation,
                 data_format="channels_last",
                 **kwargs):
        super(IbpDownBlock, self).__init__(**kwargs)
        self.down = MaxPool2d(
            pool_size=2,
            strides=2,
            data_format=data_format,
            name="down")
        self.res = IbpResUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            data_format=data_format,
            name="res")

    def call(self, x, training=None):
        x = self.down(x, training=training)
        x = self.res(x, training=training)
        return x


class IbpUpBlock(nn.Layer):
    """
    IBPPose up block for the hourglass.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn,
                 activation,
                 data_format="channels_last",
                 **kwargs):
        super(IbpUpBlock, self).__init__(**kwargs)
        self.res = IbpResUnit(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            data_format=data_format,
            name="res")
        self.up = InterpolationBlock(
            scale_factor=2,
            interpolation="nearest",
            data_format=data_format,
            name="up")
        self.conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bias=(not use_bn),
            use_bn=use_bn,
            activation=activation,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.res(x, training=training)
        x = self.up(x, training=training)
        x = self.conv(x, training=training)
        return x


class MergeBlock(nn.Layer):
    """
    IBPPose merge block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bn : bool
        Whether to use BatchNorm layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn,
                 data_format="channels_last",
                 **kwargs):
        super(MergeBlock, self).__init__(**kwargs)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=(not use_bn),
            use_bn=use_bn,
            activation=None,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        return self.conv(x, training=training)


class IbpPreBlock(nn.Layer):
    """
    IBPPose preliminary decoder block.

    Parameters:
    ----------
    out_channels : int
        Number of output channels.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 out_channels,
                 use_bn,
                 activation,
                 data_format="channels_last",
                 **kwargs):
        super(IbpPreBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bias=(not use_bn),
            use_bn=use_bn,
            activation=activation,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bias=(not use_bn),
            use_bn=use_bn,
            activation=activation,
            data_format=data_format,
            name="conv2")
        self.se = SEBlock(
            channels=out_channels,
            use_conv=False,
            mid_activation=activation,
            data_format=data_format,
            name="se")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.se(x, training=training)
        return x


class IbpPass(nn.Layer):
    """
    IBPPose single pass decoder block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    mid_channels : int
        Number of middle channels.
    depth : int
        Depth of hourglass.
    growth_rate : int
        Addition for number of channel for each level.
    use_bn : bool
        Whether to use BatchNorm layer.
    activation : function or str or None
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 mid_channels,
                 depth,
                 growth_rate,
                 merge,
                 use_bn,
                 activation,
                 data_format="channels_last",
                 **kwargs):
        super(IbpPass, self).__init__(**kwargs)
        self.merge = merge

        down_seq = SimpleSequential(name="down_seq")
        up_seq = SimpleSequential(name="up_seq")
        skip_seq = SimpleSequential(name="skip_seq")
        top_channels = channels
        bottom_channels = channels
        for i in range(depth + 1):
            skip_seq.add(IbpResUnit(
                in_channels=top_channels,
                out_channels=top_channels,
                activation=activation,
                data_format=data_format,
                name="skip{}".format(i + 1)))
            bottom_channels += growth_rate
            if i < depth:
                down_seq.add(IbpDownBlock(
                    in_channels=top_channels,
                    out_channels=bottom_channels,
                    activation=activation,
                    data_format=data_format,
                    name="down{}".format(i + 1)))
                up_seq.add(IbpUpBlock(
                    in_channels=bottom_channels,
                    out_channels=top_channels,
                    use_bn=use_bn,
                    activation=activation,
                    data_format=data_format,
                    name="up{}".format(i + 1)))
            top_channels = bottom_channels
        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            name="hg")

        self.pre_block = IbpPreBlock(
            out_channels=channels,
            use_bn=use_bn,
            activation=activation,
            data_format=data_format,
            name="pre_block")
        self.post_block = conv1x1_block(
            in_channels=channels,
            out_channels=mid_channels,
            use_bias=True,
            use_bn=False,
            activation=None,
            data_format=data_format,
            name="post_block")

        if self.merge:
            self.pre_merge_block = MergeBlock(
                in_channels=channels,
                out_channels=channels,
                use_bn=use_bn,
                data_format=data_format,
                name="pre_merge_block")
            self.post_merge_block = MergeBlock(
                in_channels=mid_channels,
                out_channels=channels,
                use_bn=use_bn,
                data_format=data_format,
                name="post_merge_block")

    def call(self, x, x_prev, training=None):
        x = self.hg(x, training=training)
        if x_prev is not None:
            x = x + x_prev
        y = self.pre_block(x, training=training)
        z = self.post_block(y, training=training)
        if self.merge:
            z = self.post_merge_block(z, training=training) + self.pre_merge_block(y, training=training)
        return z


class IbpPose(tf.keras.Model):
    """
    IBPPose model from 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person Pose Estimation,'
    https://arxiv.org/abs/1911.10529.

    Parameters:
    ----------
    passes : int
        Number of passes.
    backbone_out_channels : int
        Number of output channels for the backbone.
    outs_channels : int
        Number of output channels for the backbone.
    depth : int
        Depth of hourglass.
    growth_rate : int
        Addition for number of channel for each level.
    use_bn : bool
        Whether to use BatchNorm layer.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 256)
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 passes,
                 backbone_out_channels,
                 outs_channels,
                 depth,
                 growth_rate,
                 use_bn,
                 in_channels=3,
                 in_size=(256, 256),
                 data_format="channels_last",
                 **kwargs):
        super(IbpPose, self).__init__(**kwargs)
        self.in_size = in_size
        self.data_format = data_format
        activation = nn.LeakyReLU(alpha=0.01)

        self.backbone = IbpBackbone(
            in_channels=in_channels,
            out_channels=backbone_out_channels,
            activation=activation,
            data_format=data_format,
            name="backbone")

        self.decoder = SimpleSequential(name="decoder")
        for i in range(passes):
            merge = (i != passes - 1)
            self.decoder.add(IbpPass(
                channels=backbone_out_channels,
                mid_channels=outs_channels,
                depth=depth,
                growth_rate=growth_rate,
                merge=merge,
                use_bn=use_bn,
                activation=activation,
                data_format=data_format,
                name="pass{}".format(i + 1)))

    def call(self, x, training=None):
        x = self.backbone(x, training=training)
        x_prev = None
        for block in self.decoder.children:
            if x_prev is not None:
                x = x + x_prev
            x_prev = block(x, x_prev, training=training)
        return x_prev


def get_ibppose(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create IBPPose model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    passes = 4
    backbone_out_channels = 256
    outs_channels = 50
    depth = 4
    growth_rate = 128
    use_bn = True

    net = IbpPose(
        passes=passes,
        backbone_out_channels=backbone_out_channels,
        outs_channels=outs_channels,
        depth=depth,
        growth_rate=growth_rate,
        use_bn=use_bn,
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


def ibppose_coco(data_format="channels_last", **kwargs):
    """
    IBPPose model for COCO Keypoint from 'Simple Pose: Rethinking and Improving a Bottom-up Approach for Multi-Person
    Pose Estimation,' https://arxiv.org/abs/1911.10529.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_ibppose(model_name="ibppose_coco", data_format=data_format, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    # os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (256, 256)
    pretrained = False

    models = [
        ibppose_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (y.shape[0] == batch)
        if is_channels_first(data_format):
            assert ((y.shape[1] == 50) and (y.shape[2] == x.shape[2] // 4) and
                    (y.shape[3] == x.shape[3] // 4))
        else:
            assert ((y.shape[3] == 50) and (y.shape[1] == x.shape[1] // 4) and
                    (y.shape[2] == x.shape[2] // 4))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != ibppose_coco or weight_count == 95827784)


if __name__ == "__main__":
    _test()
