"""
    PNASNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Progressive Neural Architecture Search,' https://arxiv.org/abs/1712.00559.
"""

__all__ = ['PNASNet', 'pnasnet5large']


import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, conv1x1, SimpleSequential, flatten, is_channels_first, get_channel_axis
from .nasnet import nasnet_dual_path_sequential, nasnet_batch_norm, NasConv, NasDwsConv, NasPathBlock, NASNetInitBlock


class PnasMaxPoolBlock(nn.Layer):
    """
    PNASNet specific Max pooling layer with extra padding.

    Parameters:
    ----------
    strides : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 strides=2,
                 extra_padding=False,
                 data_format="channels_last",
                 **kwargs):
        super(PnasMaxPoolBlock, self).__init__(**kwargs)
        self.extra_padding = extra_padding
        self.data_format = data_format

        self.pool = MaxPool2d(
            pool_size=3,
            strides=strides,
            padding=1,
            data_format=data_format,
            name="pool")
        if self.extra_padding:
            self.pad = nn.ZeroPadding2D(
                padding=((1, 0), (1, 0)),
                data_format=data_format)

    def call(self, x, training=None):
        if self.extra_padding:
            x = self.pad(x)
        x = self.pool(x)
        if self.extra_padding:
            if is_channels_first(self.data_format):
                x = x[:, :, 1:, 1:]
            else:
                x = x[:, 1:, 1:, :]
        return x


def pnas_conv1x1(in_channels,
                 out_channels,
                 strides=1,
                 data_format="channels_last",
                 **kwargs):
    """
    1x1 version of the PNASNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return NasConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=1,
        data_format=data_format,
        **kwargs)


class DwsBranch(nn.Layer):
    """
    PNASNet specific block with depthwise separable convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 extra_padding=False,
                 stem=False,
                 data_format="channels_last",
                 **kwargs):
        super(DwsBranch, self).__init__(**kwargs)
        assert (not stem) or (not extra_padding)
        mid_channels = out_channels if stem else in_channels
        padding = kernel_size // 2

        self.conv1 = NasDwsConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            extra_padding=extra_padding,
            data_format=data_format,
            name="conv1")
        self.conv2 = NasDwsConv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


def dws_branch_k3(in_channels,
                  out_channels,
                  strides=2,
                  extra_padding=False,
                  stem=False,
                  data_format="channels_last",
                  **kwargs):
    """
    3x3 version of the PNASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        extra_padding=extra_padding,
        stem=stem,
        data_format=data_format,
        **kwargs)


def dws_branch_k5(in_channels,
                  out_channels,
                  strides=2,
                  extra_padding=False,
                  stem=False,
                  data_format="channels_last",
                  **kwargs):
    """
    5x5 version of the PNASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    stem : bool, default False
        Whether to use squeeze reduction if False.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        strides=strides,
        extra_padding=extra_padding,
        stem=stem,
        data_format=data_format,
        **kwargs)


def dws_branch_k7(in_channels,
                  out_channels,
                  strides=2,
                  extra_padding=False,
                  data_format="channels_last",
                  **kwargs):
    """
    7x7 version of the PNASNet specific depthwise separable convolution branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    extra_padding : bool, default False
        Whether to use extra padding.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DwsBranch(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        extra_padding=extra_padding,
        stem=False,
        data_format=data_format,
        **kwargs)


class PnasMaxPathBlock(nn.Layer):
    """
    PNASNet specific `max path` auxiliary block.

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
        super(PnasMaxPathBlock, self).__init__(**kwargs)
        self.maxpool = PnasMaxPoolBlock(
            data_format=data_format,
            name="maxpool")
        self.conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")
        self.bn = nasnet_batch_norm(
            channels=out_channels,
            data_format=data_format,
            name="bn")

    def call(self, x, training=None):
        x = self.maxpool(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        return x


class PnasBaseUnit(nn.Layer):
    """
    PNASNet base unit.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(PnasBaseUnit, self).__init__(**kwargs)
        self.data_format = data_format

    def cell_forward(self, x, x_prev, training=None):
        assert (hasattr(self, 'comb0_left'))
        x_left = x_prev
        x_right = x

        x0 = self.comb0_left(x_left, training=training) + self.comb0_right(x_left, training=training)
        x1 = self.comb1_left(x_right, training=training) + self.comb1_right(x_right, training=training)
        x2 = self.comb2_left(x_right, training=training) + self.comb2_right(x_right, training=training)
        x3 = self.comb3_left(x2, training=training) + self.comb3_right(x_right, training=training)
        x4 = self.comb4_left(x_left, training=training) + (self.comb4_right(x_right, training=training) if
                                                           self.comb4_right else x_right)

        x_out = tf.concat([x0, x1, x2, x3, x4], axis=get_channel_axis(self.data_format))
        return x_out


class Stem1Unit(PnasBaseUnit):
    """
    PNASNet Stem1 unit.

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
        super(Stem1Unit, self).__init__(**kwargs)
        mid_channels = out_channels // 5

        self.conv_1x1 = pnas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv_1x1")

        self.comb0_left = dws_branch_k5(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True,
            data_format=data_format,
            name="comb0_left")
        self.comb0_right = PnasMaxPathBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb0_right")

        self.comb1_left = dws_branch_k7(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb1_left")
        self.comb1_right = PnasMaxPoolBlock(
            data_format=data_format,
            name="comb1_right")

        self.comb2_left = dws_branch_k5(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb2_left")
        self.comb2_right = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="comb2_right")

        self.comb3_left = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=1,
            data_format=data_format,
            name="comb3_left")
        self.comb3_right = PnasMaxPoolBlock(
            data_format=data_format,
            name="comb3_right")

        self.comb4_left = dws_branch_k3(
            in_channels=in_channels,
            out_channels=mid_channels,
            stem=True,
            data_format=data_format,
            name="comb4_left")
        self.comb4_right = pnas_conv1x1(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="comb4_right")

    def call(self, x, training=None):
        x_prev = x
        x = self.conv_1x1(x, training=training)
        x_out = self.cell_forward(x, x_prev, training=training)
        return x_out


class PnasUnit(PnasBaseUnit):
    """
    PNASNet ordinary unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    reduction : bool, default False
        Whether to use reduction.
    extra_padding : bool, default False
        Whether to use extra padding.
    match_prev_layer_dimensions : bool, default False
        Whether to match previous layer dimensions.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 reduction=False,
                 extra_padding=False,
                 match_prev_layer_dimensions=False,
                 data_format="channels_last",
                 **kwargs):
        super(PnasUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 5
        stride = 2 if reduction else 1

        if match_prev_layer_dimensions:
            self.conv_prev_1x1 = NasPathBlock(
                in_channels=prev_in_channels,
                out_channels=mid_channels,
                data_format=data_format,
                name="conv_prev_1x1")
        else:
            self.conv_prev_1x1 = pnas_conv1x1(
                in_channels=prev_in_channels,
                out_channels=mid_channels,
                data_format=data_format,
                name="conv_prev_1x1")

        self.conv_1x1 = pnas_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv_1x1")

        self.comb0_left = dws_branch_k5(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb0_left")
        self.comb0_right = PnasMaxPoolBlock(
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb0_right")

        self.comb1_left = dws_branch_k7(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb1_left")
        self.comb1_right = PnasMaxPoolBlock(
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb1_right")

        self.comb2_left = dws_branch_k5(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb2_left")
        self.comb2_right = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb2_right")

        self.comb3_left = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=1,
            data_format=data_format,
            name="comb3_left")
        self.comb3_right = PnasMaxPoolBlock(
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb3_right")

        self.comb4_left = dws_branch_k3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=stride,
            extra_padding=extra_padding,
            data_format=data_format,
            name="comb4_left")
        if reduction:
            self.comb4_right = pnas_conv1x1(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=stride,
                data_format=data_format,
                name="comb4_right")
        else:
            self.comb4_right = None

    def call(self, x, x_prev, training=None):
        x_prev = self.conv_prev_1x1(x_prev, training=training)
        x = self.conv_1x1(x, training=training)
        x_out = self.cell_forward(x, x_prev, training=training)
        return x_out


class PNASNet(tf.keras.Model):
    """
    PNASNet model from 'Progressive Neural Architecture Search,' https://arxiv.org/abs/1712.00559.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    stem1_blocks_channels : list of 2 int
        Number of output channels for the Stem1 unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (331, 331)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 stem1_blocks_channels,
                 in_channels=3,
                 in_size=(331, 331),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(PNASNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = nasnet_dual_path_sequential(
            return_two=False,
            first_ordinals=2,
            last_ordinals=2,
            name="features")
        self.features.add(NASNetInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels

        self.features.add(Stem1Unit(
            in_channels=in_channels,
            out_channels=stem1_blocks_channels,
            data_format=data_format,
            name="stem1_unit"))
        prev_in_channels = in_channels
        in_channels = stem1_blocks_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nasnet_dual_path_sequential(
                name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                reduction = (j == 0)
                extra_padding = (j == 0) and (i not in [0, 2])
                match_prev_layer_dimensions = (j == 1) or ((j == 0) and (i == 0))
                stage.add(PnasUnit(
                    in_channels=in_channels,
                    prev_in_channels=prev_in_channels,
                    out_channels=out_channels,
                    reduction=reduction,
                    extra_padding=extra_padding,
                    match_prev_layer_dimensions=match_prev_layer_dimensions,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                prev_in_channels = in_channels
                in_channels = out_channels
            self.features.add(stage)

        self.features.add(nn.ReLU(name="activ"))
        self.features.add(nn.AveragePooling2D(
            pool_size=11,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = SimpleSequential(name="output1")
        self.output1.add(nn.Dropout(
            rate=0.5,
            name="dropout"))
        self.output1.add(nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="fc"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_pnasnet(model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create PNASNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    repeat = 4
    init_block_channels = 96
    stem_blocks_channels = [270, 540]
    norm_channels = [1080, 2160, 4320]
    channels = [[ci] * repeat for ci in norm_channels]
    stem1_blocks_channels = stem_blocks_channels[0]
    channels[0] = [stem_blocks_channels[1]] + channels[0]

    net = PNASNet(
        channels=channels,
        init_block_channels=init_block_channels,
        stem1_blocks_channels=stem1_blocks_channels,
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


def pnasnet5large(**kwargs):
    """
    PNASNet-5-Large model from 'Progressive Neural Architecture Search,' https://arxiv.org/abs/1712.00559.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_pnasnet(model_name="pnasnet5large", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        pnasnet5large,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 331, 331) if is_channels_first(data_format) else (batch, 331, 331, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != pnasnet5large or weight_count == 86057668)


if __name__ == "__main__":
    _test()
