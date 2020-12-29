"""
    DPN for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.
"""

__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn98', 'dpn107', 'dpn131']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MaxPool2d, GlobalAvgPool2d, BatchNorm, Conv2d, conv1x1, DualPathSequential, SimpleSequential,\
    flatten, is_channels_first, get_channel_axis


class GlobalAvgMaxPool2D(nn.Layer):
    """
    Global average+max pooling operation for spatial data.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(GlobalAvgMaxPool2D, self).__init__(**kwargs)
        self.axis = get_channel_axis(data_format)

        self.avg_pool = nn.GlobalAvgPool2D(
            data_format=data_format,
            name="avg_pool")
        self.max_pool = nn.GlobalMaxPool2D(
            data_format=data_format,
            name="max_pool")

    def call(self, x, training=None):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = 0.5 * (x_avg + x_max)
        x = tf.expand_dims(tf.expand_dims(x, axis=self.axis), axis=self.axis)
        return x


def dpn_batch_norm(channels,
                   data_format="channels_last",
                   **kwargs):
    """
    DPN specific Batch normalization layer.

    Parameters:
    ----------
    channels : int
        Number of channels in input data.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    assert (channels is not None)
    return BatchNorm(
        epsilon=0.001,
        data_format=data_format,
        **kwargs)


class PreActivation(nn.Layer):
    """
    DPN specific block, which performs the preactivation like in RreResNet.

    Parameters:
    ----------
    channels : int
        Number of channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 data_format="channels_last",
                 **kwargs):
        super(PreActivation, self).__init__(**kwargs)
        self.bn = dpn_batch_norm(
            channels=channels,
            data_format=data_format,
            name="bn")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        x = self.bn(x, training=training)
        x = self.activ(x)
        return x


class DPNConv(nn.Layer):
    """
    DPN specific convolution block.

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
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups,
                 data_format="channels_last",
                 **kwargs):
        super(DPNConv, self).__init__(**kwargs)
        self.bn = dpn_batch_norm(
            channels=in_channels,
            data_format=data_format,
            name="bn")
        self.activ = nn.ReLU()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=False,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.bn(x, training=training)
        x = self.activ(x)
        x = self.conv(x)
        return x


def dpn_conv1x1(in_channels,
                out_channels,
                strides=1,
                data_format="channels_last",
                **kwargs):
    """
    1x1 version of the DPN specific convolution block.

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
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=1,
        data_format=data_format,
        **kwargs)


def dpn_conv3x3(in_channels,
                out_channels,
                strides,
                groups,
                data_format="channels_last",
                **kwargs):
    """
    3x3 version of the DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int
        Number of groups.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=groups,
        data_format=data_format,
        **kwargs)


class DPNUnit(nn.Layer):
    """
    DPN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of intermediate channels.
    bw : int
        Number of residual channels.
    inc : int
        Incrementing step for channels.
    groups : int
        Number of groups in the units.
    has_proj : bool
        Whether to use projection.
    key_strides : int
        Key strides of the convolutions.
    b_case : bool, default False
        Whether to use B-case model.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 bw,
                 inc,
                 groups,
                 has_proj,
                 key_strides,
                 b_case=False,
                 data_format="channels_last",
                 **kwargs):
        super(DPNUnit, self).__init__(**kwargs)
        self.bw = bw
        self.has_proj = has_proj
        self.b_case = b_case
        self.data_format = data_format

        if self.has_proj:
            self.conv_proj = dpn_conv1x1(
                in_channels=in_channels,
                out_channels=bw + 2 * inc,
                strides=key_strides,
                data_format=data_format,
                name="conv_proj")

        self.conv1 = dpn_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = dpn_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=key_strides,
            groups=groups,
            data_format=data_format,
            name="conv2")

        if b_case:
            self.preactiv = PreActivation(
                channels=mid_channels,
                data_format=data_format,
                name="preactiv")
            self.conv3a = conv1x1(
                in_channels=mid_channels,
                out_channels=bw,
                data_format=data_format,
                name="conv3a")
            self.conv3b = conv1x1(
                in_channels=mid_channels,
                out_channels=inc,
                data_format=data_format,
                name="conv3b")
        else:
            self.conv3 = dpn_conv1x1(
                in_channels=mid_channels,
                out_channels=bw + inc,
                data_format=data_format,
                name="conv3")

    def call(self, x1, x2=None, training=None):
        axis = get_channel_axis(self.data_format)
        x_in = tf.concat([x1, x2], axis=axis) if x2 is not None else x1
        if self.has_proj:
            x_s = self.conv_proj(x_in, training=training)
            channels = (x_s.get_shape().as_list())[axis]
            x_s1, x_s2 = tf.split(x_s, num_or_size_splits=[self.bw, channels - self.bw], axis=axis)
            # x_s1 = F.slice_axis(x_s, axis=1, begin=0, end=self.bw)
            # x_s2 = F.slice_axis(x_s, axis=1, begin=self.bw, end=None)
        else:
            assert (x2 is not None)
            x_s1 = x1
            x_s2 = x2
        x_in = self.conv1(x_in, training=training)
        x_in = self.conv2(x_in, training=training)
        if self.b_case:
            x_in = self.preactiv(x_in, training=training)
            y1 = self.conv3a(x_in, training=training)
            y2 = self.conv3b(x_in, training=training)
        else:
            x_in = self.conv3(x_in, training=training)
            # y1 = F.slice_axis(x_in, axis=1, begin=0, end=self.bw)
            # y2 = F.slice_axis(x_in, axis=1, begin=self.bw, end=None)
            channels = (x_in.get_shape().as_list())[axis]
            y1, y2 = tf.split(x_in, num_or_size_splits=[self.bw, channels - self.bw], axis=axis)
        residual = x_s1 + y1
        dense = tf.concat([x_s2, y2], axis=axis)
        return residual, dense


class DPNInitBlock(nn.Layer):
    """
    DPN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 data_format="channels_last",
                 **kwargs):
        super(DPNInitBlock, self).__init__(**kwargs)
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=2,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = dpn_batch_norm(
            channels=out_channels,
            data_format=data_format,
            name="bn")
        self.activ = nn.ReLU()
        self.pool = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name="pool")

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.activ(x)
        x = self.pool(x)
        return x


class DPNFinalBlock(nn.Layer):
    """
    DPN final block, which performs the preactivation with cutting.

    Parameters:
    ----------
    channels : int
        Number of channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 data_format="channels_last",
                 **kwargs):
        super(DPNFinalBlock, self).__init__(**kwargs)
        self.data_format = data_format

        self.activ = PreActivation(
            channels=channels,
            data_format=data_format,
            name="activ")

    def call(self, x1, x2, training=None):
        assert (x2 is not None)
        x = tf.concat([x1, x2], axis=get_channel_axis(self.data_format))
        x = self.activ(x)
        return x, None


class DPN(tf.keras.Model):
    """
    DPN model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    init_block_kernel_size : int or tuple/list of 2 int
        Convolution window size for the initial unit.
    init_block_padding : int or tuple/list of 2 int
        Padding value for convolution layer in the initial unit.
    rs : list f int
        Number of intermediate channels for each unit.
    bws : list f int
        Number of residual channels for each unit.
    incs : list f int
        Incrementing step for channels for each unit.
    groups : int
        Number of groups in the units.
    b_case : bool
        Whether to use B-case model.
    for_training : bool
        Whether to use model for training.
    test_time_pool : bool
        Whether to use the avg-max pooling in the inference mode.
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
                 init_block_kernel_size,
                 init_block_padding,
                 rs,
                 bws,
                 incs,
                 groups,
                 b_case,
                 for_training,
                 test_time_pool,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DPN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=0,
            name="features")
        self.features.children.append(DPNInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=init_block_kernel_size,
            padding=init_block_padding,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential(name="stage{}".format(i + 1))
            r = rs[i]
            bw = bws[i]
            inc = incs[i]
            for j, out_channels in enumerate(channels_per_stage):
                has_proj = (j == 0)
                key_strides = 2 if (j == 0) and (i != 0) else 1
                stage.children.append(DPNUnit(
                    in_channels=in_channels,
                    mid_channels=r,
                    bw=bw,
                    inc=inc,
                    groups=groups,
                    has_proj=has_proj,
                    key_strides=key_strides,
                    b_case=b_case,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.children.append(stage)
        self.features.children.append(DPNFinalBlock(
            channels=in_channels,
            data_format=data_format,
            name="final_block"))

        self.output1 = SimpleSequential(name="output1")
        if for_training or not test_time_pool:
            self.output1.add(GlobalAvgPool2d(
                data_format=data_format,
                name="final_pool"))
            self.output1.add(conv1x1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True,
                data_format=data_format,
                name="classifier"))
        else:
            self.output1.add(nn.AveragePooling2D(
                pool_size=7,
                strides=1,
                data_format=data_format,
                name="avg_pool"))
            self.output1.add(conv1x1(
                in_channels=in_channels,
                out_channels=classes,
                use_bias=True,
                data_format=data_format,
                name="classifier"))
            self.output1.add(GlobalAvgMaxPool2D(
                data_format=data_format,
                name="avgmax_pool"))

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        x = flatten(x, self.data_format)
        return x


def get_dpn(num_layers,
            b_case=False,
            for_training=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".tensorflow", "models"),
            **kwargs):
    """
    Create DPN model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    b_case : bool, default False
        Whether to use B-case model.
    for_training : bool
        Whether to use model for training.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if num_layers == 68:
        init_block_channels = 10
        init_block_kernel_size = 3
        init_block_padding = 1
        bw_factor = 1
        k_r = 128
        groups = 32
        k_sec = (3, 4, 12, 3)
        incs = (16, 32, 32, 64)
        test_time_pool = True
    elif num_layers == 98:
        init_block_channels = 96
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 160
        groups = 40
        k_sec = (3, 6, 20, 3)
        incs = (16, 32, 32, 128)
        test_time_pool = True
    elif num_layers == 107:
        init_block_channels = 128
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 200
        groups = 50
        k_sec = (4, 8, 20, 3)
        incs = (20, 64, 64, 128)
        test_time_pool = True
    elif num_layers == 131:
        init_block_channels = 128
        init_block_kernel_size = 7
        init_block_padding = 3
        bw_factor = 4
        k_r = 160
        groups = 40
        k_sec = (4, 8, 28, 3)
        incs = (16, 32, 32, 128)
        test_time_pool = True
    else:
        raise ValueError("Unsupported DPN version with number of layers {}".format(num_layers))

    channels = [[0] * li for li in k_sec]
    rs = [0 * li for li in k_sec]
    bws = [0 * li for li in k_sec]
    for i in range(len(k_sec)):
        rs[i] = (2 ** i) * k_r
        bws[i] = (2 ** i) * 64 * bw_factor
        inc = incs[i]
        channels[i][0] = bws[i] + 3 * inc
        for j in range(1, k_sec[i]):
            channels[i][j] = channels[i][j - 1] + inc

    net = DPN(
        channels=channels,
        init_block_channels=init_block_channels,
        init_block_kernel_size=init_block_kernel_size,
        init_block_padding=init_block_padding,
        rs=rs,
        bws=bws,
        incs=incs,
        groups=groups,
        b_case=b_case,
        for_training=for_training,
        test_time_pool=test_time_pool,
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


def dpn68(**kwargs):
    """
    DPN-68 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=68, b_case=False, model_name="dpn68", **kwargs)


def dpn68b(**kwargs):
    """
    DPN-68b model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=68, b_case=True, model_name="dpn68b", **kwargs)


def dpn98(**kwargs):
    """
    DPN-98 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=98, b_case=False, model_name="dpn98", **kwargs)


def dpn107(**kwargs):
    """
    DPN-107 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=107, b_case=False, model_name="dpn107", **kwargs)


def dpn131(**kwargs):
    """
    DPN-131 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=131, b_case=False, model_name="dpn131", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        dpn68,
        dpn68b,
        dpn98,
        dpn107,
        dpn131,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dpn68 or weight_count == 12611602)
        assert (model != dpn68b or weight_count == 12611602)
        assert (model != dpn98 or weight_count == 61570728)
        assert (model != dpn107 or weight_count == 86917800)
        assert (model != dpn131 or weight_count == 79254504)


if __name__ == "__main__":
    _test()
