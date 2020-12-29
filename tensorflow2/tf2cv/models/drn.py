"""
    DRN for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.
"""

__all__ = ['DRN', 'drnc26', 'drnc42', 'drnc58', 'drnd22', 'drnd38', 'drnd54', 'drnd105']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import Conv2d, BatchNorm, SimpleSequential, flatten, is_channels_first


class DRNConv(nn.Layer):
    """
    DRN specific convolution block.

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
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation,
                 activate,
                 data_format="channels_last",
                 **kwargs):
        super(DRNConv, self).__init__(**kwargs)
        self.activate = activate

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            use_bias=False,
            data_format=data_format,
            name="conv")
        self.bn = BatchNorm(
            data_format=data_format,
            name="bn")
        if self.activate:
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


def drn_conv1x1(in_channels,
                out_channels,
                strides,
                activate,
                data_format="channels_last",
                **kwargs):
    """
    1x1 version of the DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        dilation=1,
        activate=activate,
        data_format=data_format,
        **kwargs)


def drn_conv3x3(in_channels,
                out_channels,
                strides,
                dilation,
                activate,
                data_format="channels_last",
                **kwargs):
    """
    3x3 version of the DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=dilation,
        dilation=dilation,
        activate=activate,
        data_format=data_format,
        **kwargs)


class DRNBlock(nn.Layer):
    """
    Simple DRN block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for convolution layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation,
                 data_format="channels_last",
                 **kwargs):
        super(DRNBlock, self).__init__(**kwargs)
        self.conv1 = drn_conv3x3(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=stride,
            dilation=dilation,
            activate=True,
            data_format=data_format,
            name="conv1")
        self.conv2 = drn_conv3x3(
            in_channels=out_channels,
            out_channels=out_channels,
            strides=1,
            dilation=dilation,
            activate=False,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class DRNBottleneck(nn.Layer):
    """
    DRN bottleneck block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layer.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 dilation,
                 data_format="channels_last",
                 **kwargs):
        super(DRNBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        self.conv1 = drn_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=1,
            activate=True,
            data_format=data_format,
            name="conv1")
        self.conv2 = drn_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            dilation=dilation,
            activate=True,
            data_format=data_format,
            name="conv2")
        self.conv3 = drn_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=1,
            activate=False,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class DRNUnit(nn.Layer):
    """
    DRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layers.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    simplified : bool
        Whether to use a simple or simplified block in units.
    residual : bool
        Whether do residual calculations.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 dilation,
                 bottleneck,
                 simplified,
                 residual,
                 data_format="channels_last",
                 **kwargs):
        super(DRNUnit, self).__init__(**kwargs)
        assert residual or (not bottleneck)
        assert (not (bottleneck and simplified))
        assert (not (residual and simplified))
        self.residual = residual
        self.resize_identity = ((in_channels != out_channels) or (strides != 1)) and self.residual and (not simplified)

        if bottleneck:
            self.body = DRNBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dilation=dilation,
                data_format=data_format,
                name="body")
        elif simplified:
            self.body = drn_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                dilation=dilation,
                activate=False,
                data_format=data_format,
                name="body")
        else:
            self.body = DRNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=strides,
                dilation=dilation,
                data_format=data_format,
                name="body")
        if self.resize_identity:
            self.identity_conv = drn_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activate=False,
                data_format=data_format,
                name="identity_conv")
        self.activ = nn.ReLU()

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        if self.residual:
            x = x + identity
        x = self.activ(x)
        return x


def drn_init_block(in_channels,
                   out_channels,
                   data_format="channels_last",
                   **kwargs):
    """
    DRN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=1,
        padding=3,
        dilation=1,
        activate=True,
        data_format=data_format,
        **kwargs)


class DRN(tf.keras.Model):
    """
    DRN-C&D model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of list of int
        Dilation values for 3x3 convolution layers for each unit.
    bottlenecks : list of list of int
        Whether to use a bottleneck or simple block in each unit.
    simplifieds : list of list of int
        Whether to use a simple or simplified block in each unit.
    residuals : list of list of int
        Whether to use residual block in each unit.
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
                 dilations,
                 bottlenecks,
                 simplifieds,
                 residuals,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DRN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(drn_init_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(DRNUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    dilation=dilations[i][j],
                    bottleneck=(bottlenecks[i][j] == 1),
                    simplified=(simplifieds[i][j] == 1),
                    residual=(residuals[i][j] == 1),
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=28,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = Conv2d(
            in_channels=in_channels,
            out_channels=classes,
            kernel_size=1,
            data_format=data_format,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        x = flatten(x, self.data_format)
        return x


def get_drn(blocks,
            simplified=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".tensorflow", "models"),
            **kwargs):
    """
    Create DRN-C or DRN-D model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    simplified : bool, default False
        Whether to use simplified scheme (D architecture).
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if blocks == 22:
        assert simplified
        layers = [1, 1, 2, 2, 2, 2, 1, 1]
    elif blocks == 26:
        layers = [1, 1, 2, 2, 2, 2, 1, 1]
    elif blocks == 38:
        assert simplified
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 42:
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 54:
        assert simplified
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 58:
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 105:
        assert simplified
        layers = [1, 1, 3, 4, 23, 3, 1, 1]
    else:
        raise ValueError("Unsupported DRN with number of blocks: {}".format(blocks))

    if blocks < 50:
        channels_per_layers = [16, 32, 64, 128, 256, 512, 512, 512]
        bottlenecks_per_layers = [0, 0, 0, 0, 0, 0, 0, 0]
    else:
        channels_per_layers = [16, 32, 256, 512, 1024, 2048, 512, 512]
        bottlenecks_per_layers = [0, 0, 1, 1, 1, 1, 0, 0]

    if simplified:
        simplifieds_per_layers = [1, 1, 0, 0, 0, 0, 1, 1]
        residuals_per_layers = [0, 0, 1, 1, 1, 1, 0, 0]
    else:
        simplifieds_per_layers = [0, 0, 0, 0, 0, 0, 0, 0]
        residuals_per_layers = [1, 1, 1, 1, 1, 1, 0, 0]

    dilations_per_layers = [1, 1, 1, 1, 2, 4, 2, 1]
    downsample = [0, 1, 1, 1, 0, 0, 0, 0]

    def expand(property_per_layers):
        from functools import reduce
        return reduce(
            lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(property_per_layers, layers, downsample),
            [[]])

    channels = expand(channels_per_layers)
    dilations = expand(dilations_per_layers)
    bottlenecks = expand(bottlenecks_per_layers)
    residuals = expand(residuals_per_layers)
    simplifieds = expand(simplifieds_per_layers)

    init_block_channels = channels_per_layers[0]

    net = DRN(
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
        bottlenecks=bottlenecks,
        simplifieds=simplifieds,
        residuals=residuals,
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


def drnc26(**kwargs):
    """
    DRN-C-26 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=26, model_name="drnc26", **kwargs)


def drnc42(**kwargs):
    """
    DRN-C-42 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=42, model_name="drnc42", **kwargs)


def drnc58(**kwargs):
    """
    DRN-C-58 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=58, model_name="drnc58", **kwargs)


def drnd22(**kwargs):
    """
    DRN-D-58 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=22, simplified=True, model_name="drnd22", **kwargs)


def drnd38(**kwargs):
    """
    DRN-D-38 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=38, simplified=True, model_name="drnd38", **kwargs)


def drnd54(**kwargs):
    """
    DRN-D-54 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=54, simplified=True, model_name="drnd54", **kwargs)


def drnd105(**kwargs):
    """
    DRN-D-105 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=105, simplified=True, model_name="drnd105", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        drnc26,
        drnc42,
        drnc58,
        drnd22,
        drnd38,
        drnd54,
        drnd105,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != drnc26 or weight_count == 21126584)
        assert (model != drnc42 or weight_count == 31234744)
        assert (model != drnc58 or weight_count == 40542008)  # 41591608
        assert (model != drnd22 or weight_count == 16393752)
        assert (model != drnd38 or weight_count == 26501912)
        assert (model != drnd54 or weight_count == 35809176)
        assert (model != drnd105 or weight_count == 54801304)


if __name__ == "__main__":
    _test()
