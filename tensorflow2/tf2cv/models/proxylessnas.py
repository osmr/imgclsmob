"""
    ProxylessNAS for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.
"""

__all__ = ['ProxylessNAS', 'proxylessnas_cpu', 'proxylessnas_gpu', 'proxylessnas_mobile', 'proxylessnas_mobile14',
           'ProxylessUnit', 'get_proxylessnas']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import ConvBlock, conv1x1_block, conv3x3_block, SimpleSequential, flatten, is_channels_first


class ProxylessBlock(nn.Layer):
    """
    ProxylessNAS block for residual path in ProxylessNAS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    strides : int
        Strides of the convolution.
    bn_eps : float
        Small float added to variance in Batch norm.
    expansion : int
        Expansion ratio.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 bn_eps,
                 expansion,
                 data_format="channels_last",
                 **kwargs):
        super(ProxylessBlock, self).__init__(**kwargs)
        self.use_bc = (expansion > 1)
        mid_channels = in_channels * expansion

        if self.use_bc:
            self.bc_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_eps=bn_eps,
                activation="relu6",
                data_format=data_format,
                name="bc_conv")

        padding = (kernel_size - 1) // 2
        self.dw_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=mid_channels,
            bn_eps=bn_eps,
            activation="relu6",
            data_format=data_format,
            name="dw_conv")
        self.pw_conv = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None,
            data_format=data_format,
            name="pw_conv")

    def call(self, x, training=None):
        if self.use_bc:
            x = self.bc_conv(x, training=training)
        x = self.dw_conv(x, training=training)
        x = self.pw_conv(x, training=training)
        return x


class ProxylessUnit(nn.Layer):
    """
    ProxylessNAS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size for body block.
    strides : int
        Strides of the convolution.
    bn_eps : float
        Small float added to variance in Batch norm.
    expansion : int
        Expansion ratio for body block.
    residual : bool
        Whether to use residual branch.
    shortcut : bool
        Whether to use identity branch.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 bn_eps,
                 expansion,
                 residual,
                 shortcut,
                 data_format="channels_last",
                 **kwargs):
        super(ProxylessUnit, self).__init__(**kwargs)
        assert (residual or shortcut)
        self.residual = residual
        self.shortcut = shortcut

        if self.residual:
            self.body = ProxylessBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                bn_eps=bn_eps,
                expansion=expansion,
                data_format=data_format,
                name="body")

    def call(self, x, training=None):
        if not self.residual:
            return x
        if not self.shortcut:
            return self.body(x, training=training)
        identity = x
        x = self.body(x, training=training)
        x = identity + x
        return x


class ProxylessNAS(tf.keras.Model):
    """
    ProxylessNAS model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    residuals : list of list of int
        Whether to use residual branch in units.
    shortcuts : list of list of int
        Whether to use identity branch in units.
    kernel_sizes : list of list of int
        Convolution window size for each units.
    expansions : list of list of int
        Expansion ratio for each units.
    bn_eps : float, default 1e-3
        Small float added to variance in Batch norm.
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
                 final_block_channels,
                 residuals,
                 shortcuts,
                 kernel_sizes,
                 expansions,
                 bn_eps=1e-3,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ProxylessNAS, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            strides=2,
            bn_eps=bn_eps,
            activation="relu6",
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            residuals_per_stage = residuals[i]
            shortcuts_per_stage = shortcuts[i]
            kernel_sizes_per_stage = kernel_sizes[i]
            expansions_per_stage = expansions[i]
            for j, out_channels in enumerate(channels_per_stage):
                residual = (residuals_per_stage[j] == 1)
                shortcut = (shortcuts_per_stage[j] == 1)
                kernel_size = kernel_sizes_per_stage[j]
                expansion = expansions_per_stage[j]
                strides = 2 if (j == 0) and (i != 0) else 1
                stage.add(ProxylessUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    bn_eps=bn_eps,
                    expansion=expansion,
                    residual=residual,
                    shortcut=shortcut,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            bn_eps=bn_eps,
            activation="relu6",
            data_format=data_format,
            name="final_block"))
        in_channels = final_block_channels
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_proxylessnas(version,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".tensorflow", "models"),
                     **kwargs):
    """
    Create ProxylessNAS model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of ProxylessNAS ('cpu', 'gpu', 'mobile' or 'mobile14').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if version == "cpu":
        residuals = [[1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [32, 32, 32, 32], [48, 48, 48, 48], [88, 88, 88, 88, 104, 104, 104, 104],
                    [216, 216, 216, 216, 360]]
        kernel_sizes = [[3], [3, 3, 3, 3], [3, 3, 3, 5], [3, 3, 3, 3, 5, 3, 3, 3], [5, 5, 5, 3, 5]]
        expansions = [[1], [6, 3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 3, 3, 3, 6]]
        init_block_channels = 40
        final_block_channels = 1432
    elif version == "gpu":
        residuals = [[1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [32, 32, 32, 32], [56, 56, 56, 56], [112, 112, 112, 112, 128, 128, 128, 128],
                    [256, 256, 256, 256, 432]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 3, 3], [7, 5, 5, 5, 5, 3, 3, 5], [7, 7, 7, 5, 7]]
        expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 6, 6, 6]]
        init_block_channels = 40
        final_block_channels = 1728
    elif version == "mobile":
        residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[16], [32, 32, 32, 32], [40, 40, 40, 40], [80, 80, 80, 80, 96, 96, 96, 96],
                    [192, 192, 192, 192, 320]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
        expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
        init_block_channels = 32
        final_block_channels = 1280
    elif version == "mobile14":
        residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [40, 40, 40, 40], [56, 56, 56, 56], [112, 112, 112, 112, 136, 136, 136, 136],
                    [256, 256, 256, 256, 448]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
        expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
        init_block_channels = 48
        final_block_channels = 1792
    else:
        raise ValueError("Unsupported ProxylessNAS version: {}".format(version))

    shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

    net = ProxylessNAS(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        residuals=residuals,
        shortcuts=shortcuts,
        kernel_sizes=kernel_sizes,
        expansions=expansions,
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


def proxylessnas_cpu(**kwargs):
    """
    ProxylessNAS (CPU) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="cpu", model_name="proxylessnas_cpu", **kwargs)


def proxylessnas_gpu(**kwargs):
    """
    ProxylessNAS (GPU) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="gpu", model_name="proxylessnas_gpu", **kwargs)


def proxylessnas_mobile(**kwargs):
    """
    ProxylessNAS (Mobile) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="mobile", model_name="proxylessnas_mobile", **kwargs)


def proxylessnas_mobile14(**kwargs):
    """
    ProxylessNAS (Mobile-14) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="mobile14", model_name="proxylessnas_mobile14", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        proxylessnas_cpu,
        proxylessnas_gpu,
        proxylessnas_mobile,
        proxylessnas_mobile14,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != proxylessnas_cpu or weight_count == 4361648)
        assert (model != proxylessnas_gpu or weight_count == 7119848)
        assert (model != proxylessnas_mobile or weight_count == 4080512)
        assert (model != proxylessnas_mobile14 or weight_count == 6857568)


if __name__ == "__main__":
    _test()
