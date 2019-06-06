"""
    MnasNet, implemented in TensorFlow.
    Original paper: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,' https://arxiv.org/abs/1807.11626.
"""

__all__ = ['MnasNet', 'mnasnet']

import os
import tensorflow as tf
from .common import is_channels_first, flatten, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block


def dws_conv_block(x,
                   in_channels,
                   out_channels,
                   training,
                   data_format,
                   name="dws_conv_block"):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'dws_conv_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = dwconv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=in_channels,
        training=training,
        data_format=data_format,
        name=name + "/dw_conv")
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        training=training,
        data_format=data_format,
        name=name + "/pw_conv")
    return x


def mnas_unit(x,
              in_channels,
              out_channels,
              kernel_size,
              strides,
              expansion_factor,
              training,
              data_format,
              name="mnas_unit"):
    """
    So-called 'Linear Bottleneck' layer. It is used as a MobileNetV2 unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    expansion_factor : int
        Factor for expansion of channels.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'mnas_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    residual = (in_channels == out_channels) and (strides == 1)
    mid_channels = in_channels * expansion_factor
    dwconv_block_fn = dwconv3x3_block if kernel_size == 3 else (dwconv5x5_block if kernel_size == 5 else None)

    if residual:
        identity = x

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = dwconv_block_fn(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        strides=strides,
        training=training,
        data_format=data_format,
        name=name + "/conv2")
    x = conv1x1_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activate=False,
        training=training,
        data_format=data_format,
        name=name + "/conv3")

    if residual:
        x = x + identity

    return x


def mnas_init_block(x,
                    in_channels,
                    out_channels_list,
                    training,
                    data_format,
                    name="mnas_init_block"):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels_list : list of 2 int
        Numbers of output channels.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'mnas_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels_list[0],
        strides=2,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = dws_conv_block(
        x=x,
        in_channels=out_channels_list[0],
        out_channels=out_channels_list[1],
        training=training,
        data_format=data_format,
        name=name + "/conv2")
    return x


class MnasNet(object):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Numbers of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list of list of int
        Number of kernel sizes for each unit.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
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
                 kernel_sizes,
                 expansion_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(MnasNet, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.final_block_channels = final_block_channels
        self.kernel_sizes = kernel_sizes
        self.expansion_factors = expansion_factors
        self.in_channels = in_channels
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

    def __call__(self,
                 x,
                 training=False):
        """
        Build a model graph.

        Parameters:
        ----------
        x : Tensor
            Input tensor.
        training : bool, or a TensorFlow boolean scalar tensor, default False
          Whether to return the output in training mode or in inference mode.

        Returns
        -------
        Tensor
            Resulted tensor.
        """
        in_channels = self.in_channels
        x = mnas_init_block(
            x=x,
            in_channels=in_channels,
            out_channels_list=self.init_block_channels,
            training=training,
            data_format=self.data_format,
            name="features/init_block")
        in_channels = self.init_block_channels[-1]
        for i, channels_per_stage in enumerate(self.channels):
            kernel_sizes_per_stage = self.kernel_sizes[i]
            expansion_factors_per_stage = self.expansion_factors[i]
            for j, out_channels in enumerate(channels_per_stage):
                kernel_size = kernel_sizes_per_stage[j]
                expansion_factor = expansion_factors_per_stage[j]
                strides = 2 if (j == 0) else 1
                x = mnas_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    expansion_factor=expansion_factor,
                    training=training,
                    data_format=self.data_format,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
        x = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.final_block_channels,
            activate=True,
            training=training,
            data_format=self.data_format,
            name="features/final_block")
        # in_channels = self.final_block_channels
        x = tf.layers.average_pooling2d(
            inputs=x,
            pool_size=7,
            strides=1,
            data_format=self.data_format,
            name="features/final_pool")

        # x = tf.layers.flatten(x)
        x = flatten(
            x=x,
            data_format=self.data_format)
        x = tf.layers.dense(
            inputs=x,
            units=self.classes,
            name="output")

        return x


def get_mnasnet(model_name=None,
                pretrained=False,
                root=os.path.join('~', '.keras', 'models'),
                **kwargs):
    """
    Create MnasNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """

    init_block_channels = [32, 16]
    final_block_channels = 1280
    layers = [3, 3, 3, 2, 4, 1]
    downsample = [1, 1, 1, 0, 1, 0]
    channels_per_layers = [24, 40, 80, 96, 192, 320]
    expansion_factors_per_layers = [3, 3, 6, 6, 6, 6]
    kernel_sizes_per_layers = [3, 5, 5, 3, 5, 3]
    default_kernel_size = 3

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] + [default_kernel_size] * (y[1] - 1)] if y[2] != 0 else x[:-1] + [
        x[-1] + [y[0]] + [default_kernel_size] * (y[1] - 1)], zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])

    net = MnasNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        expansion_factors=expansion_factors,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_state_dict
        net.state_dict, net.file_path = download_state_dict(
            model_name=model_name,
            local_model_store_dir_path=root)
    else:
        net.state_dict = None
        net.file_path = None

    return net


def mnasnet(**kwargs):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    functor
        Functor for model graph creation with extra fields.
    """
    return get_mnasnet(model_name="mnasnet", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        mnasnet,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)
        x = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224) if is_channels_first(data_format) else (None, 224, 224, 3),
            name="xx")
        y_net = net(x)

        weight_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != mnasnet or weight_count == 4308816)

        with tf.Session() as sess:
            if pretrained:
                from .model_store import init_variables_from_state_dict
                init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
            else:
                sess.run(tf.global_variables_initializer())
            x_value = np.zeros((1, 3, 224, 224) if is_channels_first(data_format) else (1, 224, 224, 3), np.float32)
            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
