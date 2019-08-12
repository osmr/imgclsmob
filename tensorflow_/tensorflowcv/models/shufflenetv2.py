"""
    ShuffleNet V2 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.
"""

__all__ = ['ShuffleNetV2', 'shufflenetv2_wd2', 'shufflenetv2_w1', 'shufflenetv2_w3d2', 'shufflenetv2_w2']

import os
import tensorflow as tf
from .common import conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, batchnorm, channel_shuffle, maxpool2d,\
    se_block, is_channels_first, get_channel_axis, flatten


def shuffle_unit(x,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 training,
                 data_format,
                 name="shuffle_unit"):
    """
    ShuffleNetV2 unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether do downsample.
    use_se : bool
        Whether to use SE block.
    use_residual : bool
        Whether to use residual connection.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'shuffle_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 2

    if downsample:
        y1 = depthwise_conv3x3(
            x=x,
            channels=in_channels,
            strides=2,
            data_format=data_format,
            name=name + "/dw_conv4")
        y1 = batchnorm(
            x=y1,
            training=training,
            data_format=data_format,
            name=name + "/dw_bn4")
        y1 = conv1x1(
            x=y1,
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name=name + "/expand_conv5/conv")
        y1 = batchnorm(
            x=y1,
            training=training,
            data_format=data_format,
            name=name + "/expand_bn5")
        y1 = tf.nn.relu(y1, name=name + "/expand_activ5")
        x2 = x
    else:
        y1, x2 = tf.split(x, num_or_size_splits=2, axis=get_channel_axis(data_format))

    y2 = conv1x1(
        x=x2,
        in_channels=(in_channels if downsample else mid_channels),
        out_channels=mid_channels,
        data_format=data_format,
        name=name + "/compress_conv1/conv")
    y2 = batchnorm(
        x=y2,
        training=training,
        data_format=data_format,
        name=name + "/compress_bn1")
    y2 = tf.nn.relu(y2, name=name + "/compress_activ1")

    y2 = depthwise_conv3x3(
        x=y2,
        channels=mid_channels,
        strides=(2 if downsample else 1),
        data_format=data_format,
        name=name + "/dw_conv2")
    y2 = batchnorm(
        x=y2,
        training=training,
        data_format=data_format,
        name=name + "/dw_bn2")

    y2 = conv1x1(
        x=y2,
        in_channels=mid_channels,
        out_channels=mid_channels,
        data_format=data_format,
        name=name + "/expand_conv3/conv")
    y2 = batchnorm(
        x=y2,
        training=training,
        data_format=data_format,
        name=name + "/expand_bn3")
    y2 = tf.nn.relu(y2, name=name + "/expand_activ3")

    if use_se:
        y2 = se_block(
            x=y2,
            channels=mid_channels,
            data_format=data_format,
            name=name + "/se")

    if use_residual and not downsample:
        y2 = y2 + x2

    x = tf.concat([y1, y2], axis=get_channel_axis(data_format), name=name + "/concat")

    assert (mid_channels % 2 == 0)
    x = channel_shuffle(
        x=x,
        groups=2,
        data_format=data_format)

    return x


def shuffle_init_block(x,
                       in_channels,
                       out_channels,
                       training,
                       data_format,
                       name="shuffle_init_block"):
    """
    ShuffleNetV2 specific initial block.

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
    name : str, default 'shuffle_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        training=training,
        data_format=data_format,
        name=name + "/conv")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=0,
        ceil_mode=True,
        data_format=data_format,
        name=name + "/pool")
    return x


class ShuffleNetV2(object):
    """
    ShuffleNetV2 model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    use_se : bool, default False
        Whether to use SE block.
    use_residual : bool, default False
        Whether to use residual connections.
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
                 use_se=False,
                 use_residual=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleNetV2, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.final_block_channels = final_block_channels
        self.use_se = use_se
        self.use_residual = use_residual
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
        x = shuffle_init_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            training=training,
            data_format=self.data_format,
            name="features/init_block")
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                x = shuffle_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    downsample=downsample,
                    use_se=self.use_se,
                    use_residual=self.use_residual,
                    training=training,
                    data_format=self.data_format,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
        x = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.final_block_channels,
            training=training,
            data_format=self.data_format,
            name="features/final_block")
        x = tf.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=self.data_format,
            name="features/final_pool")(x)

        # x = tf.layers.flatten(x)
        x = flatten(
            x=x,
            data_format=self.data_format)
        x = tf.keras.layers.Dense(
            units=self.classes,
            name="output")(x)

        return x


def get_shufflenetv2(width_scale,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".tensorflow", "models"),
                     **kwargs):
    """
    Create ShuffleNetV2 model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
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

    init_block_channels = 24
    final_block_channels = 1024
    layers = [4, 8, 4]
    channels_per_layers = [116, 232, 464]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        if width_scale > 1.5:
            final_block_channels = int(final_block_channels * width_scale)

    net = ShuffleNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
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


def shufflenetv2_wd2(**kwargs):
    """
    ShuffleNetV2 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

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
    return get_shufflenetv2(width_scale=(12.0 / 29.0), model_name="shufflenetv2_wd2", **kwargs)


def shufflenetv2_w1(**kwargs):
    """
    ShuffleNetV2 1x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

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
    return get_shufflenetv2(width_scale=1.0, model_name="shufflenetv2_w1", **kwargs)


def shufflenetv2_w3d2(**kwargs):
    """
    ShuffleNetV2 1.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

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
    return get_shufflenetv2(width_scale=(44.0 / 29.0), model_name="shufflenetv2_w3d2", **kwargs)


def shufflenetv2_w2(**kwargs):
    """
    ShuffleNetV2 2x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

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
    return get_shufflenetv2(width_scale=(61.0 / 29.0), model_name="shufflenetv2_w2", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        shufflenetv2_wd2,
        shufflenetv2_w1,
        shufflenetv2_w3d2,
        shufflenetv2_w2,
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
        assert (model != shufflenetv2_wd2 or weight_count == 1366792)
        assert (model != shufflenetv2_w1 or weight_count == 2278604)
        assert (model != shufflenetv2_w3d2 or weight_count == 4406098)
        assert (model != shufflenetv2_w2 or weight_count == 7601686)

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
