"""
    ShuffleNet V2 for ImageNet-1K, implemented in TensorFlow. The alternative variant.
    Original paper: 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.
"""

__all__ = ['ShuffleNetV2b', 'shufflenetv2b_wd2', 'shufflenetv2b_w1', 'shufflenetv2b_w3d2', 'shufflenetv2b_w2']

import os
import tensorflow as tf
from .common import conv1x1_block, conv3x3_block, dwconv3x3_block, channel_shuffle, channel_shuffle2, maxpool2d,\
    se_block, is_channels_first, get_channel_axis, flatten


def shuffle_unit(x,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual,
                 shuffle_group_first,
                 training,
                 data_format,
                 name="shuffle_unit"):
    """
    ShuffleNetV2(b) unit.

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
    shuffle_group_first : bool
        Whether to use channel shuffle in group first mode.
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
    in_channels2 = in_channels // 2
    assert (in_channels % 2 == 0)

    if downsample:
        y1 = dwconv3x3_block(
            x=x,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            activation=None,
            training=training,
            data_format=data_format,
            name=name + "/shortcut_dconv")
        y1 = conv1x1_block(
            x=y1,
            in_channels=in_channels,
            out_channels=in_channels,
            training=training,
            data_format=data_format,
            name=name + "/shortcut_conv")
        x2 = x
    else:
        y1, x2 = tf.split(x, num_or_size_splits=2, axis=get_channel_axis(data_format))

    y2_in_channels = (in_channels if downsample else in_channels2)
    y2_out_channels = out_channels - y2_in_channels

    y2 = conv1x1_block(
        x=x2,
        in_channels=y2_in_channels,
        out_channels=mid_channels,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    y2 = dwconv3x3_block(
        x=y2,
        in_channels=mid_channels,
        out_channels=mid_channels,
        strides=(2 if downsample else 1),
        activation=None,
        training=training,
        data_format=data_format,
        name=name + "/dconv")
    y2 = conv1x1_block(
        x=y2,
        in_channels=mid_channels,
        out_channels=y2_out_channels,
        training=training,
        data_format=data_format,
        name=name + "/conv2")

    if use_se:
        y2 = se_block(
            x=y2,
            channels=y2_out_channels,
            data_format=data_format,
            name=name + "/se")

    if use_residual and not downsample:
        assert (y2_out_channels == in_channels2)
        y2 = y2 + x2

    x = tf.concat([y1, y2], axis=get_channel_axis(data_format), name=name + "/concat")

    assert (out_channels % 2 == 0)
    if shuffle_group_first:
        x = channel_shuffle(
            x=x,
            groups=2,
            data_format=data_format)
    else:
        x = channel_shuffle2(
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
    ShuffleNetV2(b) specific initial block.

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
        padding=1,
        ceil_mode=False,
        data_format=data_format,
        name=name + "/pool")
    return x


class ShuffleNetV2b(object):
    """
    ShuffleNetV2(b) model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
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
    shuffle_group_first : bool, default True
        Whether to use channel shuffle in group first mode.
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
                 shuffle_group_first=True,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ShuffleNetV2b, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.final_block_channels = final_block_channels
        self.use_se = use_se
        self.use_residual = use_residual
        self.shuffle_group_first = shuffle_group_first
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
                    shuffle_group_first=self.shuffle_group_first,
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


def get_shufflenetv2b(width_scale,
                      shuffle_group_first=True,
                      model_name=None,
                      pretrained=False,
                      root=os.path.join("~", ".tensorflow", "models"),
                      **kwargs):
    """
    Create ShuffleNetV2(b) model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    shuffle_group_first : bool, default True
        Whether to use channel shuffle in group first mode.
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

    net = ShuffleNetV2b(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        shuffle_group_first=shuffle_group_first,
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


def shufflenetv2b_wd2(**kwargs):
    """
    ShuffleNetV2(b) 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
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
    return get_shufflenetv2b(
        width_scale=(12.0 / 29.0),
        shuffle_group_first=True,
        model_name="shufflenetv2b_wd2",
        **kwargs)


def shufflenetv2b_w1(**kwargs):
    """
    ShuffleNetV2(b) 1x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
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
    return get_shufflenetv2b(
        width_scale=1.0,
        shuffle_group_first=True,
        model_name="shufflenetv2b_w1",
        **kwargs)


def shufflenetv2b_w3d2(**kwargs):
    """
    ShuffleNetV2(b) 1.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
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
    return get_shufflenetv2b(
        width_scale=(44.0 / 29.0),
        shuffle_group_first=True,
        model_name="shufflenetv2b_w3d2",
        **kwargs)


def shufflenetv2b_w2(**kwargs):
    """
    ShuffleNetV2(b) 2x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
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
    return get_shufflenetv2b(
        width_scale=(61.0 / 29.0),
        shuffle_group_first=True,
        model_name="shufflenetv2b_w2",
        **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        shufflenetv2b_wd2,
        shufflenetv2b_w1,
        shufflenetv2b_w3d2,
        shufflenetv2b_w2,
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
        assert (model != shufflenetv2b_wd2 or weight_count == 1366792)
        assert (model != shufflenetv2b_w1 or weight_count == 2279760)
        assert (model != shufflenetv2b_w3d2 or weight_count == 4410194)
        assert (model != shufflenetv2b_w2 or weight_count == 7611290)

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
