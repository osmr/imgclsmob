"""
    MENet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications,'
    https://arxiv.org/abs/1803.09127.
"""

__all__ = ['MENet', 'menet108_8x1_g3', 'menet128_8x1_g4', 'menet160_8x1_g8', 'menet228_12x1_g3', 'menet256_12x1_g4',
           'menet348_12x1_g3', 'menet352_12x1_g8', 'menet456_24x1_g3']

import os
import tensorflow as tf
from .common import conv2d, conv1x1, conv3x3, depthwise_conv3x3, batchnorm, channel_shuffle, maxpool2d, avgpool2d,\
    is_channels_first, get_channel_axis, flatten


def me_unit(x,
            in_channels,
            out_channels,
            side_channels,
            groups,
            downsample,
            ignore_group,
            training,
            data_format,
            name="me_unit"):
    """
    MENet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    side_channels : int
        Number of side channels.
    groups : int
        Number of groups in convolution layers.
    downsample : bool
        Whether do downsample.
    ignore_group : bool
        Whether ignore group value in the first convolution layer.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'me_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 4

    if downsample:
        out_channels -= in_channels

    identity = x

    # pointwise group convolution 1
    x = conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        groups=(1 if ignore_group else groups),
        data_format=data_format,
        name=name + "/compress_conv1")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/compress_bn1")
    x = tf.nn.relu(x, name=name + "/compress_activ")

    assert (mid_channels % groups == 0)
    x = channel_shuffle(
        x=x,
        groups=groups,
        data_format=data_format)

    # merging
    y = conv1x1(
        x=x,
        in_channels=mid_channels,
        out_channels=side_channels,
        data_format=data_format,
        name=name + "/s_merge_conv/conv")
    y = batchnorm(
        x=y,
        training=training,
        data_format=data_format,
        name=name + "/s_merge_bn")
    y = tf.nn.relu(y, name=name + "/s_merge_activ")

    # depthwise convolution (bottleneck)
    x = depthwise_conv3x3(
        x=x,
        channels=mid_channels,
        strides=(2 if downsample else 1),
        data_format=data_format,
        name=name + "/dw_conv2")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/dw_bn2")

    # evolution
    y = conv3x3(
        x=y,
        in_channels=side_channels,
        out_channels=side_channels,
        strides=(2 if downsample else 1),
        data_format=data_format,
        name=name + "/s_conv")
    y = batchnorm(
        x=y,
        training=training,
        data_format=data_format,
        name=name + "/s_conv_bn")
    y = tf.nn.relu(y, name=name + "/s_conv_activ")

    y = conv1x1(
        x=y,
        in_channels=side_channels,
        out_channels=mid_channels,
        data_format=data_format,
        name=name + "/s_evolve_conv/conv")
    y = batchnorm(
        x=y,
        training=training,
        data_format=data_format,
        name=name + "/s_evolve_bn")
    y = tf.nn.sigmoid(y, name=name + "/s_evolve_activ")

    x = x * y

    # pointwise group convolution 2
    x = conv1x1(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        groups=groups,
        data_format=data_format,
        name=name + "/expand_conv3")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/expand_bn3")

    if downsample:
        identity = avgpool2d(
            x=identity,
            pool_size=3,
            strides=2,
            padding=1,
            data_format=data_format,
            name=name + "/avgpool")
        x = tf.concat([x, identity], axis=get_channel_axis(data_format), name=name + "/concat")
    else:
        x = x + identity

    x = tf.nn.relu(x, name=name + "/final_activ")
    return x


def me_init_block(x,
                  in_channels,
                  out_channels,
                  training,
                  data_format,
                  name="me_init_block"):
    """
    MENet specific initial block.

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
    name : str, default 'me_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=2,
        padding=1,
        use_bias=False,
        data_format=data_format,
        name=name + "/conv")
    x = batchnorm(
        x=x,
        training=training,
        data_format=data_format,
        name=name + "/bn")
    x = tf.nn.relu(x, name=name + "/activ")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=1,
        data_format=data_format,
        name=name + "/pool")
    return x


class MENet(object):
    """
    MENet model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications,'
    https://arxiv.org/abs/1803.09127.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    side_channels : int
        Number of side channels in a ME-unit.
    groups : int
        Number of groups in convolution layers.
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
                 side_channels,
                 groups,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(MENet, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.side_channels = side_channels
        self.groups = groups
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
        x = me_init_block(
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
                ignore_group = (i == 0) and (j == 0)
                x = me_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    side_channels=self.side_channels,
                    groups=self.groups,
                    downsample=downsample,
                    ignore_group=ignore_group,
                    training=training,
                    data_format=self.data_format,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
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


def get_menet(first_stage_channels,
              side_channels,
              groups,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create MENet model with specific parameters.

    Parameters:
    ----------
    first_stage_channels : int
        Number of output channels at the first stage.
    side_channels : int
        Number of side channels in a ME-unit.
    groups : int
        Number of groups in convolution layers.
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

    layers = [4, 8, 4]

    if first_stage_channels == 108:
        init_block_channels = 12
        channels_per_layers = [108, 216, 432]
    elif first_stage_channels == 128:
        init_block_channels = 12
        channels_per_layers = [128, 256, 512]
    elif first_stage_channels == 160:
        init_block_channels = 16
        channels_per_layers = [160, 320, 640]
    elif first_stage_channels == 228:
        init_block_channels = 24
        channels_per_layers = [228, 456, 912]
    elif first_stage_channels == 256:
        init_block_channels = 24
        channels_per_layers = [256, 512, 1024]
    elif first_stage_channels == 348:
        init_block_channels = 24
        channels_per_layers = [348, 696, 1392]
    elif first_stage_channels == 352:
        init_block_channels = 24
        channels_per_layers = [352, 704, 1408]
    elif first_stage_channels == 456:
        init_block_channels = 48
        channels_per_layers = [456, 912, 1824]
    else:
        raise ValueError("The {} of `first_stage_channels` is not supported".format(first_stage_channels))

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = MENet(
        channels=channels,
        init_block_channels=init_block_channels,
        side_channels=side_channels,
        groups=groups,
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


def menet108_8x1_g3(**kwargs):
    """
    108-MENet-8x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=108, side_channels=8, groups=3, model_name="menet108_8x1_g3", **kwargs)


def menet128_8x1_g4(**kwargs):
    """
    128-MENet-8x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=128, side_channels=8, groups=4, model_name="menet128_8x1_g4", **kwargs)


def menet160_8x1_g8(**kwargs):
    """
    160-MENet-8x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=160, side_channels=8, groups=8, model_name="menet160_8x1_g8", **kwargs)


def menet228_12x1_g3(**kwargs):
    """
    228-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=228, side_channels=12, groups=3, model_name="menet228_12x1_g3", **kwargs)


def menet256_12x1_g4(**kwargs):
    """
    256-MENet-12x1 (g=4) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=256, side_channels=12, groups=4, model_name="menet256_12x1_g4", **kwargs)


def menet348_12x1_g3(**kwargs):
    """
    348-MENet-12x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=348, side_channels=12, groups=3, model_name="menet348_12x1_g3", **kwargs)


def menet352_12x1_g8(**kwargs):
    """
    352-MENet-12x1 (g=8) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=352, side_channels=12, groups=8, model_name="menet352_12x1_g8", **kwargs)


def menet456_24x1_g3(**kwargs):
    """
    456-MENet-24x1 (g=3) model from 'Merging and Evolution: Improving Convolutional Neural Networks for Mobile
    Applications,' https://arxiv.org/abs/1803.09127.

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
    return get_menet(first_stage_channels=456, side_channels=24, groups=3, model_name="menet456_24x1_g3", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        menet108_8x1_g3,
        menet128_8x1_g4,
        menet160_8x1_g8,
        menet228_12x1_g3,
        menet256_12x1_g4,
        menet348_12x1_g3,
        menet352_12x1_g8,
        menet456_24x1_g3,
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
        assert (model != menet108_8x1_g3 or weight_count == 654516)
        assert (model != menet128_8x1_g4 or weight_count == 750796)
        assert (model != menet160_8x1_g8 or weight_count == 850120)
        assert (model != menet228_12x1_g3 or weight_count == 1806568)
        assert (model != menet256_12x1_g4 or weight_count == 1888240)
        assert (model != menet348_12x1_g3 or weight_count == 3368128)
        assert (model != menet352_12x1_g8 or weight_count == 2272872)
        assert (model != menet456_24x1_g3 or weight_count == 5304784)

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
