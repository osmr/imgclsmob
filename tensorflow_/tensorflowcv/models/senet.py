"""
    SENet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['SENet', 'senet16', 'senet28', 'senet40', 'senet52', 'senet103', 'senet154']

import os
import math
import tensorflow as tf
from .common import conv1x1_block, conv3x3_block, maxpool2d, se_block, is_channels_first, flatten


def senet_bottleneck(x,
                     in_channels,
                     out_channels,
                     strides,
                     cardinality,
                     bottleneck_width,
                     training,
                     data_format,
                     name="senet_bottleneck"):
    """
    SENet bottleneck block for residual path in SENet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'senet_bottleneck'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 4
    D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
    group_width = cardinality * D
    group_width2 = group_width // 2

    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=group_width2,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=group_width2,
        out_channels=group_width,
        strides=strides,
        groups=cardinality,
        training=training,
        data_format=data_format,
        name=name + "/conv2")
    x = conv1x1_block(
        x=x,
        in_channels=group_width,
        out_channels=out_channels,
        activation=None,
        training=training,
        data_format=data_format,
        name=name + "/conv3")
    return x


def senet_unit(x,
               in_channels,
               out_channels,
               strides,
               cardinality,
               bottleneck_width,
               identity_conv3x3,
               training,
               data_format,
               name="senet_unit"):
    """
    SENet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    identity_conv3x3 : bool, default False
        Whether to use 3x3 convolution in the identity link.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'senet_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        if identity_conv3x3:
            identity = conv3x3_block(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activation=None,
                training=training,
                data_format=data_format,
                name=name + "/identity_conv")
        else:
            identity = conv1x1_block(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                activation=None,
                training=training,
                data_format=data_format,
                name=name + "/identity_conv")
    else:
        identity = x

    x = senet_bottleneck(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        training=training,
        data_format=data_format,
        name=name + "/body")

    x = se_block(
        x=x,
        channels=out_channels,
        data_format=data_format,
        name=name + "/se")

    x = x + identity

    x = tf.nn.relu(x, name=name + "/activ")
    return x


def senet_init_block(x,
                     in_channels,
                     out_channels,
                     training,
                     data_format,
                     name="senet_init_block"):
    """
    SENet specific initial block.

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
    name : str, default 'senet_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 2

    x = conv3x3_block(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        strides=2,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = conv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        training=training,
        data_format=data_format,
        name=name + "/conv2")
    x = conv3x3_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        training=training,
        data_format=data_format,
        name=name + "/conv3")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=1,
        data_format=data_format,
        name=name + "/pool")
    return x


class SENet(object):
    """
    SENet model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
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
                 cardinality,
                 bottleneck_width,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(SENet, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
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
        x = senet_init_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            training=training,
            data_format=self.data_format,
            name="features/init_block")
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            identity_conv3x3 = (i != 0)
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                x = senet_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    identity_conv3x3=identity_conv3x3,
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
        x = tf.keras.layers.Dropout(
            rate=0.2,
            name="output/dropout")(
            inputs=x,
            training=training)
        x = tf.keras.layers.Dense(
            units=self.classes,
            name="output/fc")(x)

        return x


def get_senet(blocks,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create SENet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
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

    if blocks == 16:
        layers = [1, 1, 1, 1]
        cardinality = 32
    elif blocks == 28:
        layers = [2, 2, 2, 2]
        cardinality = 32
    elif blocks == 40:
        layers = [3, 3, 3, 3]
        cardinality = 32
    elif blocks == 52:
        layers = [3, 4, 6, 3]
        cardinality = 32
    elif blocks == 103:
        layers = [3, 4, 23, 3]
        cardinality = 32
    elif blocks == 154:
        layers = [3, 8, 36, 3]
        cardinality = 64
    else:
        raise ValueError("Unsupported SENet with number of blocks: {}".format(blocks))

    bottleneck_width = 4
    init_block_channels = 128
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = SENet(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
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


def senet16(**kwargs):
    """
    SENet-16 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_senet(blocks=16, model_name="senet16", **kwargs)


def senet28(**kwargs):
    """
    SENet-28 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_senet(blocks=28, model_name="senet28", **kwargs)


def senet40(**kwargs):
    """
    SENet-40 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_senet(blocks=40, model_name="senet40", **kwargs)


def senet52(**kwargs):
    """
    SENet-52 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_senet(blocks=52, model_name="senet52", **kwargs)


def senet103(**kwargs):
    """
    SENet-103 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_senet(blocks=103, model_name="senet103", **kwargs)


def senet154(**kwargs):
    """
    SENet-154 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_senet(blocks=154, model_name="senet154", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        senet16,
        senet28,
        senet40,
        senet52,
        senet103,
        senet154,
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
        assert (model != senet16 or weight_count == 31366168)
        assert (model != senet28 or weight_count == 36453768)
        assert (model != senet40 or weight_count == 41541368)
        assert (model != senet52 or weight_count == 44659416)
        assert (model != senet103 or weight_count == 60963096)
        assert (model != senet154 or weight_count == 115088984)

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
