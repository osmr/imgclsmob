"""
    DenseNet, implemented in TensorFlow.
    Original paper: 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.
"""

__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201']

import os
import tensorflow as tf
from .common import conv2d, batchnorm, maxpool2d


def dense_conv(x,
               in_channels,
               out_channels,
               kernel_size,
               strides,
               padding,
               training,
               name="dense_conv"):
    """
    DenseNet specific convolution block.

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
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'dense_conv'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = batchnorm(
        x=x,
        training=training,
        name=name + "/bn")
    x = tf.nn.relu(x, name=name + "/activ")
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name + "/conv")
    return x


def dense_conv1x1(x,
                  in_channels,
                  out_channels,
                  training,
                  name="dense_conv1x1"):
    """
    1x1 version of the DenseNet specific convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'dense_conv1x1'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return dense_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=1,
        padding=0,
        training=training,
        name=name)


def dense_conv3x3(x,
                  in_channels,
                  out_channels,
                  training,
                  name="dense_conv3x3"):
    """
    3x3 version of the DenseNet specific convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'dense_conv3x3'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return dense_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=1,
        padding=1,
        training=training,
        name=name)


def dense_unit(x,
               in_channels,
               out_channels,
               dropout_rate,
               training,
               name="dense_unit"):
    """
    DenseNet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : bool
        Parameter of Dropout layer. Faction of the input units to drop.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'dense_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    bn_size = 4
    inc_channels = out_channels - in_channels
    mid_channels = inc_channels * bn_size

    identity = x

    x = dense_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        training=training,
        name=name + "/conv1")
    x = dense_conv3x3(
        x=x,
        in_channels=mid_channels,
        out_channels=inc_channels,
        training=training,
        name=name + "/conv2")

    use_dropout = (dropout_rate != 0.0)
    if use_dropout:
        x = tf.layers.dropout(
            inputs=x,
            rate=dropout_rate,
            training=training,
            name=name + "dropout")

    x = tf.concat([identity, x], axis=1, name=name + "/concat")
    return x


def transition_block(x,
                     in_channels,
                     out_channels,
                     training,
                     name="transition_block"):
    """
    DenseNet's auxiliary block, which can be treated as the initial part of the DenseNet unit, triggered only in the
    first unit of each stage.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'transition_block'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = dense_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        training=training,
        name=name + "/conv")
    x = tf.layers.average_pooling2d(
        inputs=x,
        pool_size=2,
        strides=2,
        data_format='channels_first',
        name=name + "/pool")
    return x


def dense_init_block(x,
                     in_channels,
                     out_channels,
                     training,
                     name="dense_init_block"):
    """
    DenseNet specific initial block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'dense_init_block'
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
        kernel_size=7,
        strides=2,
        padding=3,
        use_bias=False,
        name=name + "/conv")
    x = batchnorm(
        x=x,
        training=training,
        name=name + "/bn")
    x = tf.nn.relu(x, name=name + "/activ")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=1,
        name=name + "/pool")
    return x


def post_activation(x,
                    training,
                    name="post_activation"):
    """
    DenseNet final block, which performs the same function of postactivation as in PreResNet.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'post_activation'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = batchnorm(
        x=x,
        training=training,
        name=name + "/bn")
    x = tf.nn.relu(x, name=name + "/activ")
    return x


class DenseNet(object):
    """
    DenseNet model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels
        self.in_size = in_size
        self.classes = classes

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
        x = dense_init_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            training=training,
            name="features/init_block")
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            if i != 0:
                x = transition_block(
                    x=x,
                    in_channels=in_channels,
                    out_channels=(in_channels // 2),
                    training=training,
                    name="features/stage{}/trans{}".format(i + 1, i + 1))
                in_channels = in_channels // 2
            for j, out_channels in enumerate(channels_per_stage):
                x = dense_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=self.dropout_rate,
                    training=training,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
        x = post_activation(
            x=x,
            training=training,
            name="features/post_activ")
        x = tf.layers.average_pooling2d(
            inputs=x,
            pool_size=7,
            strides=1,
            data_format='channels_first',
            name="features/final_pool")

        x = tf.layers.flatten(x)
        x = tf.layers.dense(
            inputs=x,
            units=self.classes,
            name="output")

        return x


def get_densenet(num_layers,
                 model_name=None,
                 pretrained=False,
                 root=os.path.join('~', '.tensorflow', 'models'),
                 **kwargs):
    """
    Create DenseNet model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    net_lambda : function
        Function for model graph creation.
    net_file_path : str or None
        File path for pretrained model or None.
    """

    if num_layers == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif num_layers == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif num_layers == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif num_layers == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported DenseNet version with number of layers {}".format(num_layers))

    from functools import reduce
    channels = reduce(lambda xi, yi:
                      xi + [reduce(lambda xj, yj:
                                   xj + [xj[-1] + yj],
                                   [growth_rate] * yi,
                                   [xi[-1][-1] // 2])[1:]],
                      layers,
                      [[init_block_channels * 2]])[1:]

    net = DenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def densenet121(**kwargs):
    """
    DenseNet-121 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    net_lambda : function
        Function for model graph creation.
    net_file_path : str or None
        File path for pretrained model or None.
    """
    return get_densenet(num_layers=121, model_name="densenet121", **kwargs)


def densenet161(**kwargs):
    """
    DenseNet-161 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    net_lambda : function
        Function for model graph creation.
    net_file_path : str or None
        File path for pretrained model or None.
    """
    return get_densenet(num_layers=161, model_name="densenet161", **kwargs)


def densenet169(**kwargs):
    """
    DenseNet-169 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    net_lambda : function
        Function for model graph creation.
    net_file_path : str or None
        File path for pretrained model or None.
    """
    return get_densenet(num_layers=169, model_name="densenet169", **kwargs)


def densenet201(**kwargs):
    """
    DenseNet-201 model from 'Densely Connected Convolutional Networks,' https://arxiv.org/abs/1608.06993.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    net_lambda : function
        Function for model graph creation.
    net_file_path : str or None
        File path for pretrained model or None.
    """
    return get_densenet(num_layers=201, model_name="densenet201", **kwargs)


def _test():
    import numpy as np
    from .model_store import init_variables_from_state_dict

    pretrained = False

    models = [
        densenet121,
        densenet161,
        densenet169,
        densenet201,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        x = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224),
            name='xx')
        y_net = net(x)

        weight_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != densenet121 or weight_count == 7978856)
        assert (model != densenet161 or weight_count == 28681000)
        assert (model != densenet169 or weight_count == 14149480)
        assert (model != densenet201 or weight_count == 20013928)

        with tf.Session() as sess:
            if pretrained:
                init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
            else:
                sess.run(tf.global_variables_initializer())
            x_value = np.zeros((1, 3, 224, 224), np.float32)
            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
