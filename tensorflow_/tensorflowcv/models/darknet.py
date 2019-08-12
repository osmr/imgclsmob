"""
    DarkNet for ImageNet-1K, implemented in TensorFlow.
    Original source: 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.
"""

__all__ = ['DarkNet', 'darknet_ref', 'darknet_tiny', 'darknet19']

import os
import tensorflow as tf
from .common import conv2d, maxpool2d, conv1x1_block, conv3x3_block, is_channels_first, flatten


def dark_convYxY(x,
                 in_channels,
                 out_channels,
                 alpha,
                 pointwise,
                 training,
                 data_format,
                 name="dark_convYxY"):
    """
    DarkNet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    alpha : float
        Slope coefficient for Leaky ReLU activation.
    pointwise : bool
        Whether use 1x1 (pointwise) convolution or 3x3 convolution.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'dark_convYxY'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if pointwise:
        return conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=(lambda y: tf.nn.leaky_relu(y, alpha=alpha, name=name + "/activ")),
            training=training,
            data_format=data_format,
            name=name)
    else:
        return conv3x3_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=(lambda y: tf.nn.leaky_relu(y, alpha=alpha, name=name + "/activ")),
            training=training,
            data_format=data_format,
            name=name)


class DarkNet(object):
    """
    DarkNet model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    odd_pointwise : bool
        Whether pointwise convolution layer is used for each odd unit.
    avg_pool_size : int
        Window size of the final average pooling.
    cls_activ : bool
        Whether classification convolution layer uses an activation.
    alpha : float, default 0.1
        Slope coefficient for Leaky ReLU activation.
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
                 odd_pointwise,
                 avg_pool_size,
                 cls_activ,
                 alpha=0.1,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DarkNet, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.odd_pointwise = odd_pointwise
        self.avg_pool_size = avg_pool_size
        self.cls_activ = cls_activ
        self.alpha = alpha
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
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                x = dark_convYxY(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    alpha=self.alpha,
                    pointwise=(len(channels_per_stage) > 1) and not (((j + 1) % 2 == 1) ^ self.odd_pointwise),
                    training=training,
                    data_format=self.data_format,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
            if i != len(self.channels) - 1:
                x = maxpool2d(
                    x=x,
                    pool_size=2,
                    strides=2,
                    data_format=self.data_format,
                    name="features/pool{}".format(i + 1))

        x = conv2d(
            x=x,
            in_channels=in_channels,
            out_channels=self.classes,
            kernel_size=1,
            data_format=self.data_format,
            name="output/final_conv")
        if self.cls_activ:
            x = tf.nn.leaky_relu(x, alpha=self.alpha, name="output/final_activ")
        x = tf.keras.layers.AveragePooling2D(
            pool_size=self.avg_pool_size,
            strides=1,
            data_format=self.data_format,
            name="output/final_pool")(x)
        # x = tf.layers.flatten(x)
        x = flatten(
            x=x,
            data_format=self.data_format)

        return x


def get_darknet(version,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create DarkNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('ref', 'tiny' or '19').
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

    if version == 'ref':
        channels = [[16], [32], [64], [128], [256], [512], [1024]]
        odd_pointwise = False
        avg_pool_size = 3
        cls_activ = True
    elif version == 'tiny':
        channels = [[16], [32], [16, 128, 16, 128], [32, 256, 32, 256], [64, 512, 64, 512, 128]]
        odd_pointwise = True
        avg_pool_size = 14
        cls_activ = False
    elif version == '19':
        channels = [[32], [64], [128, 64, 128], [256, 128, 256], [512, 256, 512, 256, 512],
                    [1024, 512, 1024, 512, 1024]]
        odd_pointwise = False
        avg_pool_size = 7
        cls_activ = False
    else:
        raise ValueError("Unsupported DarkNet version {}".format(version))

    net = DarkNet(
        channels=channels,
        odd_pointwise=odd_pointwise,
        avg_pool_size=avg_pool_size,
        cls_activ=cls_activ,
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


def darknet_ref(**kwargs):
    """
    DarkNet 'Reference' model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

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
    return get_darknet(version="ref", model_name="darknet_ref", **kwargs)


def darknet_tiny(**kwargs):
    """
    DarkNet Tiny model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

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
    return get_darknet(version="tiny", model_name="darknet_tiny", **kwargs)


def darknet19(**kwargs):
    """
    DarkNet-19 model from 'Darknet: Open source neural networks in c,' https://github.com/pjreddie/darknet.

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
    return get_darknet(version="19", model_name="darknet19", **kwargs)


def _test():
    import numpy as np

    data_format = "channels_last"
    pretrained = False

    models = [
        darknet_ref,
        darknet_tiny,
        darknet19,
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
        assert (model != darknet_ref or weight_count == 7319416)
        assert (model != darknet_tiny or weight_count == 1042104)
        assert (model != darknet19 or weight_count == 20842376)

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
