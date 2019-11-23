"""
    MnasNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,' https://arxiv.org/abs/1807.11626.
"""

__all__ = ['MnasNet', 'mnasnet_b1', 'mnasnet_a1', 'mnasnet_small']

import os
import tensorflow as tf
from .common import is_channels_first, flatten, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block,\
    se_block, round_channels


def dws_exp_se_res_unit(x,
                        in_channels,
                        out_channels,
                        strides=1,
                        use_kernel3=True,
                        exp_factor=1,
                        se_factor=0,
                        use_skip=True,
                        activation="relu",
                        training=False,
                        data_format="channels_last",
                        name="dws_exp_se_res_unit"):
    """
    Depthwise separable expanded residual unit with SE-block. Here it used as MnasNet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution layer.
    use_kernel3 : bool, default True
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int, default 1
        Expansion factor for each unit.
    se_factor : int, default 0
        SE reduction factor for each unit.
    use_skip : bool, default True
        Whether to use skip connection.
    activation : str, default 'relu'
        Activation function or name of activation function.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    name : str, default 'dws_exp_se_res_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    assert (exp_factor >= 1)
    residual = (in_channels == out_channels) and (strides == 1) and use_skip
    use_exp_conv = exp_factor > 1
    use_se = se_factor > 0
    mid_channels = exp_factor * in_channels
    dwconv_block_fn = dwconv3x3_block if use_kernel3 else dwconv5x5_block

    if residual:
        identity = x

    if use_exp_conv:
        x = conv1x1_block(
            x=x,
            in_channels=in_channels,
            out_channels=mid_channels,
            activation=activation,
            training=training,
            data_format=data_format,
            name=name + "/exp_conv")
    x = dwconv_block_fn(
        x=x,
        in_channels=mid_channels,
        out_channels=mid_channels,
        strides=strides,
        activation=activation,
        training=training,
        data_format=data_format,
        name=name + "/dw_conv")
    if use_se:
        x = se_block(
            x=x,
            channels=mid_channels,
            reduction=(exp_factor * se_factor),
            approx_sigmoid=False,
            round_mid=False,
            activation=activation,
            data_format=data_format,
            name=name + "/se")
    x = conv1x1_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activation=None,
        training=training,
        data_format=data_format,
        name=name + "/pw_conv")

    if residual:
        x = x + identity

    return x


def mnas_init_block(x,
                    in_channels,
                    out_channels,
                    mid_channels,
                    use_skip,
                    training,
                    data_format,
                    name="mnas_init_block"):
    """
    MnasNet specific initial block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    use_skip : bool
        Whether to use skip connection in the second block.
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
        out_channels=mid_channels,
        strides=2,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = dws_exp_se_res_unit(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        use_skip=use_skip,
        training=training,
        data_format=data_format,
        name=name + "/conv2")
    return x


def mnas_final_block(x,
                     in_channels,
                     out_channels,
                     mid_channels,
                     use_skip,
                     training,
                     data_format,
                     name="mnas_final_block"):
    """
    MnasNet specific final block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    use_skip : bool
        Whether to use skip connection in the second block.
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
    x = dws_exp_se_res_unit(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        exp_factor=6,
        use_skip=use_skip,
        training=training,
        data_format=data_format,
        name=name + "/conv1")
    x = conv1x1_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
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
        Number of output channels for the initial unit.
    final_block_channels : list of 2 int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    se_factors : list of list of int
        SE reduction factor for each unit.
    init_block_use_skip : bool
        Whether to use skip connection in the initial unit.
    final_block_use_skip : bool
        Whether to use skip connection in the final block of the feature extractor.
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
                 kernels3,
                 exp_factors,
                 se_factors,
                 init_block_use_skip,
                 final_block_use_skip,
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
        self.kernels3 = kernels3
        self.exp_factors = exp_factors
        self.se_factors = se_factors
        self.init_block_use_skip = init_block_use_skip
        self.final_block_use_skip = final_block_use_skip
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
            out_channels=self.init_block_channels[1],
            mid_channels=self.init_block_channels[0],
            use_skip=self.init_block_use_skip,
            training=training,
            data_format=self.data_format,
            name="features/init_block")
        in_channels = self.init_block_channels[1]
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) else 1
                use_kernel3 = self.kernels3[i][j] == 1
                exp_factor = self.exp_factors[i][j]
                se_factor = self.se_factors[i][j]
                x = dws_exp_se_res_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor,
                    se_factor=se_factor,
                    training=training,
                    data_format=self.data_format,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
        x = mnas_final_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.final_block_channels[1],
            mid_channels=self.final_block_channels[0],
            use_skip=self.final_block_use_skip,
            training=training,
            data_format=self.data_format,
            name="features/final_block")
        # in_channels = self.final_block_channels[1]
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


def get_mnasnet(version,
                width_scale,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".keras", "models"),
                **kwargs):
    """
    Create MnasNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('b1', 'a1' or 'small').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    if version == "b1":
        init_block_channels = [32, 16]
        final_block_channels = [320, 1280]
        channels = [[24, 24, 24], [40, 40, 40], [80, 80, 80, 96, 96], [192, 192, 192, 192]]
        kernels3 = [[1, 1, 1], [0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0]]
        exp_factors = [[3, 3, 3], [3, 3, 3], [6, 6, 6, 6, 6], [6, 6, 6, 6]]
        se_factors = [[0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0]]
        init_block_use_skip = False
        final_block_use_skip = False
    elif version == "a1":
        init_block_channels = [32, 16]
        final_block_channels = [320, 1280]
        channels = [[24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        kernels3 = [[1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        exp_factors = [[6, 6], [3, 3, 3], [6, 6, 6, 6, 6, 6], [6, 6, 6]]
        se_factors = [[0, 0], [4, 4, 4], [0, 0, 0, 0, 4, 4], [4, 4, 4]]
        init_block_use_skip = False
        final_block_use_skip = True
    elif version == "small":
        init_block_channels = [8, 8]
        final_block_channels = [144, 1280]
        channels = [[16], [16, 16], [32, 32, 32, 32, 32, 32, 32], [88, 88, 88]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 1, 1, 1], [0, 0, 0]]
        exp_factors = [[3], [6, 6], [6, 6, 6, 6, 6, 6, 6], [6, 6, 6]]
        se_factors = [[0], [0, 0], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4]]
        init_block_use_skip = True
        final_block_use_skip = True
    else:
        raise ValueError("Unsupported MnasNet version {}".format(version))

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale)

    net = MnasNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        se_factors=se_factors,
        init_block_use_skip=init_block_use_skip,
        final_block_use_skip=final_block_use_skip,
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


def mnasnet_b1(**kwargs):
    """
    MnasNet-B1 model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(version="b1", width_scale=1.0, model_name="mnasnet_b1", **kwargs)


def mnasnet_a1(**kwargs):
    """
    MnasNet-A1 model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(version="a1", width_scale=1.0, model_name="mnasnet_a1", **kwargs)


def mnasnet_small(**kwargs):
    """
    MnasNet-Small model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mnasnet(version="small", width_scale=1.0, model_name="mnasnet_small", **kwargs)


def _test():
    import numpy as np

    # import logging
    # logging.getLogger("tensorflow").disabled = True

    data_format = "channels_last"
    pretrained = False

    models = [
        mnasnet_b1,
        mnasnet_a1,
        mnasnet_small,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)
        x = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224) if is_channels_first(data_format) else (None, 224, 224, 3),
            name="xx")
        y_net = net(x)

        weight_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mnasnet_b1 or weight_count == 4383312)
        assert (model != mnasnet_a1 or weight_count == 3887038)
        assert (model != mnasnet_small or weight_count == 2030264)

        with tf.compat.v1.Session() as sess:
            if pretrained:
                from .model_store import init_variables_from_state_dict
                init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
            x_value = np.zeros((1, 3, 224, 224) if is_channels_first(data_format) else (1, 224, 224, 3), np.float32)
            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.compat.v1.reset_default_graph()


if __name__ == "__main__":
    _test()
