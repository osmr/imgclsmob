"""
    MobileNetV3 for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
"""

__all__ = ['MobileNetV3', 'mobilenetv3_small_w7d20', 'mobilenetv3_small_wd2', 'mobilenetv3_small_w3d4',
           'mobilenetv3_small_w1', 'mobilenetv3_small_w5d4', 'mobilenetv3_large_w7d20', 'mobilenetv3_large_wd2',
           'mobilenetv3_large_w3d4', 'mobilenetv3_large_w1', 'mobilenetv3_large_w5d4']

import os
import tensorflow as tf
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block,\
    se_block, hswish, is_channels_first, flatten
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


def mobilenetv3_unit(x,
                     in_channels,
                     out_channels,
                     exp_channels,
                     strides,
                     use_kernel3,
                     activation,
                     use_se,
                     training,
                     data_format,
                     name="mobilenetv3_unit"):
    """
    MobileNetV3 unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    exp_channels : int
        Number of middle (expanded) channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    activation : str
        Activation function or name of activation function.
    use_se : bool
        Whether to use SE-module.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'mobilenetv3_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    assert (exp_channels >= out_channels)
    residual = (in_channels == out_channels) and (strides == 1)
    use_exp_conv = exp_channels != out_channels
    mid_channels = exp_channels

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
    if use_kernel3:
        x = dwconv3x3_block(
            x=x,
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            activation=activation,
            training=training,
            data_format=data_format,
            name=name + "/conv1")
    else:
        x = dwconv5x5_block(
            x=x,
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            activation=activation,
            name=name + "/conv1")
    if use_se:
        x = se_block(
            x=x,
            channels=mid_channels,
            reduction=4,
            approx_sigmoid=True,
            round_mid=True,
            data_format=data_format,
            name=name + "/se")
    x = conv1x1_block(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        activation=None,
        training=training,
        data_format=data_format,
        name=name + "/conv2")

    if residual:
        x = x + identity

    return x


def mobilenetv3_final_block(x,
                            in_channels,
                            out_channels,
                            use_se,
                            training,
                            data_format,
                            name="mobilenetv3_final_block"):
    """
    MobileNetV3 final block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_se : bool
        Whether to use SE-module.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'mobilenetv3_final_block'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv1x1_block(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        activation="hswish",
        training=training,
        data_format=data_format,
        name=name + "/conv")
    if use_se:
        x = se_block(
            x=x,
            channels=out_channels,
            reduction=4,
            approx_sigmoid=True,
            round_mid=True,
            data_format=data_format,
            name=name + "/se")
    return x


def mobilenetv3_classifier(x,
                           in_channels,
                           out_channels,
                           mid_channels,
                           dropout_rate,
                           training,
                           data_format,
                           name="mobilenetv3_final_block"):
    """
    MobileNetV3 classifier.

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
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    data_format : str
        The ordering of the dimensions in tensors.
    name : str, default 'mobilenetv3_classifier'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        data_format=data_format,
        name=name + "/conv1")
    x = hswish(x, name=name + "/hswish")

    use_dropout = (dropout_rate != 0.0)
    if use_dropout:
        x = tf.keras.layers.Dropout(
            rate=dropout_rate,
            name=name + "dropout")(
            inputs=x,
            training=training)

    x = conv1x1(
        x=x,
        in_channels=mid_channels,
        out_channels=out_channels,
        use_bias=True,
        data_format=data_format,
        name=name + "/conv2")
    return x


class MobileNetV3(object):
    """
    MobileNetV3 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    exp_channels : list of list of int
        Number of middle (expanded) channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    use_relu : list of list of int/bool
        Using ReLU activation flag for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
    final_use_se : bool
        Whether to use SE-module in the final block.
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
                 exp_channels,
                 init_block_channels,
                 final_block_channels,
                 classifier_mid_channels,
                 kernels3,
                 use_relu,
                 use_se,
                 first_stride,
                 final_use_se,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(MobileNetV3, self).__init__(**kwargs)
        assert (data_format in ["channels_last", "channels_first"])
        self.channels = channels
        self.exp_channels = exp_channels
        self.init_block_channels = init_block_channels
        self.final_block_channels = final_block_channels
        self.classifier_mid_channels = classifier_mid_channels
        self.kernels3 = kernels3
        self.use_relu = use_relu
        self.use_se = use_se
        self.first_stride = first_stride
        self.final_use_se = final_use_se
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
        x = conv3x3_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            strides=2,
            activation="hswish",
            training=training,
            data_format=self.data_format,
            name="features/init_block")
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                exp_channels_ij = self.exp_channels[i][j]
                strides = 2 if (j == 0) and ((i != 0) or self.first_stride) else 1
                use_kernel3 = self.kernels3[i][j] == 1
                activation = "relu" if self.use_relu[i][j] == 1 else "hswish"
                use_se_flag = self.use_se[i][j] == 1
                x = mobilenetv3_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    exp_channels=exp_channels_ij,
                    use_kernel3=use_kernel3,
                    strides=strides,
                    activation=activation,
                    use_se=use_se_flag,
                    training=training,
                    data_format=self.data_format,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                in_channels = out_channels
        x = mobilenetv3_final_block(
            x=x,
            in_channels=in_channels,
            out_channels=self.final_block_channels,
            use_se=self.final_use_se,
            training=training,
            data_format=self.data_format,
            name="features/final_block")
        in_channels = self.final_block_channels
        x = tf.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=self.data_format,
            name="features/final_pool")(x)

        x = mobilenetv3_classifier(
            x=x,
            in_channels=in_channels,
            out_channels=self.classes,
            mid_channels=self.classifier_mid_channels,
            dropout_rate=0.2,
            training=training,
            data_format=self.data_format,
            name="output")
        x = flatten(
            x=x,
            data_format=self.data_format)

        return x


def get_mobilenetv3(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".tensorflow", "models"),
                    **kwargs):
    """
    Create MobileNetV3 model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('small' or 'large').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """

    if version == "small":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        first_stride = True
        final_block_channels = 576
    elif version == "large":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        first_stride = False
        final_block_channels = 960
    else:
        raise ValueError("Unsupported MobileNetV3 version {}".format(version))

    final_use_se = False
    classifier_mid_channels = 1280

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        exp_channels = [[round_channels(cij * width_scale) for cij in ci] for ci in exp_channels]
        init_block_channels = round_channels(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale)

    net = MobileNetV3(
        channels=channels,
        exp_channels=exp_channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        use_relu=use_relu,
        use_se=use_se,
        first_stride=first_stride,
        final_use_se=final_use_se,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def mobilenetv3_small_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_small_wd2(**kwargs):
    """
    MobileNetV3 Small 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.5, model_name="mobilenetv3_small_wd2", **kwargs)


def mobilenetv3_small_w3d4(**kwargs):
    """
    MobileNetV3 Small 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.75, model_name="mobilenetv3_small_w3d4", **kwargs)


def mobilenetv3_small_w1(**kwargs):
    """
    MobileNetV3 Small 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.0, model_name="mobilenetv3_small_w1", **kwargs)


def mobilenetv3_small_w5d4(**kwargs):
    """
    MobileNetV3 Small 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.25, model_name="mobilenetv3_small_w5d4", **kwargs)


def mobilenetv3_large_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_large_wd2(**kwargs):
    """
    MobileNetV3 Large 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.5, model_name="mobilenetv3_large_wd2", **kwargs)


def mobilenetv3_large_w3d4(**kwargs):
    """
    MobileNetV3 Large 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.75, model_name="mobilenetv3_large_w3d4", **kwargs)


def mobilenetv3_large_w1(**kwargs):
    """
    MobileNetV3 Large 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.0, model_name="mobilenetv3_large_w1", **kwargs)


def mobilenetv3_large_w5d4(**kwargs):
    """
    MobileNetV3 Large 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.25, model_name="mobilenetv3_large_w5d4", **kwargs)


def _test():
    import numpy as np

    # import logging
    # logging.getLogger("tensorflow").disabled = True

    data_format = "channels_last"
    pretrained = False

    models = [
        mobilenetv3_small_w7d20,
        mobilenetv3_small_wd2,
        mobilenetv3_small_w3d4,
        mobilenetv3_small_w1,
        mobilenetv3_small_w5d4,
        mobilenetv3_large_w7d20,
        mobilenetv3_large_wd2,
        mobilenetv3_large_w3d4,
        mobilenetv3_large_w1,
        mobilenetv3_large_w5d4,
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
        assert (model != mobilenetv3_small_w7d20 or weight_count == 2159600)
        assert (model != mobilenetv3_small_wd2 or weight_count == 2288976)
        assert (model != mobilenetv3_small_w3d4 or weight_count == 2581312)
        assert (model != mobilenetv3_small_w1 or weight_count == 2945288)
        assert (model != mobilenetv3_small_w5d4 or weight_count == 3643632)
        assert (model != mobilenetv3_large_w7d20 or weight_count == 2943080)
        assert (model != mobilenetv3_large_wd2 or weight_count == 3334896)
        assert (model != mobilenetv3_large_w3d4 or weight_count == 4263496)
        assert (model != mobilenetv3_large_w1 or weight_count == 5481752)
        assert (model != mobilenetv3_large_w5d4 or weight_count == 7459144)

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
