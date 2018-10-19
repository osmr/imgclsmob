"""
    ResNeXt & SE-ResNeXt, implemented in TensorFlow.
    Original papers:
    - 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
    - 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['resnext', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d', 'seresnext50_32x4d',
           'seresnext101_32x4d', 'seresnext101_64x4d', 'resnext_conv3x3', 'resnext_conv1x1']

import os
import math
import tensorflow as tf
from .common import conv2d, batchnorm, maxpool2d, se_block


def resnext_conv(x,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups,
                 activate,
                 training,
                 name="resnext_conv"):
    """
    ResNeXt specific convolution block.

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
    groups : int
        Number of groups.
    activate : bool
        Whether activate the convolution block.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'resnext_conv'
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
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=False,
        name=name + "/conv")
    x = batchnorm(
        x=x,
        training=training,
        name=name + "/bn")
    if activate:
        x = tf.nn.relu(x, name=name + "/activ")
    return x


def resnext_conv1x1(x,
                    in_channels,
                    out_channels,
                    strides,
                    activate,
                    training,
                    name="resnext_conv1x1"):
    """
    1x1 version of the ResNeXt specific convolution block.

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
    activate : bool
        Whether activate the convolution block.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'resnext_conv1x1'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return resnext_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=1,
        activate=activate,
        training=training,
        name=name)


def resnext_conv3x3(x,
                    in_channels,
                    out_channels,
                    strides,
                    groups,
                    activate,
                    training,
                    name="resnext_conv3x3"):
    """
    3x3 version of the ResNeXt specific convolution block.

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
    groups : int
        Number of groups.
    activate : bool
        Whether activate the convolution block.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'resnext_conv3x3'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return resnext_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=groups,
        activate=activate,
        training=training,
        name=name)


def resnext_bottleneck(x,
                       in_channels,
                       out_channels,
                       strides,
                       cardinality,
                       bottleneck_width,
                       training,
                       name="resnext_bottleneck"):
    """
    ResNeXt bottleneck block for residual path in ResNeXt unit.

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
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'resnext_bottleneck'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 4
    D = int(math.floor(mid_channels * (bottleneck_width / 64)))
    group_width = cardinality * D

    x = resnext_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=group_width,
        strides=1,
        activate=True,
        training=training,
        name=name + "/conv1")
    x = resnext_conv3x3(
        x=x,
        in_channels=group_width,
        out_channels=group_width,
        strides=strides,
        groups=cardinality,
        activate=True,
        training=training,
        name=name + "/conv2")
    x = resnext_conv1x1(
        x=x,
        in_channels=group_width,
        out_channels=out_channels,
        strides=1,
        activate=False,
        training=training,
        name=name + "/conv3")
    return x


def resnext_unit(x,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 use_se,
                 training,
                 name="resnext_unit"):
    """
    ResNeXt unit with residual connection.

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
    use_se : bool
        Whether to use SE block.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'resnext_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        identity = resnext_conv1x1(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            activate=False,
            training=training,
            name=name + "/identity_conv")
    else:
        identity = x

    x = resnext_bottleneck(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
        training=training,
        name=name + "/body")

    if use_se:
        x = se_block(
            x=x,
            channels=out_channels,
            name=name + "/se")

    x = x + identity

    x = tf.nn.relu(x, name=name + "activ")
    return x


def resnext_init_block(x,
                       in_channels,
                       out_channels,
                       training,
                       name="resnext_init_block"):
    """
    ResNeXt specific initial block.

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
    name : str, default 'resnext_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = resnext_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=2,
        padding=3,
        groups=1,
        activate=True,
        training=training,
        name=name + "/conv")
    x = maxpool2d(
        x=x,
        pool_size=3,
        strides=2,
        padding=1,
        name=name + "/pool")
    return x


def resnext(x,
            channels,
            init_block_channels,
            cardinality,
            bottleneck_width,
            use_se,
            in_channels=3,
            classes=1000,
            training=False):
    """
    ResNeXt model from 'Aggregated Residual Transformations for Deep Neural Networks,' http://arxiv.org/abs/1611.05431.
    Also this class implements SE-ResNeXt from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    use_se : bool
        Whether to use SE block.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = resnext_init_block(
        x=x,
        in_channels=in_channels,
        out_channels=init_block_channels,
        training=training,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            x = resnext_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                cardinality=cardinality,
                bottleneck_width=bottleneck_width,
                use_se=use_se,
                training=training,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = tf.layers.average_pooling2d(
        inputs=x,
        pool_size=7,
        strides=1,
        data_format='channels_first',
        name="features/final_pool")

    x = tf.layers.flatten(x)
    x = tf.layers.dense(
        inputs=x,
        units=classes,
        name="output")

    return x


def get_resnext(blocks,
                cardinality,
                bottleneck_width,
                use_se=False,
                model_name=None,
                pretrained=False,
                root=os.path.join('~', '.tensorflow', 'models'),
                **kwargs):
    """
    Create ResNeXt or SE-ResNeXt model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    use_se : bool
        Whether to use SE block.
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

    if blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Unsupported ResNeXt with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    def net_lambda(x,
                   training=False,
                   channels=channels,
                   init_block_channels=init_block_channels,
                   cardinality=cardinality,
                   bottleneck_width=bottleneck_width,
                   use_se=use_se):
        y_net = resnext(
            x=x,
            channels=channels,
            init_block_channels=init_block_channels,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width,
            use_se=use_se,
            training=training,
            **kwargs)
        return y_net

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net_file_path = get_model_file(
            model_name=model_name,
            local_model_store_dir_path=root)
    else:
        net_file_path = None

    return net_lambda, net_file_path


def resnext50_32x4d(**kwargs):
    """
    ResNeXt-50 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext(blocks=50, cardinality=32, bottleneck_width=4, model_name="resnext50_32x4d", **kwargs)


def resnext101_32x4d(**kwargs):
    """
    ResNeXt-101 (32x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext(blocks=101, cardinality=32, bottleneck_width=4, model_name="resnext101_32x4d", **kwargs)


def resnext101_64x4d(**kwargs):
    """
    ResNeXt-101 (64x4d) model from 'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.

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
    return get_resnext(blocks=101, cardinality=64, bottleneck_width=4, model_name="resnext101_64x4d", **kwargs)


def seresnext50_32x4d(**kwargs):
    """
    SE-ResNeXt-50 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_resnext(blocks=50, cardinality=32, bottleneck_width=4, use_se=True, model_name="seresnext50_32x4d",
                       **kwargs)


def seresnext101_32x4d(**kwargs):
    """
    SE-ResNeXt-101 (32x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_resnext(blocks=101, cardinality=32, bottleneck_width=4, use_se=True, model_name="seresnext101_32x4d",
                       **kwargs)


def seresnext101_64x4d(**kwargs):
    """
    SE-ResNeXt-101 (64x4d) model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    return get_resnext(blocks=101, cardinality=64, bottleneck_width=4, use_se=True, model_name="seresnext101_64x4d",
                       **kwargs)


def _test():
    import numpy as np
    from .model_store import load_model

    pretrained = False

    models = [
        resnext50_32x4d,
        resnext101_32x4d,
        resnext101_64x4d,
        seresnext50_32x4d,
        seresnext101_32x4d,
        seresnext101_64x4d,
    ]

    for model in models:

        net_lambda, net_file_path = model(pretrained=pretrained)

        x = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224),
            name='xx')
        y_net = net_lambda(x)

        weight_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnext50_32x4d or weight_count == 25028904)
        assert (model != resnext101_32x4d or weight_count == 44177704)
        assert (model != resnext101_64x4d or weight_count == 83455272)
        assert (model != seresnext50_32x4d or weight_count == 27559896)
        assert (model != seresnext101_32x4d or weight_count == 48955416)
        assert (model != seresnext101_64x4d or weight_count == 88232984)

        with tf.Session() as sess:
            if pretrained:
                load_model(sess=sess, file_path=net_file_path)
            else:
                sess.run(tf.global_variables_initializer())
            x_value = np.zeros((1, 3, 224, 224), np.float32)
            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
