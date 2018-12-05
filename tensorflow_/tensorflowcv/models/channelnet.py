"""
    ChannelNet, implemented in TensorFlow.
    Original paper: 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions,'
    https://arxiv.org/abs/1809.01330.
"""

__all__ = ['ChannelNet', 'channelnet']

import os
import tensorflow as tf
from common import conv2d, batchnorm


def dwconv3x3(x,
              in_channels,
              out_channels,
              strides,
              use_bias=False,
              name="dwconv3x3"):
    """
    3x3 depthwise version of the standard convolution layer.

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
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    name : str, default 'dwconv3x3'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=out_channels,
        use_bias=use_bias,
        name=name)


def channet_conv(x,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 dropout_rate=0.0,
                 activate=True,
                 training=False,
                 name="channet_conv"):
    """
    ChannelNet specific convolution block with Batch normalization and ReLU6 activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Dropout rate.
    activate : bool, default True
        Whether activate the convolution block.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'channet_conv'
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
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        name=name + "/conv")
    if dropout_rate > 0.0:
        x = tf.layers.dropout(
            inputs=x,
            rate=dropout_rate,
            training=training,
            name=name + "/dropout")
    x = batchnorm(
        x=x,
        training=training,
        name=name + "/bn")
    if activate:
        x = tf.nn.relu6(x, name=name + "/activ")
    return x


def channet_conv1x1(x,
                    in_channels,
                    out_channels,
                    strides=1,
                    groups=1,
                    use_bias=False,
                    dropout_rate=0.0,
                    activate=True,
                    training=False,
                    name="channet_conv1x1"):
    """
    1x1 version of ChannelNet specific convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Dropout rate.
    activate : bool, default True
        Whether activate the convolution block.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'channet_conv1x1'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return channet_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        groups=groups,
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        activate=activate,
        training=training,
        name=name)


def channet_conv3x3(x,
                    in_channels,
                    out_channels,
                    strides,
                    padding=1,
                    dilation=1,
                    groups=1,
                    use_bias=False,
                    dropout_rate=0.0,
                    activate=True,
                    training=False,
                    name="channet_conv3x3"):
    """
    3x3 version of the standard convolution block.

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
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Dropout rate.
    activate : bool, default True
        Whether activate the convolution block.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'channet_conv3x3'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return channet_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        dropout_rate=dropout_rate,
        activate=activate,
        training=training,
        name=name)


def channet_dws_conv_block(x,
                           in_channels,
                           out_channels,
                           strides,
                           dropout_rate=1.0,
                           training=False,
                           name="channet_dws_conv_block"):
    """
    Depthwise separable convolution block with BatchNorms and activations at last convolution layers.

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
    dropout_rate : float, default 0.0
        Dropout rate.
    training : bool, or a TensorFlow boolean scalar tensor
      Whether to return the output in training mode or in inference mode.
    name : str, default 'channet_dws_conv_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = dwconv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=in_channels,
        strides=strides,
        name=name + '/dw_conv')
    x = channet_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_rate=dropout_rate,
        training=training,
        name=name + '/pw_conv')
    return x


def single_block(x,
                 num_blocks,
                 dropout_rate,
                 training,
                 name):
    data_format = 'NCHW'
    in_channels = x.shape[data_format.index('C')].value
    for i in range(num_blocks):
        x = channet_dws_conv_block(
            x=x,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=1,
            dropout_rate=dropout_rate,
            training=training,
            name=name + '/block{}'.format(i + 1))
    return x


def pure_conv2d(outs,
                out_channels,
                kernel,
                scope,
                keep_r=1.0,
                train=True,
                padding='SAME',
                chan_num=1):
    data_format = 'NCHW'
    stride = int(outs.shape[data_format.index('C')].value / out_channels)
    if data_format == 'NHWC':
        strides = (1, 1, stride)
        axis = -1
        df = 'channels_last'
    else:
        strides = (stride, 1, 1)
        axis = 1
        df = 'channels_first'
    outs = tf.expand_dims(outs, axis=axis, name=scope + '/expand_dims')
    outs = tf.layers.conv3d(
        inputs=outs,
        filters=chan_num,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=df,
        name=scope + '/pure_conv',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.09))
    if keep_r < 1.0:
        outs = tf.contrib.layers.dropout(
            outs, keep_r, is_training=train, scope=scope)
    if chan_num == 1:
        outs = tf.squeeze(outs, axis=[axis], name=scope + '/squeeze')
    return outs


def conv_group_block(x,
                     block_num,
                     groups,
                     dropout_rate,
                     training,
                     name):
    data_format = 'NCHW'
    num_outs = int(x.shape[data_format.index('C')].value / groups)
    shape = [1, 1, 4 * groups] if data_format == 'NHWC' else [4 * groups, 1, 1]
    results = []
    conv_outs = pure_conv2d(
        outs=x,
        out_channels=num_outs,
        kernel=shape,
        scope=name + '/pure_conv',
        keep_r=dropout_rate,
        train=training,
        chan_num=groups)
    axis = -1 if data_format == 'NHWC' else 1
    conv_outs = tf.unstack(conv_outs, axis=axis, name=name + '/unstack')
    for g in range(groups):
        cur_outs = single_block(
            conv_outs[g],
            block_num,
            dropout_rate,
            training,
            name + '/group_%s' % g)
        results.append(cur_outs)
    results = tf.concat(results, data_format.index('C'), name=name + '/concat')
    return results


def simple_group_block(x,
                       channels,
                       block_num,
                       groups,
                       dropout_rate,
                       training=False,
                       name="simple_group_block"):
    data_format = 'NCHW'
    x_out_list = []
    x_in_list = tf.split(x, groups, data_format.index('C'), name=name + '/split')
    for gi in range(groups):
        xi_out = single_block(
            x_in_list[gi],
            block_num,
            dropout_rate,
            training,
            name + '/group_%s' % gi)
        x_out_list.append(xi_out)
    x_out = tf.concat(x_out_list, data_format.index('C'), name=name + '/concat')
    return x_out


def channet_unit(x,
                 in_channels,
                 out_channels_list,
                 strides,
                 num_blocks,
                 num_groups,
                 dropout_rate,
                 block_names,
                 merge_type,
                 training=False,
                 name="channet_unit"):
    assert (len(block_names) == 2)
    x_outs = []
    for i, (out_channels, block_name) in enumerate(zip(out_channels_list, block_names)):
        strides_i = (strides if i == 0 else 1)
        name_i = name + '/block{}'.format(i + 1)
        assert (x.shape[1].value == in_channels)
        if block_name == "channet_conv3x3":
            x = channet_conv3x3(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides_i,
                dropout_rate=dropout_rate,
                activate=False,
                training=training,
                name=name_i)
        elif block_name == "channet_dws_conv_block":
            x = channet_dws_conv_block(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides_i,
                dropout_rate=dropout_rate,
                training=training,
                name=name_i)
        elif block_name == "conv_group_block":
            x = conv_group_block(
                x=x,
                block_num=num_blocks,
                groups=num_groups,
                dropout_rate=dropout_rate,
                training=training,
                name=name_i)
        elif block_name == "simple_group_block":
            x = simple_group_block(
                x=x,
                channels=in_channels,
                block_num=num_blocks,
                groups=num_groups,
                dropout_rate=dropout_rate,
                training=training,
                name=name_i)
        else:
            raise NotImplementedError()
        x_outs = x_outs + [x]
        in_channels = out_channels
    if merge_type == "seq":
        x = x_outs[-1]
    elif merge_type == "add":
        x = tf.add(*x_outs, name=name + '/add')
    elif merge_type == "cat":
        x = tf.concat(x_outs, axis=1, name=name + '/cat')
    else:
        raise NotImplementedError()
    return x


class ChannelNet(object):
    """
    ChannelNet model from 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise
    Convolutions,' https://arxiv.org/abs/1809.01330.

    Parameters:
    ----------
    channels : list of list of list of int
        Number of output channels for each unit.
    block_names : list of list of list of str
        Names of blocks for each unit.
    block_names : list of list of str
        Merge types for each unit.
    dropout_rate : float, default 0.0001
        Dropout rate.
    num_blocks : int, default 2
        Block count architectural parameter.
    num_groups : int, default 2
        Group count architectural parameter.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 block_names,
                 merge_types,
                 dropout_rate=0.0001,
                 num_blocks=2,
                 num_groups=2,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(ChannelNet, self).__init__(**kwargs)
        self.channels = channels
        self.block_names = block_names
        self.merge_types = merge_types
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_groups = num_groups
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
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) else 1
                x = channet_unit(
                    x=x,
                    in_channels=in_channels,
                    out_channels_list=out_channels,
                    strides=strides,
                    num_blocks=self.num_blocks,
                    num_groups=self.num_groups,
                    dropout_rate=self.dropout_rate,
                    block_names=self.block_names[i][j],
                    merge_type=self.merge_types[i][j],
                    training=training,
                    name="features/stage{}/unit{}".format(i + 1, j + 1))
                if self.merge_types[i][j] == "cat":
                    in_channels = sum(out_channels)
                else:
                    in_channels = out_channels[-1]

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


def get_channelnet(model_name=None,
                   pretrained=False,
                   root=os.path.join('~', '.tensorflow', 'models'),
                   **kwargs):
    """
    Create ChannelNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    channels = [[[32, 64]], [[128, 128]], [[256, 256]], [[512, 512], [512, 512]], [[1024, 1024]]]
    block_names = [[["channet_conv3x3", "channet_dws_conv_block"]],
                   [["channet_dws_conv_block", "channet_dws_conv_block"]],
                   [["channet_dws_conv_block", "channet_dws_conv_block"]],
                   [["channet_dws_conv_block", "simple_group_block"], ["conv_group_block", "conv_group_block"]],
                   [["channet_dws_conv_block", "channet_dws_conv_block"]]]
    merge_types = [["cat"], ["cat"], ["cat"], ["add", "add"], ["seq"]]

    net = ChannelNet(
        channels=channels,
        block_names=block_names,
        merge_types=merge_types,
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


def channelnet(**kwargs):
    """
    AlexNet model from 'One weird trick for parallelizing convolutional neural networks,'
    https://arxiv.org/abs/1404.5997.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_channelnet(model_name="alexnet", **kwargs)


def _test():
    import numpy as np
    from model_store import init_variables_from_state_dict

    pretrained = False

    models = [
        channelnet,
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
        assert (model != channelnet or weight_count == 3875112)

        with tf.Session() as sess:
            if pretrained:
                init_variables_from_state_dict(sess=sess, state_dict=net.state_dict)
            else:
                sess.run(tf.global_variables_initializer())
            # x_value = np.zeros((1, 3, 224, 224), np.float32)
            # y = sess.run(y_net, feed_dict={x: x_value})
            # assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
