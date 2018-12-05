"""
    ChannelNet, implemented in TensorFlow.
    Original paper: 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions,'
    https://arxiv.org/abs/1809.01330.
"""

__all__ = ['ChannelNet', 'channelnet']

import os
import tensorflow as tf


def batchnorm(x,
              training=True,
              act_fn=tf.nn.relu6,
              name="batchnorm"):
    data_format = 'NCHW'
    not_final = True
    x = tf.contrib.layers.batch_norm(
        x,
        decay=0.9997,
        scale=not_final,
        center=not_final,
        activation_fn=act_fn,
        fused=True,
        epsilon=1e-3,
        is_training=training,
        data_format=data_format,
        scope=name + '/batch_norm')
    return x


def channet_conv(x,
                 out_channels,
                 kernel_size,
                 strides=1,
                 dropout_rate=1.0,
                 training=True,
                 act_fn=tf.nn.relu6,
                 name="channet_conv"):
    data_format = 'NCHW'
    x = tf.contrib.layers.conv2d(
        x,
        out_channels,
        kernel_size,
        scope=name,
        stride=strides,
        data_format=data_format,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
        biases_initializer=None)
    if dropout_rate < 1.0:
        x = tf.contrib.layers.dropout(
            x,
            dropout_rate,
            is_training=training,
            scope=name)
    x = batchnorm(
        x=x,
        training=training,
        act_fn=act_fn,
        name=name)
    return x


def dw_conv2d(x,
              kernel_size,
              strides,
              dropout_rate=1.0,
              training=True,
              act_fn=None,
              name="dw_conv2d"):
    data_format = 'NCHW'
    shape = list(kernel_size) + [x.shape[data_format.index('C')].value, 1]
    weights = tf.get_variable(
        name + '/conv/weight_depths',
        shape,
        initializer=tf.truncated_normal_initializer(stddev=0.09))
    if data_format == 'NCHW':
        strides = [1, 1, strides, strides]
    else:
        strides = [1, strides, strides, 1]
    x = tf.nn.depthwise_conv2d(
        x,
        weights,
        strides,
        'SAME',
        name=name + '/depthwise_conv2d',
        data_format=data_format)
    x = act_fn(x) if act_fn else x
    return x


def channet_dws_conv_block(x,
                           out_channels,
                           strides,
                           dropout_rate=1.0,
                           act_fn=tf.nn.relu6,
                           training=False,
                           name="channet_dws_conv_block"):
    x = dw_conv2d(
        x=x,
        kernel_size=(3, 3),
        strides=strides,
        dropout_rate=dropout_rate,
        training=training,
        name=name + '/conv1')
    x = channet_conv(
        x,
        out_channels=out_channels,
        kernel_size=(1, 1),
        strides=1,
        dropout_rate=dropout_rate,
        training=training,
        act_fn=act_fn,
        name=name + '/conv2')
    return x


def pure_conv2d(outs,
                num_outs,
                kernel,
                scope,
                keep_r=1.0,
                train=True,
                padding='SAME',
                chan_num=1):
    data_format = 'NCHW'
    stride = int(outs.shape[data_format.index('C')].value/num_outs)
    if data_format == 'NHWC':
        strides = (1, 1, stride)
        axis = -1
        df = 'channels_last'
    else:
        strides = (stride, 1, 1)
        axis = 1
        df = 'channels_first'
    outs = tf.expand_dims(outs, axis=axis, name=scope+'/expand_dims')
    outs = tf.layers.conv3d(
        inputs=outs,
        filters=chan_num,
        kernel_size=kernel,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=df,
        name=scope+'/pure_conv',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.09))
    if keep_r < 1.0:
        outs = tf.contrib.layers.dropout(
            outs, keep_r, is_training=train, scope=scope)
    if chan_num == 1:
        outs = tf.squeeze(outs, axis=[axis], name=scope+'/squeeze')
    return outs


def single_block(outs,
                 block_num,
                 dropout_rate,
                 training,
                 scope):
    data_format = 'NCHW'
    num_outs = outs.shape[data_format.index('C')].value
    for i in range(block_num):
        outs = channet_dws_conv_block(x=outs, out_channels=num_outs, strides=1, dropout_rate=dropout_rate,
                                      training=training, name=scope + '/conv_%s' % i)
    return outs


def conv_group_block(x,
                     block_num,
                     group,
                     dropout_rate,
                     training,
                     name):
    data_format = 'NCHW'
    num_outs = int(x.shape[data_format.index('C')].value / group)
    shape = [1, 1, 4*group] if data_format == 'NHWC' else [4*group, 1, 1]
    results = []
    conv_outs = pure_conv2d(
        outs=x,
        num_outs=num_outs,
        kernel=shape,
        scope=name + '/pure_conv',
        keep_r=dropout_rate,
        train=training,
        chan_num=group)
    axis = -1 if data_format=='NHWC' else 1
    conv_outs = tf.unstack(conv_outs, axis=axis, name=name + '/unstack')
    for g in range(group):
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
                       block_num,
                       group,
                       dropout_rate,
                       training,
                       name):
    data_format = 'NCHW'
    results = []
    split_outs = tf.split(x, group, data_format.index('C'), name=name + '/split')
    for g in range(group):
        cur_outs = single_block(
            split_outs[g],
            block_num,
            dropout_rate,
            training,
            name + '/group_%s' % g)
        results.append(cur_outs)
    results = tf.concat(results, data_format.index('C'), name=name + '/concat')
    return results


def dense(x,
          dim,
          name="dense"):
    x = tf.contrib.layers.fully_connected(
        x,
        dim,
        activation_fn=None,
        scope=name + '/dense',
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09))
    return x


def out_block(x,
              num_classes,
              name="out_block"):
    data_format = 'NCHW'
    axes = [2, 3] if data_format == 'NCHW' else [1, 2]
    x = tf.reduce_mean(x, axes, name=name + '/pool')
    x = dense(
        x=x,
        dim=num_classes,
        name=name)
    return x


class ChannelNet(object):
    """
    ChannelNet model from 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise
    Convolutions,' https://arxiv.org/abs/1809.01330.

    Parameters:
    ----------
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(ChannelNet, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.in_size = in_size
        self.classes = classes
        self.conf = {
            'group_num': 2,
            'block_num': 2,
        }
        self.data_format = 'NCHW'
        self.dropout_rate = 0.9999
        self.ch_num = 32
        self.block_num = 2
        self.group_num = 2

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
        cur_out_num = self.ch_num
        x = channet_conv(
            x=x,
            out_channels=cur_out_num,
            kernel_size=(3, 3),
            strides=2,
            training=training,
            act_fn=None,
            name='conv_s')

        cur_out_num *= 2
        x2 = channet_dws_conv_block(
            x=x,
            out_channels=cur_out_num,
            strides=1,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_1_0')
        x = tf.concat([x, x2], axis=1, name='add0')

        cur_out_num *= 2
        x = channet_dws_conv_block(
            x=x,
            out_channels=cur_out_num,
            strides=2,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_1_1')
        x2 = channet_dws_conv_block(
            x=x,
            out_channels=cur_out_num,
            strides=1,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_1_2')
        x = tf.concat([x, x2], axis=1, name='add1')

        cur_out_num *= 2
        x = channet_dws_conv_block(
            x=x,
            out_channels=cur_out_num,
            strides=2,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_1_3')
        x2 = channet_dws_conv_block(
            x=x,
            out_channels=cur_out_num,
            strides=1,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_1_4')
        x = tf.concat([x, x2], axis=1, name='add2')

        cur_out_num *= 2
        x = channet_dws_conv_block(
            x=x,
            out_channels=cur_out_num,
            strides=2,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_1_5')
        x2 = simple_group_block(
            x=x,
            block_num=self.block_num,
            group=self.group_num,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_2_1')
        x = tf.add(x, x2, name='add21')

        x = conv_group_block(
            x=x,
            block_num=self.block_num,
            group=self.group_num,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_2_2')
        x2 = conv_group_block(
            x=x,
            block_num=self.block_num,
            group=self.group_num,
            dropout_rate=self.dropout_rate,
            training=training,
            name='conv_2_3')
        x = tf.add(x, x2, name='add23')

        cur_out_num *= 2
        x = channet_dws_conv_block(x=x, out_channels=cur_out_num, strides=2, dropout_rate=self.dropout_rate,
                                   training=training, name='conv_3_0')
        x = channet_dws_conv_block(x=x, out_channels=cur_out_num, strides=1, dropout_rate=self.dropout_rate,
                                   training=training, name='conv_3_1')

        x = out_block(
            x=x,
            num_classes=self.classes,
            name='out')
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

    net = ChannelNet(
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
            x_value = np.zeros((1, 3, 224, 224), np.float32)
            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
