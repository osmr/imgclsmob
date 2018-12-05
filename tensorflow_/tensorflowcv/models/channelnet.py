"""
    ChannelNet, implemented in TensorFlow.
    Original paper: 'ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions,'
    https://arxiv.org/abs/1809.01330.
"""

__all__ = ['ChannelNet', 'channelnet']

import os
import tensorflow as tf


def batchnorm(x,
              scope,
              training=True,
              act_fn=tf.nn.relu6,
              data_format='NHWC'):
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
        scope=scope+'/batch_norm')
    return x


def channet_conv(x,
                 out_channels,
                 kernel,
                 scope,
                 stride=1,
                 dropout_rate=1.0,
                 training=True,
                 act_fn=tf.nn.relu6,
                 data_format='NHWC'):
    x = tf.contrib.layers.conv2d(
        x,
        out_channels,
        kernel,
        scope=scope,
        stride=stride,
        data_format=data_format,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09),
        biases_initializer=None)
    if dropout_rate < 1.0:
        x = tf.contrib.layers.dropout(
            x,
            dropout_rate,
            is_training=training,
            scope=scope)
    x = batchnorm(
        x=x,
        scope=scope,
        training=training,
        act_fn=act_fn,
        data_format=data_format)
    return x


def dw_conv2d(outs,
              kernel,
              stride,
              scope,
              keep_r=1.0,
              train=True,
              act_fn=None,
              data_format='NHWC'):
    shape = list(kernel)+[outs.shape[data_format.index('C')].value, 1]
    weights = tf.get_variable(
        scope+'/conv/weight_depths',
        shape,
        initializer=tf.truncated_normal_initializer(stddev=0.09))
    if data_format == 'NCHW':
        strides = [1, 1, stride, stride]
    else:
        strides = [1, stride, stride, 1]
    outs = tf.nn.depthwise_conv2d(
        outs,
        weights,
        strides,
        'SAME',
        name=scope+'/depthwise_conv2d',
        data_format=data_format)
    return act_fn(outs) if act_fn else outs


def dw_block(outs,
             num_outs,
             stride,
             scope,
             dropout_rate,
             is_train,
             act_fn=tf.nn.relu6,
             data_format='NHWC'):
    outs = dw_conv2d(
        outs,
        (3, 3),
        stride,
        scope+'/conv1',
        dropout_rate,
        is_train,
        data_format=data_format)
    outs = channet_conv(
        outs,
        num_outs,
        (1, 1),
        scope + '/conv2',
        1,
        dropout_rate,
        is_train,
        act_fn=act_fn,
        data_format=data_format)
    return outs


def pure_conv2d(outs,
                num_outs,
                kernel,
                scope,
                keep_r=1.0,
                train=True,
                padding='SAME',
                chan_num=1,
                data_format='NHWC'):
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
                 keep_r,
                 is_train,
                 scope,
                 data_format,
                 *args):
    num_outs = outs.shape[data_format.index('C')].value
    for i in range(block_num):
        outs = dw_block(
            outs=outs,
            num_outs=num_outs,
            stride=1,
            scope=scope+'/conv_%s' % i,
            dropout_rate=keep_r,
            is_train=is_train,
            data_format=data_format)
    return outs


def conv_group_block(outs,
                     block_num,
                     keep_r,
                     is_train,
                     scope,
                     data_format,
                     group,
                     *args):
    num_outs = int(outs.shape[data_format.index('C')].value/group)
    shape = [1, 1, 4*group] if data_format == 'NHWC' else [4*group, 1, 1]
    results = []
    conv_outs = pure_conv2d(
        outs=outs,
        num_outs=num_outs,
        kernel=shape,
        scope=scope+'/pure_conv',
        keep_r=keep_r,
        train=is_train,
        chan_num=group,
        data_format=data_format)
    axis = -1 if data_format=='NHWC' else 1
    conv_outs = tf.unstack(conv_outs, axis=axis, name=scope+'/unstack')
    for g in range(group):
        cur_outs = single_block(
            conv_outs[g],
            block_num,
            keep_r,
            is_train,
            scope+'/group_%s' % g,
            data_format)
        results.append(cur_outs)
    results = tf.concat(results, data_format.index('C'), name=scope+'/concat')
    return results


def simple_group_block(outs,
                       block_num,
                       keep_r,
                       is_train,
                       scope,
                       data_format,
                       group,
                       *args):
    results = []
    split_outs = tf.split(outs, group, data_format.index('C'), name=scope+'/split')
    for g in range(group):
        cur_outs = single_block(
            split_outs[g],
            block_num,
            keep_r,
            is_train,
            scope+'/group_%s' % g,
            data_format)
        results.append(cur_outs)
    results = tf.concat(results, data_format.index('C'), name=scope+'/concat')
    return results


def dense(outs,
          dim,
          scope,
          train=True,
          data_format='NHWC'):
    outs = tf.contrib.layers.fully_connected(
        outs,
        dim,
        activation_fn=None,
        scope=scope+'/dense',
        weights_initializer=tf.truncated_normal_initializer(stddev=0.09))
    return outs


def out_block(outs,
              scope,
              class_num,
              is_train,
              data_format='NHWC'):
    axes = [2, 3] if data_format == 'NCHW' else [1, 2]
    outs = tf.reduce_mean(outs, axes, name=scope+'/pool')
    outs = dense(
        outs=outs,
        dim=class_num,
        scope=scope,
        train=is_train,
        data_format=data_format)
    return outs


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
        outs = channet_conv(
            x=x,
            out_channels=cur_out_num,
            kernel=(3, 3),
            scope='conv_s',
            training=training,
            stride=2,
            act_fn=None,
            data_format=self.data_format)

        cur_out_num *= 2
        cur_outs = dw_block(  # 112 * 112 * 64
            outs=outs,
            num_outs=cur_out_num,
            stride=1,
            scope='conv_1_0',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)
        outs = tf.concat([outs, cur_outs], axis=1, name='add0')

        cur_out_num *= 2
        outs = dw_block(  # 56 * 56 * 128
            outs=outs,
            num_outs=cur_out_num,
            stride=2,
            scope='conv_1_1',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)
        cur_outs = dw_block(  # 56 * 56 * 128
            outs=outs,
            num_outs=cur_out_num,
            stride=1,
            scope='conv_1_2',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)
        outs = tf.concat([outs, cur_outs], axis=1, name='add1')

        cur_out_num *= 2
        outs = dw_block(  # 28 * 28 * 256
            outs=outs,
            num_outs=cur_out_num,
            stride=2,
            scope='conv_1_3',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)
        cur_outs = dw_block(  # 28 * 28 * 256
            outs=outs,
            num_outs=cur_out_num,
            stride=1,
            scope='conv_1_4',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)
        outs = tf.concat([outs, cur_outs], axis=1, name='add2')

        cur_out_num *= 2
        outs = dw_block(  # 14 * 14 * 512
            outs=outs,
            num_outs=cur_out_num,
            stride=2,
            scope='conv_1_5',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)
        cur_outs = simple_group_block(  # 14 * 14 * 512
            outs=outs,
            block_num=self.block_num,
            keep_r=self.dropout_rate,
            is_train=training,
            scope='conv_2_1',
            data_format=self.data_format,
            group=self.group_num)
        outs = tf.add(outs, cur_outs, name='add21')

        outs = conv_group_block(  # 14 * 14 * 512
            outs=outs,
            block_num=self.block_num,
            keep_r=self.dropout_rate,
            is_train=training,
            scope='conv_2_2',
            data_format=self.data_format,
            group=self.group_num)
        cur_outs = conv_group_block(  # 14 * 14 * 512
            outs=outs,
            block_num=self.block_num,
            keep_r=self.dropout_rate,
            is_train=training,
            scope='conv_2_3',
            data_format=self.data_format,
            group=self.group_num)
        outs = tf.add(outs, cur_outs, name='add23')

        cur_out_num *= 2
        outs = dw_block(  # 7 * 7 * 1024
            outs=outs,
            num_outs=cur_out_num,
            stride=2,
            scope='conv_3_0',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)
        outs = dw_block(  # 7 * 7 * 1024
            outs=outs,
            num_outs=cur_out_num,
            stride=1,
            scope='conv_3_1',
            dropout_rate=self.dropout_rate,
            is_train=training,
            data_format=self.data_format)

        outs = out_block(
            outs=outs,
            scope='out',
            class_num=self.classes,
            is_train=training,
            data_format=self.data_format)
        return outs
    

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
