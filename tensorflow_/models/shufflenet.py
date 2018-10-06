
__all__ = ['ShufflenetModel']


import math
from abc import abstractmethod

import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger


class ImageNetModel(ModelDesc):

    def __init__(self,
                 **kwargs):
        super(ImageNetModel, self).__init__(**kwargs)

        self.image_shape = 224

        """
        uint8 instead of float32 is used as input type to reduce copy overhead.
        It might hurt the performance a liiiitle bit.
        The pretrained models were trained with float32.
        """
        self.image_dtype = tf.uint8

        """
        Either 'NCHW' or 'NHWC'
        """
        self.data_format = 'NCHW'

        """
        Whether the image is BGR or RGB. If using DataFlow, then it should be BGR.
        """
        self.image_bgr = True

        self.weight_decay = 1e-4

        """
        To apply on normalization parameters, use '.*/W|.*/gamma|.*/beta'
        """
        self.weight_decay_pattern = '.*/W'

        """
        Scale the loss, for whatever reasons (e.g., gradient averaging, fp16 training, etc)
        """
        self.loss_scale = 1.0

        """
        Label smoothing (See tf.losses.softmax_cross_entropy)
        """
        self.label_smoothing = 0.0

    def inputs(self):
        return [tf.placeholder(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = self.image_preprocess(image)
        assert self.data_format in ['NCHW', 'NHWC']
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)
        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing)

        if self.weight_decay > 0:
            wd_loss = regularize_cost(self.weight_decay_pattern,
                                      tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            total_cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            total_cost = tf.identity(loss, name='cost')
            add_moving_summary(total_cost)

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of ``self.input_shape`` in ``self.data_format``

        Returns:
            Nx#class logits
        """

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if self.image_bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32) * 255.
            image_std = tf.constant(std, dtype=tf.float32) * 255.
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(
            logits,
            label,
            label_smoothing=0.0):

        if label_smoothing == 0.0:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        else:
            nclass = logits.shape[-1]
            loss = tf.losses.softmax_cross_entropy(
                tf.one_hot(label, nclass),
                logits, label_smoothing=label_smoothing)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss


@layer_register(log_shape=True)
def DepthConv(x,
              out_channel,
              kernel_shape,
              padding='SAME',
              stride=1,
              W_init=None,
              activation=tf.identity):

    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0, (out_channel, in_channel)
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.variance_scaling_initializer(2.0)
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return activation(conv, name='output')


@under_name_scope()
def channel_shuffle(xl,
                    group):

    in_shape = xl.get_shape().as_list()
    in_channel = in_shape[1]
    assert in_channel % group == 0, in_channel
    xl = tf.reshape(xl, [-1, in_channel // group, group] + in_shape[-2:])
    xl = tf.transpose(xl, [0, 2, 1, 3, 4])
    xl = tf.reshape(xl, [-1, in_channel] + in_shape[-2:])
    return xl


@layer_register()
def shufflenet_unit(xl,
                    out_channel,
                    group,
                    stride):

    in_shape = xl.get_shape().as_list()
    in_channel = in_shape[1]
    shortcut = xl

    # "We do not apply group convolution on the first pointwise layer
    #  because the number of input channels is relatively small."
    first_split = group if in_channel > 24 else 1
    xl = Conv2D('conv1', xl, out_channel // 4, 1, split=first_split, activation=BNReLU)
    xl = channel_shuffle(xl, group)
    xl = DepthConv('dconv', xl, out_channel // 4, 3, stride=stride)
    xl = BatchNorm('dconv_bn', xl)

    xl = Conv2D('conv2', xl,
                out_channel if stride == 1 else out_channel - in_channel,
                1, split=group)
    xl = BatchNorm('conv2_bn', xl)
    if stride == 1:     # unit (b)
        output = tf.nn.relu(shortcut + xl)
    else:   # unit (c)
        shortcut = AvgPooling('avgpool', shortcut, 3, 2, padding='SAME')
        output = tf.concat([shortcut, tf.nn.relu(xl)], axis=1)
    return output


@layer_register()
def shufflenet_unit_v2(xl,
                       out_channel,
                       stride):

    if stride == 1:
        shortcut, xl = tf.split(xl, 2, axis=1)
    else:
        shortcut, xl = xl, xl
    shortcut_channel = int(shortcut.shape[1])

    xl = Conv2D('conv1', xl, out_channel // 2, 1, activation=BNReLU)
    xl = DepthConv('dconv', xl, out_channel // 2, 3, stride=stride)
    xl = BatchNorm('dconv_bn', xl)
    xl = Conv2D('conv2', xl, out_channel - shortcut_channel, 1, activation=BNReLU)

    if stride == 2:
        shortcut = DepthConv('shortcut_dconv', shortcut, shortcut_channel, 3, stride=2)
        shortcut = BatchNorm('shortcut_dconv_bn', shortcut)
        shortcut = Conv2D('shortcut_conv', shortcut, shortcut_channel, 1, activation=BNReLU)
    output = tf.concat([shortcut, xl], axis=1)
    output = channel_shuffle(output, 2)
    return output


@layer_register(log_shape=True)
def shufflenet_stage(input, channel, num_blocks, group, v2):
    xl = input
    for i in range(num_blocks):
        name = 'block{}'.format(i)
        if v2:
            xl = shufflenet_unit_v2(name, xl, channel, 2 if i == 0 else 1)
        else:
            xl = shufflenet_unit(name, xl, channel, group, 2 if i == 0 else 1)
    return xl


class ShufflenetModel(ImageNetModel):

    def __init__(self,
                 v2,
                 ratio,
                 group,
                 **kwargs):
        super(ShufflenetModel, self).__init__(**kwargs)

        self.v2 = v2
        self.ratio = ratio
        self.group = group
        self.weight_decay = 4e-5

    def get_logits(self, image):

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False):

            group = self.group
            if not self.v2:
                # Copied from the paper
                channels = {
                    3: [240, 480, 960],
                    4: [272, 544, 1088],
                    8: [384, 768, 1536]
                }
                mul = group * 4  # #chan has to be a multiple of this number
                channels = [int(math.ceil(x * self.ratio / mul) * mul)
                            for x in channels[group]]
                # The first channel must be a multiple of group
                first_chan = int(math.ceil(24 * self.ratio / group) * group)
            else:
                # Copied from the paper
                channels = {
                    0.5: [48, 96, 192],
                    1.: [116, 232, 464]
                }[self.ratio]
                first_chan = 24

            logger.info("#Channels: " + str([first_chan] + channels))

            xl = Conv2D('conv1', image, first_chan, 3, strides=2, activation=BNReLU)
            xl = MaxPooling('pool1', xl, 3, 2, padding='SAME')

            xl = shufflenet_stage('stage2', xl, channels[0], 4, group, self.v2)
            xl = shufflenet_stage('stage3', xl, channels[1], 8, group, self.v2)
            xl = shufflenet_stage('stage4', xl, channels[2], 4, group, self.v2)

            if self.v2:
                xl = Conv2D('conv5', xl, 1024, 1, activation=BNReLU)

            xl = GlobalAvgPooling('gap', xl)
            logits = FullyConnected('linear', xl, 1000)
            return logits


def shufflenetv2_wd2(**kwargs):
    return ShufflenetModel(
        v2=True,
        ratio=0.5,
        group=8,
        **kwargs)


if __name__ == '__main__':
    pass
