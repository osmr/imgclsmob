"""
    Common routines for models in Gluon.
"""

__all__ = ['depthwise_conv', 'channel_shuffle', 'ImageNetModel']

from abc import abstractmethod
import tensorflow as tf
from tensorpack import ModelDesc, regularize_cost, layer_register
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger


@layer_register(log_shape=True)
def depthwise_conv(input,
                   channels,
                   kernel_size,
                   strides=1,
                   padding='SAME',
                   W_init=None,
                   activation=tf.identity):
    assert channels == input.get_shape().as_list()[1]

    if type(kernel_size) is not list:
        kernel_size = [kernel_size, kernel_size]
    channel_multiplier = 1
    filter_shape = kernel_size + [channels, channel_multiplier]

    if W_init is None:
        W_init = tf.variance_scaling_initializer(2.0)

    W = tf.get_variable(
        name='W',
        shape=filter_shape,
        initializer=W_init)
    conv = tf.nn.depthwise_conv2d(
        input=input,
        filter=W,
        strides=[1, 1, strides, strides],
        padding=padding,
        data_format='NCHW')
    activ = activation(
        input=conv,
        name='output')
    return activ


@under_name_scope()
def channel_shuffle(x,
                    groups):

    x_shape = x.get_shape().as_list()
    channels = x_shape[1]
    assert channels % groups == 0, channels
    x = tf.reshape(x, [-1, channels // groups, groups] + x_shape[-2:])
    x = tf.transpose(x, [0, 2, 1, 3, 4])
    x = tf.reshape(x, [-1, channels] + x_shape[-2:])
    return x


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

    def build_graph(self,
                    image,
                    label):

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
    def get_logits(self,
                   image):
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

    def image_preprocess(self,
                         image):

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
    def compute_loss_and_error(logits,
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
