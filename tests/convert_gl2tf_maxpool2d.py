import numpy as np
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
import tensorflow as tf


class GluonModel(HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=64,
                kernel_size=7,
                strides=2,
                padding=3,
                use_bias=False,
                in_channels=3)
            self.pool = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=1)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


def tensorflow_model(x):

    padding = 3
    x = tf.pad(x, [[0, 0], [0, 0], [padding, padding], [padding, padding]])
    x = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=7,
        strides=2,
        padding='valid',
        data_format='channels_first',
        use_bias=False,
        name='conv')
    padding = 1
    x = tf.pad(x, [[0, 0], [0, 0], [padding, padding], [padding, padding]])
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=3,
        strides=2,
        padding='valid',
        data_format='channels_first',
        name="pool")
    return x


def main():

    w = np.random.randint(10, size=(64, 3, 7, 7)).astype(np.float32)
    x = np.random.randint(10, size=(1, 3, 224, 224)).astype(np.float32)

    gl_model = GluonModel()

    ctx = mx.cpu()
    gl_params = gl_model._collect_params_with_prefix()
    gl_params['conv.weight']._load_init(mx.nd.array(w, ctx), ctx)

    gl_x = mx.nd.array(x, ctx)
    gl_y = gl_model(gl_x).asnumpy()
    # print(gl_y)

    xx = tf.placeholder(
        dtype=tf.float32,
        shape=(None, 3, 224, 224),
        name='xx')
    tf_model = tensorflow_model(xx)
    tf_params = {v.name: v for v in tf.global_variables()}
    with tf.Session() as sess:
        tf_w = np.transpose(w, axes=(2, 3, 1, 0))
        sess.run(tf_params['conv/kernel:0'].assign(tf_w))

        tf_y = sess.run(tf_model, feed_dict={xx: x})
        # print(tf_y)

    print("dist={}".format(np.sum(gl_y - tf_y)))


if __name__ == '__main__':
    main()
