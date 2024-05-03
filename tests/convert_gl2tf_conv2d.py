import numpy as np
import mxnet as mx
import tensorflow as tf


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.conv = mx.gluon.nn.Conv2D(
                channels=64,
                kernel_size=7,
                strides=2,
                padding=3,
                use_bias=True,
                in_channels=3)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x


# def tensorflow_model(x):
#
#     padding = 3
#     x = tf.pad(x, [[0, 0], [0, 0], [padding, padding], [padding, padding]])
#     x = tf.layers.conv2d(
#         inputs=x,
#         filters=64,
#         kernel_size=7,
#         strides=2,
#         padding='valid',
#         data_format='channels_first',
#         use_bias=False,
#         name='conv')
#     return x


def conv2d(x,
           in_channels,
           out_channels,
           kernel_size,
           strides=1,
           padding=0,
           groups=1,
           use_bias=True,
           name="conv2d"):
    """
    Convolution 2D layer wrapper.

    Parameters
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
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    name : str, default 'conv2d'
        Layer name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(padding, int):
        padding = (padding, padding)

    if groups != 1:
        raise NotImplementedError

    if (padding[0] > 0) or (padding[1] > 0):
        x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)])
    x = tf.layers.conv2d(
        inputs=x,
        filters=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding='valid',
        data_format='channels_first',
        use_bias=use_bias,
        name=name)
    return x


def tensorflow_model(x):

    x = conv2d(
        x=x,
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        strides=2,
        padding=3,
        use_bias=True,
        name="conv")
    return x


def main():

    success = True
    for i in range(10):
        # gl_w = np.random.randn(64, 3, 7, 7).astype(np.float32)
        tf_w = np.random.randn(7, 7, 3, 64).astype(np.float32)
        b = np.random.randn(64, ).astype(np.float32)
        x = np.random.randn(10, 3, 224, 224).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)
        gl_params = gl_model._collect_params_with_prefix()
        gl_w = np.transpose(tf_w, axes=(3, 2, 0, 1))
        gl_params['conv.weight']._load_init(mx.nd.array(gl_w, ctx), ctx)
        gl_params['conv.bias']._load_init(mx.nd.array(b, ctx), ctx)

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        xx = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224),
            name='xx')
        tf_model = tensorflow_model(xx)
        tf_params = {v.name: v for v in tf.global_variables()}
        with tf.Session() as sess:
            # tf_w = np.transpose(gl_w, axes=(2, 3, 1, 0))
            sess.run(tf_params['conv/kernel:0'].assign(tf_w))
            sess.run(tf_params['conv/bias:0'].assign(b))

            tf_y = sess.run(tf_model, feed_dict={xx: x})
        tf.reset_default_graph()

        dist = np.sum(np.abs(gl_y - tf_y))
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(tf_y)

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
