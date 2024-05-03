import numpy as np
import mxnet as mx
import tensorflow as tf


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.dw_conv = mx.gluon.nn.Conv2D(
                channels=32,
                kernel_size=7,
                strides=2,
                padding=3,
                groups=32,
                use_bias=False,
                in_channels=32)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        return x


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

    if (padding[0] > 0) or (padding[1] > 0):
        x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)])

    if groups == 1:
        x = tf.layers.conv2d(
            inputs=x,
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_first',
            use_bias=use_bias,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
            name=name)
    elif (groups == out_channels) and (out_channels == in_channels):
        kernel = tf.get_variable(
            name=name + '/dw_kernel',
            shape=kernel_size + (in_channels, 1),
            initializer=tf.variance_scaling_initializer(2.0))
        x = tf.nn.depthwise_conv2d(
            input=x,
            filter=kernel,
            strides=(1, 1) + strides,
            padding='VALID',
            rate=(1, 1),
            name=name,
            data_format='NCHW')
        if use_bias:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return x


def tensorflow_model(x):

    x = conv2d(
        x=x,
        in_channels=32,
        out_channels=32,
        kernel_size=7,
        strides=2,
        padding=3,
        groups=32,
        use_bias=False,
        name="dw_conv")
    return x


def main():

    success = True
    for i in range(10):
        # gl_w = np.random.randn(32, 1, 7, 7).astype(np.float32)
        tf_w = np.random.randn(7, 7, 32, 1).astype(np.float32)
        x = np.random.randn(10, 32, 224, 224).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)
        gl_params = gl_model._collect_params_with_prefix()
        gl_w = np.transpose(tf_w, axes=(2, 3, 0, 1))
        gl_params['dw_conv.weight']._load_init(mx.nd.array(gl_w, ctx), ctx)

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        xx = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 32, 224, 224),
            name='xx')
        tf_model = tensorflow_model(xx)
        tf_params = {v.name: v for v in tf.global_variables()}
        with tf.Session() as sess:
            # tf_w = np.transpose(gl_w, axes=(2, 3, 0, 1))
            sess.run(tf_params['dw_conv/dw_kernel:0'].assign(tf_w))

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
