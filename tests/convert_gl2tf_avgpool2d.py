# import math
import numpy as np
import mxnet as mx
import tensorflow as tf


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.pool = mx.gluon.nn.AvgPool2D(
                pool_size=2,
                strides=2,
                padding=0)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        return x


# def avgpool2d(x,
#               pool_size,
#               strides,
#               padding=0,
#               ceil_mode=False,
#               name=None):
#     """
#     Average pooling operation for two dimensional (spatial) data.
#
#     Parameters:
#     ----------
#     x : Tensor
#         Input tensor.
#     pool_size : int or tuple/list of 2 int
#         Size of the max pooling windows.
#     strides : int or tuple/list of 2 int
#         Strides of the pooling.
#     padding : int or tuple/list of 2 int, default 0
#         Padding value for convolution layer.
#     ceil_mode : bool, default False
#         When `True`, will use ceil instead of floor to compute the output shape.
#     name : str, default 'conv2d'
#         Layer name.
#
#     Returns:
#     -------
#     Tensor
#         Resulted tensor.
#     """
#     if isinstance(padding, int):
#         padding = (padding, padding)
#
#     if ceil_mode:
#         height = x.shape[2]
#         out_height = float(height + 2 * padding[0] - pool_size[0]) / strides[0] + 1.0
#         if math.ceil(out_height) > math.floor(out_height):
#             padding[0] += 1
#         width = x.shape[3]
#         out_width = float(width + 2 * padding[1] - pool_size[1]) / strides[1] + 1.0
#         if math.ceil(out_width) > math.floor(out_width):
#             padding[1] += 1
#
#     if (padding[0] > 0) or (padding[1] > 0):
#         x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)], mode="REFLECT")
#
#     x = tf.layers.average_pooling2d(
#         inputs=x,
#         pool_size=pool_size,
#         strides=strides,
#         padding='valid',
#         data_format='channels_first',
#         name=name)
#     return x


def tensorflow_model(x):

    x = tf.layers.average_pooling2d(
        inputs=x,
        pool_size=2,
        strides=2,
        padding='valid',
        data_format='channels_first',
        name="pool")
    # x = avgpool2d(
    #     x=x,
    #     pool_size=2,
    #     strides=2,
    #     padding=1,
    #     ceil_mode=False,
    #     name="pool")
    return x


def main():

    success = True
    for i in range(10):
        x = np.random.randn(10, 10, 224, 224).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        xx = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 10, 224, 224),
            name='xx')
        tf_model = tensorflow_model(xx)
        with tf.Session() as sess:
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
