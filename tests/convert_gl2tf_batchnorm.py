import numpy as np
import mxnet as mx
import tensorflow as tf

LENGTH = 64


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.bn = mx.gluon.nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                in_channels=LENGTH,
                use_global_stats=False)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        return x


def batchnorm(x,
              momentum=0.9,
              epsilon=1e-5,
              training=False,
              name=None):
    """
    Batch normalization layer.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    training : bool, or a TensorFlow boolean scalar tensor, default False
      Whether to return the output in training mode or in inference mode.
    name : str, default 'conv2d'
        Layer name.

    Returns:
    -------
    Tensor
        Resulted tensor.
    """
    x = tf.layers.batch_normalization(
        inputs=x,
        axis=1,
        momentum=momentum,
        epsilon=epsilon,
        training=training,
        name=name)
    return x


def tensorflow_model(x):

    x = batchnorm(
        x=x,
        training=False,
        name="bn")
    return x


def main():

    success = True
    for i in range(10):
        g = np.random.randn(LENGTH, ).astype(np.float32)
        b = np.random.randn(LENGTH, ).astype(np.float32)
        m = np.random.randn(LENGTH, ).astype(np.float32)
        v = np.random.randn(LENGTH, ).astype(np.float32)
        b = b - b.min() + 1.0
        v = v - v.min() + 1.0

        IMG_SIZE = 224
        x = np.random.randn(10, LENGTH, IMG_SIZE, IMG_SIZE).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)
        gl_params = gl_model._collect_params_with_prefix()
        gl_params['bn.gamma']._load_init(mx.nd.array(g, ctx), ctx)
        gl_params['bn.beta']._load_init(mx.nd.array(b, ctx), ctx)
        gl_params['bn.running_mean']._load_init(mx.nd.array(m, ctx), ctx)
        gl_params['bn.running_var']._load_init(mx.nd.array(v, ctx), ctx)
        # gl_model.initialize()

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        xx = tf.placeholder(
            dtype=tf.float32,
            shape=(None, LENGTH, IMG_SIZE, IMG_SIZE),
            name='xx')
        tf_model = tensorflow_model(xx)
        tf_params = {v.name: v for v in tf.global_variables()}
        with tf.Session() as sess:
            sess.run(tf_params['bn/gamma:0'].assign(g))
            sess.run(tf_params['bn/beta:0'].assign(b))
            sess.run(tf_params['bn/moving_mean:0'].assign(m))
            sess.run(tf_params['bn/moving_variance:0'].assign(v))

            tf_y = sess.run(tf_model, feed_dict={xx: x})
        tf.reset_default_graph()

        diff = np.abs(gl_y - tf_y)
        dist = np.sum(diff)
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(tf_y)

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
