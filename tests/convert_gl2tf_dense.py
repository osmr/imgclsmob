import numpy as np
import mxnet as mx
import tensorflow as tf
# import tensorflow.contrib.slim as slim


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.dense = mx.gluon.nn.Dense(
                units=1000,
                use_bias=False,
                flatten=True,
                in_units=1024)

    def hybrid_forward(self, F, x):
        x = self.dense(x)
        return x


def tensorflow_model(x):

    # x = slim.fully_connected(
    #     inputs=x,
    #     num_outputs=1000,
    #     activation_fn=None,
    #     scope='dense')
    x = tf.layers.dense(
        inputs=x,
        units=1000,
        use_bias=False,
        name="dense")
    return x


def main():

    success = True
    for i in range(10):
        # gl_w = np.random.randn(1000, 1024).astype(np.float32)
        tf_w = np.random.randn(1024, 1000).astype(np.float32)
        # b = np.random.randn(1000, ).astype(np.float32)
        x = np.random.randn(1, 1024).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)
        gl_params = gl_model._collect_params_with_prefix()
        gl_w = np.transpose(tf_w, axes=(1, 0))
        gl_params['dense.weight']._load_init(mx.nd.array(gl_w, ctx), ctx)
        # gl_params['dense.bias']._load_init(mx.nd.array(b, ctx), ctx)

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        xx = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 1024),
            name='xx')
        tf_model = tensorflow_model(xx)
        tf_params = {v.name: v for v in tf.global_variables()}
        with tf.Session() as sess:
            # tf_w = np.transpose(gl_w, axes=(1, 0))
            sess.run(tf_params['dense/kernel:0'].assign(tf_w))
            # sess.run(tf_params['dense/bias:0'].assign(b))
            # sess.run(tf_params['dense/weights:0'].assign(tf_w))
            # sess.run(tf_params['dense/biases:0'].assign(b))

            tf_y = sess.run(tf_model, feed_dict={xx: x})
        tf.reset_default_graph()

        dist = np.sum(np.abs(gl_y - tf_y))
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(tf_y)
            y = np.matmul(gl_w.astype(np.float64), x[0].astype(np.float64))
            # y = np.dot(w, x[0])
            gl_dist = np.sum(np.abs(gl_y - y))
            tf_dist = np.sum(np.abs(tf_y - y))
            print("i={}, gl_dist={}".format(i, gl_dist))
            print("i={}, tf_dist={}".format(i, tf_dist))

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
