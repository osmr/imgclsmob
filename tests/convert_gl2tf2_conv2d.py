import numpy as np
import mxnet as mx
import tensorflow as tf
import tensorflow.keras.layers as nn


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


class TF2Model(tf.keras.Model):

    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(TF2Model, self).__init__(**kwargs)

        padding = 3
        padding = (padding, padding)
        self.paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
        self.conv = nn.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding="valid",
            data_format="channels_last",
            dilation_rate=1,
            use_bias=True,
            name="conv")

    def call(self, x):
        x = tf.pad(x, paddings=self.paddings_tf)
        x = self.conv(x)
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

        tf2_model = TF2Model()
        tf2_model.build(input_shape=(10, 224, 224, 3))
        tf2_params = {v.name: v for v in tf2_model.weights}
        tf2_params["conv/kernel:0"].assign(tf_w)
        tf2_params["conv/bias:0"].assign(b)

        tf2_x = tf.convert_to_tensor(x.transpose((0, 2, 3, 1)))
        tf2_y = tf2_model(tf2_x).numpy()

        tf.reset_default_graph()

        dist = np.sum(np.abs(gl_y - tf2_y))
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(tf_y)

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
