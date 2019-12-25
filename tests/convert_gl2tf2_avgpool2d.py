import math
import numpy as np
import mxnet as mx
import tensorflow as tf
import tensorflow.keras.layers as nn


class GluonModel(mx.gluon.HybridBlock):

    def __init__(self,
                 **kwargs):
        super(GluonModel, self).__init__(**kwargs)

        with self.name_scope():
            self.pool = mx.gluon.nn.AvgPool2D(
                pool_size=3,
                strides=2,
                padding=1,
                ceil_mode=True,
                count_include_pad=True)

    def hybrid_forward(self, F, x):
        x = self.pool(x)
        return x


def is_channels_first(data_format):
    return data_format == "channels_first"


class TF2Model(tf.keras.Model):

    def __init__(self,
                 pool_size=3,
                 strides=2,
                 padding=1,
                 ceil_mode=True,
                 data_format="channels_last",
                 **kwargs):
        super(TF2Model, self).__init__(**kwargs)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.use_stride_pool = (strides[0] > 1) or (strides[1] > 1)
        self.ceil_mode = ceil_mode and self.use_stride_pool
        self.use_pad = (padding[0] > 0) or (padding[1] > 0)

        if self.ceil_mode:
            self.padding = padding
            self.pool_size = pool_size
            self.strides = strides
            self.data_format = data_format
        elif self.use_pad:
            if is_channels_first(data_format):
                self.paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
            else:
                self.paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]

        self.pool = nn.AveragePooling2D(
            pool_size=pool_size,
            strides=1,
            padding="valid",
            data_format=data_format,
            name="pool")
        if self.use_stride_pool:
            self.stride_pool = nn.AveragePooling2D(
                pool_size=1,
                strides=strides,
                padding="valid",
                data_format=data_format,
                name="stride_pool")

    def call(self, x):
        if self.ceil_mode:
            x_shape = x.get_shape().as_list()
            if is_channels_first(self.data_format):
                height = x_shape[2]
                width = x_shape[3]
            else:
                height = x_shape[1]
                width = x_shape[2]
            padding = self.padding
            out_height = float(height + 2 * padding[0] - self.pool_size[0]) / self.strides[0] + 1.0
            out_width = float(width + 2 * padding[1] - self.pool_size[1]) / self.strides[1] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)
            if (padding[0] > 0) or (padding[1] > 0):
                if is_channels_first(self.data_format):
                    paddings_tf = [[0, 0], [0, 0], list(padding), list(padding)]
                else:
                    paddings_tf = [[0, 0], list(padding), list(padding), [0, 0]]
                x = tf.pad(x, paddings=paddings_tf)
        elif self.use_pad:
            x = tf.pad(x, paddings=self.paddings_tf)

        x = self.pool(x)
        if self.use_stride_pool:
            x = self.stride_pool(x)
        return x


def main():

    success = True
    for i in range(10):
        x = np.random.randn(12, 10, 224, 224).astype(np.float32)

        gl_model = GluonModel()

        # ctx = mx.cpu()
        ctx = mx.gpu(0)

        gl_x = mx.nd.array(x, ctx)
        gl_y = gl_model(gl_x).asnumpy()

        data_format = "channels_last"
        # data_format = "channels_first"
        tf2_use_cuda = True

        if not tf2_use_cuda:
            with tf.device("/cpu:0"):
                tf2_model = TF2Model(data_format=data_format)
        else:
            tf2_model = TF2Model(data_format=data_format)
        input_shape = (1, 224, 224, 10) if data_format == "channels_last" else (1, 10, 224, 224)
        tf2_model.build(input_shape=input_shape)
        # tf2_params = {v.name: v for v in tf2_model.weights}

        tf2_x = x.transpose((0, 2, 3, 1)) if data_format == "channels_last" else x
        tf2_x = tf.convert_to_tensor(tf2_x)
        tf2_y = tf2_model(tf2_x).numpy()
        if data_format == "channels_last":
            tf2_y = tf2_y.transpose((0, 3, 1, 2))

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
