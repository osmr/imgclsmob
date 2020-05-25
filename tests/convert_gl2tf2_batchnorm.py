import numpy as np
import mxnet as mx
import tensorflow as tf
import tensorflow.keras.layers as nn

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


def is_channels_first(data_format):
    """
    Is tested data format channels first.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    bool
        A flag.
    """
    return data_format == "channels_first"


def get_channel_axis(data_format):
    """
    Get channel axis.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns
    -------
    int
        Channel axis.
    """
    return 1 if is_channels_first(data_format) else -1


class BatchNorm(nn.BatchNormalization):
    """
    MXNet/Gluon-like batch normalization.

    Parameters:
    ----------
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(BatchNorm, self).__init__(
            axis=get_channel_axis(data_format),
            momentum=momentum,
            epsilon=epsilon,
            **kwargs)


class TF2Model(tf.keras.Model):

    def __init__(self,
                 bn_eps=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(TF2Model, self).__init__(**kwargs)
        self.bn = BatchNorm(
            epsilon=bn_eps,
            data_format=data_format,
            name="bn")

    def call(self, x, training=None):
        x = self.bn(x, training=training)
        return x


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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

        data_format = "channels_last"
        # data_format = "channels_first"
        tf2_use_cuda = True

        if not tf2_use_cuda:
            with tf.device("/cpu:0"):
                tf2_model = TF2Model(data_format=data_format)
        else:
            tf2_model = TF2Model(data_format=data_format)
        input_shape = (1, IMG_SIZE, IMG_SIZE, LENGTH) if data_format == "channels_last" else\
            (1, LENGTH, IMG_SIZE, IMG_SIZE)
        tf2_model.build(input_shape=input_shape)

        tf2_params = {v.name: v for v in tf2_model.weights}
        tf2_params["bn/gamma:0"].assign(g)
        tf2_params["bn/beta:0"].assign(b)
        tf2_params["bn/moving_mean:0"].assign(m)
        tf2_params["bn/moving_variance:0"].assign(v)

        tf2_x = x.transpose((0, 2, 3, 1)) if data_format == "channels_last" else x
        tf2_x = tf.convert_to_tensor(tf2_x)
        tf2_y = tf2_model(tf2_x).numpy()
        if data_format == "channels_last":
            tf2_y = tf2_y.transpose((0, 3, 1, 2))

        diff = np.abs(gl_y - tf2_y)
        dist = diff.mean()
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(tf_y)

    if success:
        print("All ok.")


if __name__ == '__main__':
    main()
