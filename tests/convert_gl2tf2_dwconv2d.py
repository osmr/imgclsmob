import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn

channels = 12


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


class TF2Model(tf.keras.Model):

    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(TF2Model, self).__init__(**kwargs)
        self.conv = nn.DepthwiseConv2D(
            # filters=channels,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
            data_format=data_format,
            dilation_rate=1,
            use_bias=False,
            name="conv")

    def call(self, x):
        x = self.conv(x)
        return x


def gl_calc(gl_w, x):
    import mxnet as mx

    class GluonModel(mx.gluon.HybridBlock):

        def __init__(self,
                     **kwargs):
            super(GluonModel, self).__init__(**kwargs)

            with self.name_scope():
                self.conv = mx.gluon.nn.Conv2D(
                    channels=channels,
                    kernel_size=(7, 7),
                    strides=2,
                    padding=(3, 3),
                    groups=channels,
                    use_bias=False,
                    in_channels=channels)

        def hybrid_forward(self, F, x):
            x = self.conv(x)
            return x

    gl_model = GluonModel()

    # ctx = mx.cpu()
    ctx = mx.gpu(0)
    gl_params = gl_model._collect_params_with_prefix()
    # gl_w = np.transpose(tf2_w, axes=(3, 2, 0, 1))
    gl_params['conv.weight']._load_init(mx.nd.array(gl_w, ctx), ctx)
    # gl_params['conv.bias']._load_init(mx.nd.array(b, ctx), ctx)

    gl_x = mx.nd.array(x, ctx)
    gl_y = gl_model(gl_x).asnumpy()

    return gl_y


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    success = True
    for i in range(10):
        gl_w = np.random.randn(channels, 1, 7, 7).astype(np.float32)
        # tf2_w = np.random.randn(7, 7, 1, channels).astype(np.float32)
        b = np.random.randn(channels, ).astype(np.float32)
        x = np.random.randn(10, channels, 224, 256).astype(np.float32)
        assert (b is not None)

        data_format = "channels_last"
        # data_format = "channels_first"
        tf2_use_cuda = True

        if not tf2_use_cuda:
            with tf.device("/cpu:0"):
                tf2_model = TF2Model(data_format=data_format)
        else:
            tf2_model = TF2Model(data_format=data_format)
        input_shape = (1, 224, 256, channels) if data_format == "channels_last" else (1, channels, 224, 256)
        tf2_model.build(input_shape=input_shape)
        tf2_params = {v.name: v for v in tf2_model.weights}
        # print(tf2_params["conv/kernel:0"].shape)
        # tf2_w = np.transpose(gl_w, axes=(2, 3, 1, 0))
        tf2_w = np.transpose(gl_w, axes=(2, 3, 0, 1))
        tf2_params["conv/depthwise_kernel:0"].assign(tf2_w)
        # tf2_params["conv/bias:0"].assign(b)

        tf2_x = x.transpose((0, 2, 3, 1)) if data_format == "channels_last" else x
        tf2_x = tf.convert_to_tensor(tf2_x)
        tf2_y = tf2_model(tf2_x).numpy()
        if data_format == "channels_last":
            tf2_y = tf2_y.transpose((0, 3, 1, 2))

        gl_y = gl_calc(gl_w, x)

        dist = np.sum(np.abs(gl_y - tf2_y))
        if dist > 1e-5:
            success = False
            print("i={}, dist={}".format(i, dist))
            # print(gl_y)
            # print(tf_y)

    if success:
        print("All ok.")


if __name__ == "__main__":
    main()
