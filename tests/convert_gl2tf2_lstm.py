import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn


def _calc_width(net):
    import numpy as np
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


class TF2Model(tf.keras.Model):

    def __init__(self,
                 **kwargs):
        super(TF2Model, self).__init__(**kwargs)
        # self.rnn = nn.LSTM(
        #     units=100,
        #     dropout=0.2,
        #     name="rnn")
        self.rnn = nn.RNN([nn.LSTMCell(
            units=100,
            dropout=0.2,
            unit_forget_bias=False,
            name="rnn{}".format(i)
        ) for i in range(2)])

    def call(self, x):
        x = self.rnn(x)
        return x


def gl_calc():
    import mxnet as mx

    class GluonModel(mx.gluon.HybridBlock):

        def __init__(self,
                     **kwargs):
            super(GluonModel, self).__init__(**kwargs)

            with self.name_scope():
                self.rnn = mx.gluon.rnn.LSTM(
                    hidden_size=100,
                    num_layers=2,
                    dropout=0.2,
                    input_size=80)

        def hybrid_forward(self, F, x):
            x = self.rnn(x)
            # src_params = self._collect_params_with_prefix()
            # src_param_keys = list(src_params.keys())
            # src_params[src_param_keys[0]]._data[0].asnumpy()
            # dst_params[dst_key]._load_init(mx.nd.array(src_params[src_key].numpy(), ctx), ctx)
            return x

    gl_model = GluonModel()

    # # ctx = mx.cpu()
    # ctx = mx.gpu(0)
    # gl_params = gl_model._collect_params_with_prefix()
    # # gl_w = np.transpose(tf2_w, axes=(3, 2, 0, 1))
    # gl_params['conv.weight']._load_init(mx.nd.array(gl_w, ctx), ctx)
    # # gl_params['conv.bias']._load_init(mx.nd.array(b, ctx), ctx)
    #
    # gl_x = mx.nd.array(x, ctx)
    # gl_y = gl_model(gl_x).asnumpy()

    ctx = mx.gpu(0)
    gl_x = mx.nd.zeros((3, 7, 80), ctx)
    gl_model.initialize(ctx=ctx)
    gl_model(gl_x)
    gl_params = gl_model._collect_params_with_prefix()
    _calc_width(gl_model)

    return gl_model


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    success = True
    for i in range(10):
        tf2_model = TF2Model()
        batch_size = 1
        input_shape = (3, 7, 80)
        tf2_model(tf.random.normal(input_shape))
        dst_param_keys = [v.name for v in tf2_model.weights]
        dst_params = {v.name: v for v in tf2_model.weights}

        gl_calc()

        gl_w = np.random.randn(64, 3, 7, 7).astype(np.float32)
        # tf2_w = np.random.randn(7, 7, 3, 64).astype(np.float32)
        b = np.random.randn(64, ).astype(np.float32)
        x = np.random.randn(10, 3, 224, 256).astype(np.float32)
        assert (b is not None)

        data_format = "channels_last"
        # data_format = "channels_first"
        tf2_use_cuda = True

        if not tf2_use_cuda:
            with tf.device("/cpu:0"):
                tf2_model = TF2Model(data_format=data_format)
        else:
            tf2_model = TF2Model(data_format=data_format)
        input_shape = (1, 224, 256, 3) if data_format == "channels_last" else (1, 3, 224, 256)
        tf2_model.build(input_shape=input_shape)
        tf2_params = {v.name: v for v in tf2_model.weights}
        # print(tf2_params["conv/kernel:0"].shape)
        # tf2_w = np.transpose(gl_w, axes=(2, 3, 1, 0))
        tf2_w = np.transpose(gl_w, axes=(2, 3, 1, 0))
        tf2_params["conv/kernel:0"].assign(tf2_w)
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
