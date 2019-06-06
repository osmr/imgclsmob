
from mxnet.gluon import nn
from math import ceil


def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True, momentum=0.99, epsilon=1e-3))
    if active:
        out.add(nn.Swish())


class MBConv(nn.HybridBlock):
    def __init__(self, in_channels, channels, t, kernel, stride, **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()
            _add_conv(self.out, in_channels * t, active=True)
            _add_conv(self.out, in_channels * t, kernel=kernel, stride=stride,
                      pad=int((kernel-1)/2), num_group=in_channels * t, active=True)
            _add_conv(self.out, channels, active=False)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class EfficientNet(nn.HybridBlock):
    r"""
    Parameters
    ----------
    alpha : float, default 1.0
        The depth multiplier for controling the model size. The actual number of layers on each channel_size level
        is equal to the original number of layers multiplied by alpha.
    beta : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by beta.
    dropout_rate : float, default 0.0
        Dropout probability for the final features layer.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, alpha=1.0, beta=1.0, dropout_rate=0.0, classes=1000, **kwargs):
        super(EfficientNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                # stem conv
                _add_conv(self.features, int(32 * beta), kernel=3, stride=2, pad=1)

                # base model settings
                repeats = [1, 2, 2, 3, 3, 4, 1]
                channels_num = [16, 24, 40, 80, 112, 192, 320]
                kernels_num = [3, 3, 5, 3, 5, 5, 3]
                t_num = [1, 6, 6, 6, 6, 6, 6]
                strides_first = [1, 2, 2, 1, 2, 2, 1]

                # determine params of MBConv layers
                in_channels_group = []
                for rep, ch_num in zip([1] + repeats[:-1], [32] + channels_num[:-1]):
                    in_channels_group += [int(ch_num * beta)] * int(ceil(alpha * rep))
                channels_group, kernels, ts, strides = [], [], [], []
                for rep, ch, kernel, t, s in zip(repeats, channels_num, kernels_num, t_num, strides_first):
                    rep = int(ceil(alpha * rep))
                    channels_group += [int(ch * beta)] * rep
                    kernels += [kernel] * rep
                    ts += [t] * rep
                    strides += [s] + [1] * (rep - 1)

                # add MBConv layers
                for in_c, c, t, k, s in zip(in_channels_group, channels_group, ts, kernels, strides):
                    self.features.add(MBConv(in_channels=in_c, channels=c, t=t, kernel=k, stride=s))

                # head layers
                last_channels = int(1280 * beta) if beta > 1.0 else 1280
                _add_conv(self.features, last_channels)
                self.features.add(nn.GlobalAvgPool2D())

            # features dropout
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

            # output layer
            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'),
                    nn.Flatten()
                )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.output(x)
        return x


def get_efficientnet(model_name, pretrained=False, in_size=None):
    params_dict = { # (width_coefficient, depth_coefficient, input_resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    }
    width_coeff, depth_coeff, input_resolution, dropout_rate = params_dict[model_name]
    model = EfficientNet(alpha=depth_coeff, beta=width_coeff, dropout_rate=dropout_rate)
    assert (in_size[0] == input_resolution)
    return model


def oth_efficientnet_b0(**kwargs):
    return get_efficientnet(model_name="efficientnet-b0", **kwargs)


def oth_efficientnet_b1(**kwargs):
    return get_efficientnet(model_name="efficientnet-b1", **kwargs)


def oth_efficientnet_b2(**kwargs):
    return get_efficientnet(model_name="efficientnet-b2", **kwargs)


def oth_efficientnet_b3(**kwargs):
    return get_efficientnet(model_name="efficientnet-b3", **kwargs)


def oth_efficientnet_b4(**kwargs):
    return get_efficientnet(model_name="efficientnet-b4", **kwargs)


def oth_efficientnet_b5(**kwargs):
    return get_efficientnet(model_name="efficientnet-b5", **kwargs)


def oth_efficientnet_b6(**kwargs):
    return get_efficientnet(model_name="efficientnet-b6", **kwargs)


def oth_efficientnet_b7(**kwargs):
    return get_efficientnet(model_name="efficientnet-b7", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        # (oth_efficientnet_b0, 224),
        # (oth_efficientnet_b1, 240),
        # (oth_efficientnet_b2, 260),
        # (oth_efficientnet_b3, 300),
        # (oth_efficientnet_b4, 380),
        # (oth_efficientnet_b5, 456),
        # (oth_efficientnet_b6, 528),
        (oth_efficientnet_b7, 600),
    ]

    for model, in_size_x in models:

        net = model(pretrained=pretrained, in_size=(in_size_x, in_size_x))

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()

        x = mx.nd.zeros((1, 3, in_size_x, in_size_x), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_efficientnet_b0 or weight_count == 4652096)
        assert (model != oth_efficientnet_b1 or weight_count == 5869408)
        assert (model != oth_efficientnet_b2 or weight_count == 6920275)
        assert (model != oth_efficientnet_b3 or weight_count == 9089226)
        assert (model != oth_efficientnet_b4 or weight_count == 14333294)
        assert (model != oth_efficientnet_b5 or weight_count == 21448176)
        assert (model != oth_efficientnet_b6 or weight_count == 30560505)
        assert (model != oth_efficientnet_b7 or weight_count == 44965952)


if __name__ == "__main__":
    _test()
