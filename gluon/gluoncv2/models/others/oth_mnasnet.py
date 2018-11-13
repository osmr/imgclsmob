import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

def ConvBlock(channels, kernel_size, strides, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, kernel_size, strides=strides, padding=1, use_bias=False),
            nn.BatchNorm(scale=True),
            nn.Activation('relu')
        )
    return out

def Conv1x1(channels, is_linear=False, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, 1, padding=0, use_bias=False),
            nn.BatchNorm(scale=True)
        )
        if not is_linear:
            out.add(nn.Activation('relu'))
    return out

def DWise(channels, strides, kernel_size=3, **kwargs):
    out = nn.HybridSequential(**kwargs)
    with out.name_scope():
        out.add(
            nn.Conv2D(channels, kernel_size, strides=strides, padding=kernel_size // 2, groups=channels, use_bias=False),
            nn.BatchNorm(scale=True),
            nn.Activation('relu')
        )
    return out

class SepCONV(nn.HybridBlock):
    def __init__(self, inp, output, kernel_size, depth_multiplier=1, with_bn=True, **kwargs):
        super(SepCONV, self).__init__(**kwargs)
        with self.name_scope():
            self.net = nn.HybridSequential()
            cn = int(inp*depth_multiplier)

            if output is None:
                self.net.add(
                    nn.Conv2D(in_channels=inp, channels=cn, groups=inp, kernel_size=kernel_size, strides=(1,1), padding=kernel_size // 2
                        , use_bias=not with_bn)
                )
            else:
                self.net.add(
                    nn.Conv2D(in_channels=inp, channels=cn, groups=inp, kernel_size=kernel_size, strides=(1,1), padding=kernel_size // 2
                        , use_bias=False),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.Conv2D(in_channels=cn, channels=output, kernel_size=(1,1), strides=(1,1)
                        , use_bias=not with_bn)
                )

            self.with_bn = with_bn
            self.act = nn.Activation('relu')
            if with_bn:
                self.bn = nn.BatchNorm()
    def hybrid_forward(self, F ,x):
        x = self.net(x)
        if self.with_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ExpandedConv(nn.HybridBlock):
    def __init__(self, inp, oup, t, strides, kernel=3, same_shape=True, **kwargs):
        super(ExpandedConv, self).__init__(**kwargs)
        # print("inp={}, oup={}, t={}, strides={}, kernel={}".format(inp, oup, t, strides, kernel))

        self.same_shape = same_shape
        self.strides = strides
        with self.name_scope(): 
            self.bottleneck = nn.HybridSequential()
            self.bottleneck.add(
                Conv1x1(inp*t, prefix="expand_"),
                DWise(inp*t, self.strides, kernel, prefix="dwise_"),
                Conv1x1(oup, is_linear=True, prefix="linear_")
            )
    def hybrid_forward(self, F, x):
        out = self.bottleneck(x)
        if self.strides == 1 and self.same_shape:
            out = F.elemwise_add(out, x)
        return out

def ExpandedConvSequence(t, k, inp, oup, repeats, first_strides, **kwargs):
    seq = nn.HybridSequential(**kwargs)
    with seq.name_scope():
        seq.add(ExpandedConv(inp, oup, t, first_strides, k, same_shape=False))
        curr_inp = oup
        for i in range(1, repeats):
            seq.add(ExpandedConv(curr_inp, oup, t, 1))
            curr_inp = oup
    return seq

class MobilenetV2(nn.HybridBlock):
    def __init__(self, num_classes=1000, **kwargs):
        super(MobilenetV2, self).__init__(**kwargs)
        
        self.first_oup = 32
        self.interverted_residual_setting = [
            # t, c,  n, s, k
            [3, 24,  3, 2, 3, "stage2_"],  # -> 56x56
            [3, 40,  3, 2, 5, "stage3_"],  # -> 28x28
            [6, 80,  3, 2, 5, "stage4_1_"],  # -> 14x14
            [6, 96,  2, 1, 3, "stage4_2_"],  # -> 14x14
            [6, 192, 4, 2, 5, "stage5_1_"], # -> 7x7
            [6, 320, 1, 1, 3, "stage5_2_"], # -> 7x7          
        ]
        self.last_channels = 1280

        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(ConvBlock(self.first_oup, 3, 2, prefix="stage1_conv0_"))
            self.features.add(SepCONV(self.first_oup, 16, 3, prefix="stage1_sepconv0_"))
            inp = 16
            for i, (t, c, n, s, k, prefix) in enumerate(self.interverted_residual_setting):
                oup = c
                # print("i={}, (t={}, c={}, n={}, s={}, k={}, prefix={}), inp={}, oup={}".format(i, t, c, n, s, k, prefix, inp, oup))
                self.features.add(ExpandedConvSequence(t, k, inp, oup, n, s, prefix=prefix))
                inp = oup

            self.features.add(Conv1x1(self.last_channels, prefix="stage5_3_"))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.output = nn.Dense(num_classes)
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# if __name__ == '__main__':
#     net = MobilenetV2(1000, prefix="")
#     x = nd.random.normal(shape=(1,3,224,224))
#     net.initialize()
#     y = net(x)
#     net.summary(x)
#
#     # save as symbol
#     #data =mx.sym.var('data')
#     #sym = net(data)
#
#     ## plot network graph
#     #mx.viz.print_summary(sym, shape={'data':(8,3,224,224)})
#     #mx.viz.plot_network(sym,shape={'data':(8,3,224,224)}, node_attrs={'shape':'oval','fixedsize':'fasl==false'}).view()


def oth_mnasnet(model_name=None,
                pretrained=False,
                ctx=None,
                root=None,
                **kwargs):

    net = MobilenetV2(**kwargs)

    # if pretrained:
    #     import mxnet as mx
    #     from .model_store import get_model_file
    #     net.load_parameters(
    #         filename=get_model_file(
    #             model_name=model_name,
    #             local_model_store_dir_path=root),
    #         ctx=mx.cpu())

    return net


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        oth_mnasnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_mnasnet or weight_count == 4308816)


if __name__ == "__main__":
    _test()
