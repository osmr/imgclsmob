from __future__ import division

__all__ = ['oth_mobile_pose_resnet18_v1b', 'oth_mobile_pose_resnet50_v1b', 'oth_mobile_pose_mobilenet1_0',
           'oth_mobile_pose_mobilenetv2_1_0', 'oth_mobile_pose_mobilenetv3_small', 'oth_mobile_pose_mobilenetv3_large']

from mxnet import initializer
from mxnet.gluon import nn
from mxnet.gluon import contrib
from mxnet.gluon.block import HybridBlock
from mxnet.context import cpu
import gluoncv as gcv


class DUC(HybridBlock):
    '''Upsampling layer with pixel shuffle
    '''
    def __init__(self,
                 planes,
                 upscale_factor=2,
                 **kwargs):
        super(DUC, self).__init__(**kwargs)
        self.conv = nn.Conv2D(planes, kernel_size=3, padding=1, use_bias=False)
        self.bn = gcv.nn.BatchNormCudnnOff(gamma_initializer=initializer.One(),
                                    beta_initializer=initializer.Zero())
        self.relu = nn.Activation('relu')
        self.pixel_shuffle = contrib.nn.PixelShuffle2D(upscale_factor)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class MobilePose(HybridBlock):
    """Pose Estimation for Mobile Device"""
    def __init__(self,
                 base_name,
                 base_attrs=('features',),
                 num_joints=17,
                 pretrained_base=False,
                 pretrained_ctx=cpu(),
                 **kwargs):
        super(MobilePose, self).__init__(**kwargs)

        with self.name_scope():
            from gluoncv.model_zoo import get_model
            base_model = get_model(base_name, pretrained=pretrained_base,
                                   ctx=pretrained_ctx)
            self.features = nn.HybridSequential()
            if base_name.startswith('mobilenetv2'):
                self.features.add(base_model.features[:-1])
            elif base_name.startswith('mobilenetv3'):
                self.features.add(base_model.features[:-4])
            elif base_name.startswith('mobilenet'):
                self.features.add(base_model.features[:-2])
            else:
                for layer in base_attrs:
                    self.features.add(getattr(base_model, layer))

            self.upsampling = nn.HybridSequential()
            self.upsampling.add(
                nn.Conv2D(256, 1, 1, 0, use_bias=False),
                DUC(512, 2),
                DUC(256, 2),
                DUC(128, 2),
                nn.Conv2D(num_joints, 1, use_bias=False,
                          weight_initializer=initializer.Normal(0.001)),
            )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        hm = self.upsampling(x)

        return hm


def get_mobile_pose(base_name, ctx=cpu(), pretrained=False,
                    root='~/.mxnet/models', **kwargs):
    net = MobilePose(base_name, **kwargs)

    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file('mobile_pose_%s'%(base_name),
                                           tag=pretrained, root=root), ctx=ctx)

    return net


def oth_mobile_pose_resnet18_v1b(**kwargs):
    return get_mobile_pose('resnet18_v1b', base_attrs=['conv1', 'bn1', 'relu', 'maxpool',
                                                       'layer1', 'layer2', 'layer3', 'layer4'],
                           **kwargs)


def oth_mobile_pose_resnet50_v1b(**kwargs):
    return get_mobile_pose('resnet50_v1b', base_attrs=['conv1', 'bn1', 'relu', 'maxpool',
                                                       'layer1', 'layer2', 'layer3', 'layer4'],
                           **kwargs)


def oth_mobile_pose_mobilenet1_0(**kwargs):
    return get_mobile_pose('mobilenet1.0', base_attrs=['features'], **kwargs)


def oth_mobile_pose_mobilenetv2_1_0(**kwargs):
    return get_mobile_pose('mobilenetv2_1.0', base_attrs=['features'], **kwargs)


def oth_mobile_pose_mobilenetv3_small(**kwargs):
    return get_mobile_pose('mobilenetv3_small', base_attrs=['features'], **kwargs)


def oth_mobile_pose_mobilenetv3_large(**kwargs):
    return get_mobile_pose('mobilenetv3_large', base_attrs=['features'], **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        oth_mobile_pose_resnet18_v1b,
        oth_mobile_pose_resnet50_v1b,
        oth_mobile_pose_mobilenet1_0,
        oth_mobile_pose_mobilenetv2_1_0,
        oth_mobile_pose_mobilenetv3_small,
        oth_mobile_pose_mobilenetv3_large,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        x = mx.nd.zeros((1, 3, 256, 192), ctx=ctx)
        y = net(x)
        # assert (y.shape == (1, 17, 64, 48))

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_mobile_pose_resnet18_v1b or weight_count == 12858208)
        assert (model != oth_mobile_pose_resnet50_v1b or weight_count == 25582944)
        assert (model != oth_mobile_pose_mobilenet1_0 or weight_count == 5019744)
        assert (model != oth_mobile_pose_mobilenetv2_1_0 or weight_count == 4102176)
        assert (model != oth_mobile_pose_mobilenetv3_small or weight_count == 2625088)
        assert (model != oth_mobile_pose_mobilenetv3_large or weight_count == 4768336)


if __name__ == "__main__":
    _test()
