import mxnet as mx
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from .oth_resnetv1b import resnet50_v1s, resnet101_v1s

__all__ = ['oth_danet_resnet50_citys', 'oth_danet_resnet101_citys']


class PAM_Module(HybridBlock):
    """
    Position attention module from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983>`
    PAM_Module captures long-range spatial contextual information.
    """
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim//8, kernel_size=(1, 1))
        self.key_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim//8, kernel_size=(1, 1))
        self.value_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim, kernel_size=(1, 1))
        self.gamma = self.params.get('gamma', shape=(1,), init=mx.init.Zero())

    def hybrid_forward(self, F, x, **kwargs):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        gamma = kwargs['gamma']
        proj_query = F.reshape(self.query_conv(x), (0, 0, -1))
        proj_key = F.reshape(self.key_conv(x), (0, 0, -1))
        energy = F.batch_dot(proj_query, proj_key, transpose_a=True)
        attention = F.softmax(energy)
        proj_value = F.reshape(self.value_conv(x), (0, 0, -1))
        out = F.batch_dot(proj_value, attention, transpose_b=True)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)

        out = F.broadcast_mul(gamma, out) + x

        return out


class CAM_Module(HybridBlock):
    """
    Channel attention module from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983>`
    CAM_Module explicitly models interdependencies between channels.
    """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = self.params.get('gamma', shape=(1,), init=mx.init.Zero())

    def hybrid_forward(self, F, x, **kwargs):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        gamma = kwargs['gamma']
        proj_query = F.reshape(x, (0, 0, -1))
        proj_key = F.reshape(x, (0, 0, -1))
        energy = F.batch_dot(proj_query, proj_key, transpose_b=True)
        energy_new = F.max(energy, -1, True).broadcast_like(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = F.reshape(x, (0, 0, -1))

        out = F.batch_dot(attention, proj_value)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)

        out = F.broadcast_mul(gamma, out) + x
        return out


def get_backbone(name, **kwargs):
    models = {
        'resnet50': resnet50_v1s,
        'resnet101': resnet101_v1s,
    }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net


class SegBaseModel(HybridBlock):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : Block
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, aux, backbone='resnet50', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        with self.name_scope():
            pretrained = get_backbone(backbone, pretrained=pretrained_base, dilated=True, **kwargs)
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c3, c4

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class DANet(SegBaseModel):
    r"""Dual Attention Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;

    Reference:
        Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang, Hanqing Lu. "Dual Attention
        Network for Scene Segmentation." *CVPR*, 2019
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, ctx=cpu(), pretrained_base=True,
                 height=None, width=None, base_size=520, crop_size=480, dilated=True, in_channels=3, in_size=(480, 480),
                 **kwargs):
        super(DANet, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                                    crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        self.in_size = in_size
        self.classes = nclass
        self.aux = aux
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size

        with self.name_scope():
            self.head = DANetHead(2048, nclass, **kwargs)
            self.head.initialize(ctx=ctx)

        self._up_kwargs = {'height': height, 'width': width}

    def hybrid_forward(self, F, x):
        self._up_kwargs = {'height': x.shape[2], 'width': x.shape[3]}

        c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = F.contrib.BilinearResize2D(x[0], **self._up_kwargs)
        x[1] = F.contrib.BilinearResize2D(x[1], **self._up_kwargs)
        x[2] = F.contrib.BilinearResize2D(x[2], **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])

        # return tuple(outputs)
        return x[0]


class DANetHead(HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.HybridSequential()
        self.conv5a.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels, kernel_size=3, 
                                  padding=1, use_bias=False))
        self.conv5a.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv5a.add(nn.Activation('relu'))

        self.conv5c = nn.HybridSequential()
        self.conv5c.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels, kernel_size=3, 
                                  padding=1, use_bias=False))
        self.conv5c.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv5c.add(nn.Activation('relu'))

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.HybridSequential()
        self.conv51.add(nn.Conv2D(in_channels=inter_channels, channels=inter_channels, kernel_size=3, 
                                  padding=1, use_bias=False))
        self.conv51.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv51.add(nn.Activation('relu'))

        self.conv52 = nn.HybridSequential()
        self.conv52.add(nn.Conv2D(in_channels=inter_channels, channels=inter_channels, kernel_size=3, 
                                  padding=1, use_bias=False))
        self.conv52.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv52.add(nn.Activation('relu'))

        self.conv6 = nn.HybridSequential()
        self.conv6.add(nn.Conv2D(in_channels=512, channels=out_channels, kernel_size=1))
        self.conv6.add(nn.Dropout(0.1))

        self.conv7 = nn.HybridSequential()
        self.conv7.add(nn.Conv2D(in_channels=512, channels=out_channels, kernel_size=1))
        self.conv7.add(nn.Dropout(0.1))

        self.conv8 = nn.HybridSequential()
        self.conv8.add(nn.Conv2D(in_channels=512, channels=out_channels, kernel_size=1))
        self.conv8.add(nn.Dropout(0.1))

    def hybrid_forward(self, F, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)

        return tuple(output)


def get_danet(nclass=19, backbone='resnet50', pretrained=False, ctx=cpu(0), **kwargs):
    """
    DANet model from the paper `"Dual Attention Network for Scene Segmentation" <https://arxiv.org/abs/1809.02983>`
    """
    model = DANet(nclass=nclass, backbone=backbone, ctx=ctx, **kwargs)
    return model


def oth_danet_resnet50_citys(**kwargs):
    return get_danet(nclass=19, backbone='resnet50', **kwargs)


def oth_danet_resnet101_citys(**kwargs):
    return get_danet(nclass=19, backbone='resnet101', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (480, 480)
    pretrained = False

    models = [
        oth_danet_resnet50_citys,
        oth_danet_resnet101_citys,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        x = mx.nd.zeros((1, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_danet_resnet50_citys or weight_count == 47586427)
        assert (model != oth_danet_resnet101_citys or weight_count == 66578555)


if __name__ == "__main__":
    _test()
