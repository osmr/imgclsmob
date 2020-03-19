"""
    CenterNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Objects as Points,' https://arxiv.org/abs/1904.07850.
"""

__all__ = ['centernet_resnet18_voc', 'centernet_resnet18_coco', 'centernet_resnet50b_voc', 'centernet_resnet50b_coco',
           'centernet_resnet101b_voc', 'centernet_resnet101b_coco']

import os
import math
import warnings
from collections import OrderedDict
import mxnet as mx
from mxnet.gluon import nn
from mxnet import autograd
from mxnet import gluon


class BilinearUpSampleInitializer(mx.init.Initializer):
    """
    Initializes weights as bilinear upsampling kernel.
    """
    def __init__(self):
        super(BilinearUpSampleInitializer, self).__init__()

    def _init_weight(self, _, arr):
        mx.nd.random.normal(0, 0.01, arr.shape, out=arr)
        f = math.ceil(arr.shape[2] / 2)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        for i in range(arr.shape[2]):
            for j in range(arr.shape[3]):
                arr[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, arr.shape[0]):
            arr[c, 0, :, :] = arr[0, 0, :, :]


class DeconvResnet(nn.HybridBlock):
    """
    Deconvolutional ResNet.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    deconv_filters : list of int
        Number of filters for deconv layers.
    deconv_kernels : list of int
        Kernel sizes for deconv layers.
    pretrained_backbone : bool
        Whether load pretrained base network.
    norm_layer : mxnet.gluon.nn.HybridBlock
        Type of Norm layers, can be BatchNorm, SyncBatchNorm, GroupNorm, etc.
    norm_kwargs : dict
        Additional kwargs for `norm_layer`.
    """
    def __init__(self,
                 base_network='resnet18_v1b',
                 deconv_filters=(256, 128, 64),
                 deconv_kernels=(4, 4, 4),
                 pretrained_backbone=True,
                 in_channels=3,
                 **kwargs):
        super(DeconvResnet, self).__init__(**kwargs)
        assert 'resnet' in base_network
        from gluoncv.model_zoo import get_model
        net = get_model(base_network, pretrained=pretrained_backbone)
        feat = nn.HybridSequential()
        feat.add(*[net.conv1,
                   net.bn1,
                   net.relu,
                   net.maxpool,
                   net.layer1,
                   net.layer2,
                   net.layer3,
                   net.layer4])
        self.base_network = feat
        with self.name_scope():
            self.deconv = self._make_deconv_layer(deconv_filters, deconv_kernels)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get the deconv configs using presets"""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError('Unsupported deconvolution kernel: {}'.format(deconv_kernel))

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):
        """Make deconv layers using the configs"""
        assert len(num_kernels) == len(num_filters), \
            'Deconv filters and kernels number mismatch: {} vs. {}'.format(
                len(num_filters), len(num_kernels))

        layers = nn.HybridSequential('deconv_')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.base_network.initialize()
        in_planes = self.base_network(mx.nd.zeros((1, 3, 256, 256))).shape[1]
        for planes, k in zip(num_filters, num_kernels):
            kernel, padding, output_padding = self._get_deconv_cfg(k)
            layers.add(nn.Conv2D(channels=planes,
                                 kernel_size=3,
                                 strides=1,
                                 padding=1,
                                 in_channels=in_planes))
            layers.add(nn.BatchNorm())
            layers.add(nn.Activation('relu'))
            layers.add(nn.Conv2DTranspose(channels=planes,
                                          kernel_size=kernel,
                                          strides=2,
                                          padding=padding,
                                          output_padding=output_padding,
                                          use_bias=False,
                                          in_channels=planes,
                                          weight_initializer=BilinearUpSampleInitializer()))
            layers.add(nn.BatchNorm())
            layers.add(nn.Activation('relu'))
            in_planes = planes

        return layers

    def hybrid_forward(self, F, x):
        y = self.base_network(x)
        out = self.deconv(y)
        return out


def get_deconv_resnet(base_network,
                      pretrained=False,
                      **kwargs):
    """
    Get resnet with deconv layers.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    pretrained : bool
        Whether load pretrained base network.
    ctx : mxnet.Context
        mx.cpu() or mx.gpu()
    pretrained : type
        Description of parameter `pretrained`.
    Returns
    -------
    get_deconv_resnet(base_network, pretrained=False,
        Description of returned object.
    """
    net = DeconvResnet(
        base_network=base_network,
        pretrained_backbone=pretrained,
        **kwargs)
    return net


def resnet18_v1b_deconv(**kwargs):
    """Resnet18 v1b model with deconv layers.

    Returns
    -------
    HybridBlock
        A Resnet18 v1b model with deconv layers.

    """
    return get_deconv_resnet('resnet18_v1b', **kwargs)


def resnet50_v1b_deconv(**kwargs):
    """Resnet50 v1b model with deconv layers.

    Returns
    -------
    HybridBlock
        A Resnet50 v1b model with deconv layers.

    """
    return get_deconv_resnet('resnet50_v1b', **kwargs)


def resnet101_v1b_deconv(**kwargs):
    """Resnet101 v1b model with deconv layers.

    Returns
    -------
    HybridBlock
        A Resnet101 v1b model with deconv layers.

    """
    return get_deconv_resnet('resnet101_v1b', **kwargs)

# ---------------------------------------------------------------------------------------------------------------------


class CenterNetDecoder(gluon.HybridBlock):
    """
    Decorder for centernet.

    Parameters
    ----------
    topk : int
        Only keep `topk` results.
    scale : float, default is 4.0
        Downsampling scale for the network.

    """
    def __init__(self,
                 topk=100,
                 scale=4.0):
        super(CenterNetDecoder, self).__init__()
        self._topk = topk
        self._scale = scale

    def hybrid_forward(self, F, x, wh, reg):
        _, _, out_h, out_w = x.shape_array().split(num_outputs=4, axis=0)
        scores, indices = x.reshape((0, -1)).topk(k=self._topk, ret_typ="both")
        indices = F.cast(indices, "int64")
        topk_classes = F.cast(F.broadcast_div(indices, (out_h * out_w)), "float32")
        topk_indices = F.broadcast_mod(indices, (out_h * out_w))
        topk_ys = F.broadcast_div(topk_indices, out_w)
        topk_xs = F.broadcast_mod(topk_indices, out_w)
        center = reg.transpose((0, 2, 3, 1)).reshape((0, -1, 2))
        wh = wh.transpose((0, 2, 3, 1)).reshape((0, -1, 2))
        batch_indices = F.cast(F.arange(256).slice_like(
            center, axes=(0,)).expand_dims(-1).tile(reps=(1, self._topk)), "int64")
        reg_xs_indices = F.zeros_like(batch_indices, dtype="int64")
        reg_ys_indices = F.ones_like(batch_indices, dtype="int64")
        reg_xs = F.concat(batch_indices, topk_indices, reg_xs_indices, dim=0).reshape((3, -1))
        reg_ys = F.concat(batch_indices, topk_indices, reg_ys_indices, dim=0).reshape((3, -1))
        xs = F.cast(F.gather_nd(center, reg_xs).reshape((-1, self._topk)), "float32")
        ys = F.cast(F.gather_nd(center, reg_ys).reshape((-1, self._topk)), "float32")
        topk_xs = F.cast(topk_xs, "float32") + xs
        topk_ys = F.cast(topk_ys, "float32") + ys
        w = F.cast(F.gather_nd(wh, reg_xs).reshape((-1, self._topk)), "float32")
        h = F.cast(F.gather_nd(wh, reg_ys).reshape((-1, self._topk)), "float32")
        half_w = w / 2
        half_h = h / 2
        results = [topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h]
        results = F.concat(*[tmp.expand_dims(-1) for tmp in results], dim=-1)
        return topk_classes, scores, results * self._scale


class CenterNet(nn.HybridBlock):
    """
    Objects as Points. https://arxiv.org/abs/1904.07850v2

    Parameters
    ----------
    backbone : mxnet.gluon.nn.HybridBlock
        The base feature extraction network.
    heads : OrderedDict
        OrderedDict with specifications for each head.
        For example: OrderedDict([
            ('heatmap', {'num_output': len(classes), 'bias': -2.19}),
            ('wh', {'num_output': 2}),
            ('reg', {'num_output': 2})
            ])
    head_conv_channel : int, default is 0
        If > 0, will use an extra conv layer before each of the real heads.
    scale : float, default is 4.0
        The downsampling ratio of the entire network.
    topk : int, default is 100
        Number of outputs .
    """
    def __init__(self,
                 backbone,
                 heads,
                 head_conv_channel=0,
                 scale=4.0,
                 topk=100,
                 in_channels=3,
                 in_size=(512, 512),
                 **kwargs):
        super(CenterNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.in_channels = in_channels

        with self.name_scope():
            self.backbone = backbone

            self.heatmap_nms = nn.MaxPool2D(
                pool_size=3,
                strides=1,
                padding=1)
            self.decoder = CenterNetDecoder(
                topk=topk,
                scale=scale)
            self.heads = nn.HybridSequential(prefix="heads")
            for name, values in heads.items():
                head = nn.HybridSequential(name)
                num_output = values['num_output']
                if head_conv_channel > 0:
                    head.add(nn.Conv2D(
                        head_conv_channel,
                        kernel_size=3,
                        padding=1,
                        use_bias=True))
                    head.add(nn.Activation('relu'))
                head.add(nn.Conv2D(
                    num_output,
                    kernel_size=1,
                    strides=1,
                    padding=0,
                    use_bias=True))

                self.heads.add(head)

    def hybrid_forward(self, F, x):
        y = self.backbone(x)
        out = [head(y) for head in self.heads]
        out[0] = F.sigmoid(out[0])
        if autograd.is_training():
            out[0] = F.clip(out[0], 1e-4, 1 - 1e-4)
            return tuple(out)
        heatmap = out[0]
        keep = F.broadcast_equal(self.heatmap_nms(heatmap), heatmap)
        results = self.decoder(keep * heatmap, out[1], out[2])

        a, b, c = results
        a = a.expand_dims(2)
        b = b.expand_dims(2)
        results = F.concat(c, a, b, dim=2)
        return results


def get_center_net(model_name,
                   classes,
                   pretrained=False,
                   ctx=mx.cpu(),
                   root=os.path.join('~', '.mxnet', 'models'),
                   **kwargs):
    """
    Get a center net instance.

    Parameters
    ----------
    model_name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    heads = OrderedDict([
        ('heatmap', {'num_output': classes, 'bias': -2.19}),  # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    head_conv_channel = 64
    topk = 40

    net = CenterNet(
        heads=heads,
        head_conv_channel=head_conv_channel,
        topk=topk,
        **kwargs)

    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file(model_name, tag=pretrained, root=root), ctx=ctx)

    return net


def centernet_resnet18_voc(pretrained_backbone=False, classes=20, **kwargs):
    backbone = resnet18_v1b_deconv(pretrained=pretrained_backbone)
    return get_center_net(classes=classes, backbone=backbone, model_name='center_net_resnet18_v1b_voc', **kwargs)


def centernet_resnet18_coco(pretrained_backbone=False, classes=80, **kwargs):
    backbone = resnet18_v1b_deconv(pretrained=pretrained_backbone)
    return get_center_net(classes=classes, backbone=backbone, model_name='center_net_resnet18_v1b_coco', **kwargs)


def centernet_resnet50b_voc(pretrained_backbone=False, classes=20, **kwargs):
    backbone = resnet50_v1b_deconv(pretrained=pretrained_backbone)
    return get_center_net(classes=classes, backbone=backbone, model_name='center_net_resnet50_v1b_voc', **kwargs)


def centernet_resnet50b_coco(pretrained_backbone=False, classes=80, **kwargs):
    backbone = resnet50_v1b_deconv(pretrained=pretrained_backbone)
    return get_center_net(classes=classes, backbone=backbone, model_name='center_net_resnet50_v1b_coco', **kwargs)


def centernet_resnet101b_voc(pretrained_backbone=False, classes=20, **kwargs):
    backbone = resnet101_v1b_deconv(pretrained=pretrained_backbone)
    return get_center_net(classes=classes, backbone=backbone, model_name='center_net_resnet101_v1b_voc', **kwargs)


def centernet_resnet101b_coco(pretrained_backbone=False, classes=80, **kwargs):
    backbone = resnet101_v1b_deconv(pretrained=pretrained_backbone)
    return get_center_net(classes=classes, backbone=backbone, model_name='center_net_resnet101_v1b_coco', **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (512, 512)
    pretrained = False

    models = [
        centernet_resnet18_voc,
        centernet_resnet18_coco,
        centernet_resnet50b_voc,
        centernet_resnet50b_coco,
        centernet_resnet101b_voc,
        centernet_resnet101b_coco,
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
        assert (model != centernet_resnet18_voc or weight_count == 14215640)
        assert (model != centernet_resnet18_coco or weight_count == 14219540)
        assert (model != centernet_resnet50b_voc or weight_count == 30086104)
        assert (model != centernet_resnet50b_coco or weight_count == 30090004)
        assert (model != centernet_resnet101b_voc or weight_count == 49078232)
        assert (model != centernet_resnet101b_coco or weight_count == 49082132)


if __name__ == "__main__":
    _test()
