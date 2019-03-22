from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from .oth_segbase import SegBaseModel
from .oth_fcn import _FCNHead

__all__ = ['PSPNet', 'get_psp', 'oth_psp_resnet101_coco', 'oth_psp_resnet101_voc',
           'oth_psp_resnet50_ade', 'oth_psp_resnet101_ade', 'oth_psp_resnet101_citys']


class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.


    Reference:

        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017

    """
    def __init__(self, nclass, backbone='resnet50', aux=True, ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(PSPNet, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        with self.name_scope():
            self.head = _PSPHead(nclass, height=self._up_kwargs['height']//8,
                                 width=self._up_kwargs['width']//8, **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                aux_kwargs = kwargs
                if "classes" in aux_kwargs:
                    del aux_kwargs["classes"]
                if "in_channels" in aux_kwargs:
                    del aux_kwargs["in_channels"]
                self.auxlayer = _FCNHead(1024, nclass, **aux_kwargs)
                self.auxlayer.initialize(ctx=ctx)
                self.auxlayer.collect_params().setattr('lr_mult', 10)
        print('self.crop_size', self.crop_size)

    def hybrid_forward(self, F, x):
        self._up_kwargs['height'] = x.shape[2]
        self._up_kwargs['width'] = x.shape[3]

        c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        else:
            return outputs[0]
        return tuple(outputs)

    # def demo(self, x):
    #     h, w = x.shape[2:]
    #     self._up_kwargs['height'] = h
    #     self._up_kwargs['width'] = w
    #     c3, c4 = self.base_forward(x)
    #     outputs = []
    #     x = self.head.demo(c4)
    #     import mxnet.ndarray as F
    #     pred = F.contrib.BilinearResize2D(x, **self._up_kwargs)
    #     return pred

def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=1, use_bias=False))
        block.add(norm_layer(in_channels=out_channels, **({} if norm_kwargs is None else norm_kwargs)))
        block.add(nn.Activation('relu'))
    return block


class _PyramidPooling(HybridBlock):
    def __init__(self, in_channels, height=60, width=60, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels/4)
        self._up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def pool(self, F, x, size):
        return F.contrib.AdaptiveAvgPooling2D(x, output_size=size)

    def upsample(self, F, x):
        return F.contrib.BilinearResize2D(x, **self._up_kwargs)

    def hybrid_forward(self, F, x):
        self._up_kwargs['height'] = x.shape[2]
        self._up_kwargs['width'] = x.shape[3]

        feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)))
        feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)))
        feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)))
        feat4 = self.upsample(F, self.conv4(self.pool(F, x, 6)))
        return F.concat(x, feat1, feat2, feat3, feat4, dim=1)

    # def demo(self, x):
    #     self._up_kwargs['height'] = x.shape[2]
    #     self._up_kwargs['width'] = x.shape[3]
    #     import mxnet.ndarray as F
    #     feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)))
    #     feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)))
    #     feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)))
    #     feat4 = self.upsample(F, self.conv4(self.pool(F, x, 6)))
    #     return F.concat(x, feat1, feat2, feat3, feat4, dim=1)

class _PSPHead(HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 height=60, width=60, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, height=height, width=width,
                                   norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.block = nn.HybridSequential(prefix='')
            self.block.add(nn.Conv2D(in_channels=4096, channels=512,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=512, **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=512, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.psp(x)
        return self.block(x)

    # def demo(self, x):
    #     x = self.psp.demo(x)
    #     return self.block(x)
        

def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), pretrained_base=True, num_class=150, **kwargs):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    # from ..data import datasets
    # infer number of classes
    model = PSPNet(num_class, backbone=backbone,
                   pretrained_base=pretrained_base, ctx=ctx, **kwargs)
    # if pretrained:
    #     from .model_store import get_model_file
    #     model.load_parameters(get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]),
    #                                          tag=pretrained, root=root), ctx=ctx)
    return model


def oth_psp_resnet101_coco(**kwargs):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_psp('coco', 'resnet101', num_class=21, **kwargs)


def oth_psp_resnet101_voc(**kwargs):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_psp('pascal_voc', 'resnet101', num_class=21, **kwargs)


def oth_psp_resnet50_ade(**kwargs):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_psp('ade20k', 'resnet50', num_class=150, **kwargs)


def oth_psp_resnet101_ade(**kwargs):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_psp('ade20k', 'resnet101', num_class=150, **kwargs)


def oth_psp_resnet101_citys(**kwargs):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_psp('citys', 'resnet101', num_class=19, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        (oth_psp_resnet50_ade, 150),
        (oth_psp_resnet101_ade, 150),
        (oth_psp_resnet101_coco, 21),
        (oth_psp_resnet101_voc, 21),
        (oth_psp_resnet101_citys, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        x = mx.nd.zeros((1, 3, 480, 480), ctx=ctx)
        y = net(x)[0]

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_psp_resnet50_ade or weight_count == 49180908)
        assert (model != oth_psp_resnet101_ade or weight_count == 68173036)
        assert (model != oth_psp_resnet101_coco or weight_count == 68073706)
        assert (model != oth_psp_resnet101_voc or weight_count == 68073706)
        assert (model != oth_psp_resnet101_citys or weight_count == 68072166)

        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
