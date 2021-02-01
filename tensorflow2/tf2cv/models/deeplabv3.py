"""
    DeepLabv3 for image segmentation, implemented in TensorFlow.
    Original paper: 'Rethinking Atrous Convolution for Semantic Image Segmentation,' https://arxiv.org/abs/1706.05587.
"""

__all__ = ['DeepLabv3', 'deeplabv3_resnetd50b_voc', 'deeplabv3_resnetd101b_voc', 'deeplabv3_resnetd152b_voc',
           'deeplabv3_resnetd50b_coco', 'deeplabv3_resnetd101b_coco', 'deeplabv3_resnetd152b_coco',
           'deeplabv3_resnetd50b_ade20k', 'deeplabv3_resnetd101b_ade20k', 'deeplabv3_resnetd50b_cityscapes',
           'deeplabv3_resnetd101b_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, Concurrent, is_channels_first, interpolate_im,\
    get_im_size
from .resnetd import resnetd50b, resnetd101b, resnetd152b


class DeepLabv3FinalBlock(nn.Layer):
    """
    DeepLabv3 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bottleneck_factor=4,
                 data_format="channels_last",
                 **kwargs):
        super(DeepLabv3FinalBlock, self).__init__(**kwargs)
        assert (in_channels % bottleneck_factor == 0)
        self.data_format = data_format
        mid_channels = in_channels // bottleneck_factor

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.dropout = nn.Dropout(
            rate=0.1,
            name="dropout")
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="conv2")

    def call(self, x, out_size, training=None):
        x = self.conv1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        x = interpolate_im(x, out_size=out_size, data_format=self.data_format)
        return x


class ASPPAvgBranch(nn.Layer):
    """
    ASPP branch with average pooling.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    upscale_out_size : tuple of 2 int or None
        Spatial size of output image for the bilinear upsampling operation.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 upscale_out_size,
                 data_format="channels_last",
                 **kwargs):
        super(ASPPAvgBranch, self).__init__(**kwargs)
        self.upscale_out_size = upscale_out_size
        self.data_format = data_format

        self.pool = nn.AveragePooling2D(
            pool_size=1,
            data_format=data_format,
            name="pool")
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        in_size = self.upscale_out_size if self.upscale_out_size is not None else\
            get_im_size(x, data_format=self.data_format)
        x = self.pool(x)
        x = self.conv(x, training=training)
        x = interpolate_im(x, out_size=in_size, data_format=self.data_format)
        return x


class AtrousSpatialPyramidPooling(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    upscale_out_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 upscale_out_size,
                 data_format="channels_last",
                 **kwargs):
        super(AtrousSpatialPyramidPooling, self).__init__(**kwargs)
        atrous_rates = [12, 24, 36]
        assert (in_channels % 8 == 0)
        mid_channels = in_channels // 8
        project_in_channels = 5 * mid_channels

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="branch1"))
        for i, atrous_rate in enumerate(atrous_rates):
            self.branches.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                padding=atrous_rate,
                dilation=atrous_rate,
                data_format=data_format,
                name="branch{}".format(i + 2)))
        self.branches.add(ASPPAvgBranch(
            in_channels=in_channels,
            out_channels=mid_channels,
            upscale_out_size=upscale_out_size,
            data_format=data_format,
            name="branch5"))
        self.conv = conv1x1_block(
            in_channels=project_in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv")
        self.dropout = nn.Dropout(
            rate=0.5,
            name="dropout")

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        x = self.conv(x, training=training)
        x = self.dropout(x, training=training)
        return x


class DeepLabv3(tf.keras.Model):
    """
    DeepLabv3 model from 'Rethinking Atrous Convolution for Semantic Image Segmentation,'
    https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int, default 2048
        Number of output channels form feature extractor.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (480, 480)
        Spatial size of the expected input image.
    classes : int, default 21
        Number of segmentation classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 classes=21,
                 data_format="channels_last",
                 **kwargs):
        super(DeepLabv3, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size
        self.data_format = data_format

        self.backbone = backbone
        pool_out_size = (self.in_size[0] // 8, self.in_size[1] // 8) if fixed_size else None
        self.pool = AtrousSpatialPyramidPooling(
            in_channels=backbone_out_channels,
            upscale_out_size=pool_out_size,
            data_format=data_format,
            name="pool")
        pool_out_channels = backbone_out_channels // 8
        self.final_block = DeepLabv3FinalBlock(
            in_channels=pool_out_channels,
            out_channels=classes,
            bottleneck_factor=1,
            data_format=data_format,
            name="final_block")
        if self.aux:
            aux_out_channels = backbone_out_channels // 2
            self.aux_block = DeepLabv3FinalBlock(
                in_channels=aux_out_channels,
                out_channels=classes,
                bottleneck_factor=4,
                data_format=data_format,
                name="aux_block")

    def call(self, x, training=None):
        in_size = self.in_size if self.fixed_size else get_im_size(x, data_format=self.data_format)
        x, y = self.backbone(x, training=training)
        x = self.pool(x, training=training)
        x = self.final_block(x, in_size, training=training)
        if self.aux:
            y = self.aux_block(y, in_size, training=training)
            return x, y
        else:
            return x


def get_deeplabv3(backbone,
                  classes,
                  aux=False,
                  model_name=None,
                  data_format="channels_last",
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
                  **kwargs):
    """
    Create DeepLabv3 model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    classes : int
        Number of segmentation classes.
    aux : bool, default False
        Whether to output an auxiliary result.
    model_name : str or None, default None
        Model name for loading pretrained model.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = DeepLabv3(
        backbone=backbone,
        classes=classes,
        aux=aux,
        data_format=data_format,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            by_name=True,
            skip_mismatch=True)

    return net


def deeplabv3_resnetd50b_voc(pretrained_backbone=False, classes=21, aux=True, data_format="channels_last", **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for Pascal VOC from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                          data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_voc",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd101b_voc(pretrained_backbone=False, classes=21, aux=True, data_format="channels_last", **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for Pascal VOC from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                           data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_voc",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd152b_voc(pretrained_backbone=False, classes=21, aux=True, data_format="channels_last", **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-152b for Pascal VOC from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd152b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                           data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd152b_voc",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd50b_coco(pretrained_backbone=False, classes=21, aux=True, data_format="channels_last", **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for COCO from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                          data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_coco",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd101b_coco(pretrained_backbone=False, classes=21, aux=True, data_format="channels_last", **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for COCO from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                           data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_coco",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd152b_coco(pretrained_backbone=False, classes=21, aux=True, data_format="channels_last", **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-152b for COCO from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 21
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd152b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                           data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd152b_coco",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd50b_ade20k(pretrained_backbone=False, classes=150, aux=True, data_format="channels_last",
                                **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for ADE20K from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 150
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                          data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_ade20k",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd101b_ade20k(pretrained_backbone=False, classes=150, aux=True, data_format="channels_last",
                                 **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for ADE20K from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 150
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                           data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_ade20k",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd50b_cityscapes(pretrained_backbone=False, classes=19, aux=True, data_format="channels_last",
                                    **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-50b for Cityscapes from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                          data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd50b_cityscapes",
                         data_format=data_format, **kwargs)


def deeplabv3_resnetd101b_cityscapes(pretrained_backbone=False, classes=19, aux=True, data_format="channels_last",
                                     **kwargs):
    """
    DeepLabv3 model on the base of ResNet(D)-101b for Cityscapes from 'Rethinking Atrous Convolution for Semantic Image
    Segmentation,' https://arxiv.org/abs/1706.05587.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                           data_format=data_format).features
    del backbone.children[-1]
    return get_deeplabv3(backbone=backbone, classes=classes, aux=aux, model_name="deeplabv3_resnetd101b_cityscapes",
                         data_format=data_format, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (480, 480)
    aux = False
    pretrained = False

    models = [
        (deeplabv3_resnetd50b_voc, 21),
        (deeplabv3_resnetd101b_voc, 21),
        (deeplabv3_resnetd152b_voc, 21),
        (deeplabv3_resnetd50b_coco, 21),
        (deeplabv3_resnetd101b_coco, 21),
        (deeplabv3_resnetd152b_coco, 21),
        (deeplabv3_resnetd50b_ade20k, 150),
        (deeplabv3_resnetd101b_ade20k, 150),
        (deeplabv3_resnetd50b_cityscapes, 19),
        (deeplabv3_resnetd101b_cityscapes, 19),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape[0] == x.shape[0])
        if is_channels_first(data_format):
            assert ((y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and (y.shape[3] == x.shape[3]))
        else:
            assert ((y.shape[3] == classes) and (y.shape[1] == x.shape[1]) and (y.shape[2] == x.shape[2]))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        if aux:
            assert (model != deeplabv3_resnetd50b_voc or weight_count == 42127850)
            assert (model != deeplabv3_resnetd101b_voc or weight_count == 61119978)
            assert (model != deeplabv3_resnetd152b_voc or weight_count == 76763626)
            assert (model != deeplabv3_resnetd50b_coco or weight_count == 42127850)
            assert (model != deeplabv3_resnetd101b_coco or weight_count == 61119978)
            assert (model != deeplabv3_resnetd152b_coco or weight_count == 76763626)
            assert (model != deeplabv3_resnetd50b_ade20k or weight_count == 42194156)
            assert (model != deeplabv3_resnetd101b_ade20k or weight_count == 61186284)
            assert (model != deeplabv3_resnetd50b_cityscapes or weight_count == 42126822)
            assert (model != deeplabv3_resnetd101b_cityscapes or weight_count == 61118950)
        else:
            assert (model != deeplabv3_resnetd50b_voc or weight_count == 39762645)
            assert (model != deeplabv3_resnetd101b_voc or weight_count == 58754773)
            assert (model != deeplabv3_resnetd152b_voc or weight_count == 74398421)
            assert (model != deeplabv3_resnetd50b_coco or weight_count == 39762645)
            assert (model != deeplabv3_resnetd101b_coco or weight_count == 58754773)
            assert (model != deeplabv3_resnetd152b_coco or weight_count == 74398421)
            assert (model != deeplabv3_resnetd50b_ade20k or weight_count == 39795798)
            assert (model != deeplabv3_resnetd101b_ade20k or weight_count == 58787926)
            assert (model != deeplabv3_resnetd50b_cityscapes or weight_count == 39762131)
            assert (model != deeplabv3_resnetd101b_cityscapes or weight_count == 58754259)


if __name__ == "__main__":
    _test()
