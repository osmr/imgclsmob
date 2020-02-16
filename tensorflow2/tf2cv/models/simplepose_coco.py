"""
    SimplePose for COCO Keypoint, implemented in TensorFlow.
    Original paper: 'Simple Baselines for Human Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.
"""

__all__ = ['SimplePose', 'simplepose_resnet18_coco', 'simplepose_resnet50b_coco', 'simplepose_resnet101b_coco',
           'simplepose_resnet152b_coco', 'simplepose_resneta50b_coco', 'simplepose_resneta101b_coco',
           'simplepose_resneta152b_coco', 'HeatmapMaxDetBlock']

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import get_activation_layer, BatchNorm, conv1x1, is_channels_first
from .resnet import resnet18, resnet50b, resnet101b, resnet152b
from .resneta import resneta50b, resneta101b, resneta152b


class Deconv2d(nn.Layer):
    """
    Standard deconvolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    out_padding : int or tuple/list of 2 int, default 0
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 out_padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=True,
                 data_format="channels_last",
                 **kwargs):
        super(Deconv2d, self).__init__(**kwargs)
        assert (dilation == 1)
        assert (groups == 1)
        assert (in_channels is not None)

        if isinstance(padding, int):
            padding = (padding, padding)

        self.use_crop = (padding[0] > 0) or (padding[1] > 0)
        if self.use_crop:
            self.crop = nn.Cropping2D(
                cropping=padding,
                data_format=data_format,
                name="crop")

        self.conv = nn.Conv2DTranspose(
            filters=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            output_padding=out_padding,
            data_format=data_format,
            dilation_rate=dilation,
            use_bias=use_bias,
            name="conv")

    def call(self, x):
        x = self.conv(x)
        if self.use_crop:
            x = self.crop(x)
        return x


class DeconvBlock(nn.Layer):
    """
    Deconvolution block with batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the deconvolution.
    padding : int or tuple/list of 2 int
        Padding value for deconvolution layer.
    out_padding : int or tuple/list of 2 int, default 0
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default 'relu'
        Activation function or name of activation function.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 out_padding=0,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation="relu",
                 data_format="channels_last",
                 **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)
        assert (in_channels is not None)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = Deconv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            out_padding=out_padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            data_format=data_format,
            name="conv")
        if self.use_bn:
            self.bn = BatchNorm(
                epsilon=bn_eps,
                data_format=data_format,
                name="bn")
        if self.activate:
            self.activ = get_activation_layer(activation)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activ(x)
        return x


class HeatmapMaxDetBlock(nn.Layer):
    """
    Heatmap maximum detector block.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(HeatmapMaxDetBlock, self).__init__(**kwargs)
        self.data_format = data_format

    def call(self, x, training=None):
        heatmap = x
        vector_dim = 2
        batch = heatmap.shape[0]
        if is_channels_first(self.data_format):
            channels = heatmap.shape[1]
            in_size = x.shape[2:]
            heatmap_vector = tf.reshape(heatmap, shape=(batch, channels, -1))
        else:
            channels = heatmap.shape[3]
            in_size = x.shape[1:3]
            heatmap_vector = tf.reshape(heatmap, shape=(batch, -1, channels))
            heatmap_vector = tf.transpose(heatmap_vector, perm=(0, 2, 1))
        indices = tf.cast(tf.expand_dims(tf.cast(tf.math.argmax(heatmap_vector, axis=vector_dim), np.int32),
                                         axis=vector_dim), np.float32)
        scores = tf.math.reduce_max(heatmap_vector, axis=vector_dim, keepdims=True)
        scores_mask = tf.cast(tf.math.greater(scores, 0.0), np.float32)
        pts_x = (indices % in_size[1]) * scores_mask
        pts_y = (indices // in_size[1]) * scores_mask
        pts = tf.concat([pts_x, pts_y, scores], axis=vector_dim).numpy()
        for b in range(batch):
            for k in range(channels):
                hm = heatmap[b, k, :, :] if is_channels_first(self.data_format) else heatmap[b, :, :, k]
                px = int(pts[b, k, 0])
                py = int(pts[b, k, 1])
                if (0 < px < in_size[1] - 1) and (0 < py < in_size[0] - 1):
                    pts[b, k, 0] += np.sign(hm[py, px + 1] - hm[py, px - 1]) * 0.25
                    pts[b, k, 1] += np.sign(hm[py + 1, px] - hm[py - 1, px]) * 0.25
        pts = tf.convert_to_tensor(pts)
        return pts


class SimplePose(tf.keras.Model):
    """
    SimplePose model from 'Simple Baselines for Human Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    channels : list of int
        Number of output channels for each decoder unit.
    return_heatmap : bool, default False
        Whether to return only heatmap.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 192)
        Spatial size of the expected input image.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels,
                 channels,
                 return_heatmap=False,
                 in_channels=3,
                 in_size=(256, 192),
                 keypoints=17,
                 data_format="channels_last",
                 **kwargs):
        super(SimplePose, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap
        self.data_format = data_format

        self.backbone = backbone
        self.backbone._name = "backbone"

        self.decoder = tf.keras.Sequential(name="decoder")
        in_channels = backbone_out_channels
        for i, out_channels in enumerate(channels):
            self.decoder.add(DeconvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                strides=2,
                padding=1,
                data_format=data_format,
                name="unit{}".format(i + 1)))
            in_channels = out_channels
        self.decoder.add(conv1x1(
            in_channels=in_channels,
            out_channels=keypoints,
            use_bias=True,
            data_format=data_format,
            name="final_block"))

        self.heatmap_max_det = HeatmapMaxDetBlock(
            data_format=data_format,
            name="heatmap_max_det")

    def call(self, x, training=None):
        x = self.backbone(x, training=training)
        heatmap = self.decoder(x, training=training)
        if self.return_heatmap or not tf.executing_eagerly():
            return heatmap
        else:
            keypoints = self.heatmap_max_det(heatmap)
            return keypoints


def get_simplepose(backbone,
                   backbone_out_channels,
                   keypoints,
                   model_name=None,
                   data_format="channels_last",
                   pretrained=False,
                   root=os.path.join("~", ".tensorflow", "models"),
                   **kwargs):
    """
    Create SimplePose model with specific parameters.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    keypoints : int
        Number of keypoints.
    model_name : str or None, default None
        Model name for loading pretrained model.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    channels = [256, 256, 256]

    net = SimplePose(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
        keypoints=keypoints,
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
                local_model_store_dir_path=root))

    return net


def simplepose_resnet18_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    SimplePose model on the base of ResNet-18 for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone, data_format=data_format).features
    backbone._layers.pop()
    return get_simplepose(backbone=backbone, backbone_out_channels=512, keypoints=keypoints,
                          model_name="simplepose_resnet18_coco", data_format=data_format, **kwargs)


def simplepose_resnet50b_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    SimplePose model on the base of ResNet-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone, data_format=data_format).features
    backbone._layers.pop()
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resnet50b_coco", data_format=data_format, **kwargs)


def simplepose_resnet101b_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    SimplePose model on the base of ResNet-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone, data_format=data_format).features
    backbone._layers.pop()
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resnet101b_coco", data_format=data_format, **kwargs)


def simplepose_resnet152b_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    SimplePose model on the base of ResNet-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnet152b(pretrained=pretrained_backbone, data_format=data_format).features
    backbone._layers.pop()
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resnet152b_coco", data_format=data_format, **kwargs)


def simplepose_resneta50b_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    SimplePose model on the base of ResNet(A)-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resneta50b(pretrained=pretrained_backbone, data_format=data_format).features
    backbone._layers.pop()
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resneta50b_coco", data_format=data_format, **kwargs)


def simplepose_resneta101b_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    SimplePose model on the base of ResNet(A)-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resneta101b(pretrained=pretrained_backbone, data_format=data_format).features
    backbone._layers.pop()
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resneta101b_coco", data_format=data_format, **kwargs)


def simplepose_resneta152b_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    SimplePose model on the base of ResNet(A)-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resneta152b(pretrained=pretrained_backbone, data_format=data_format).features
    backbone._layers.pop()
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resneta152b_coco", data_format=data_format, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (256, 192)
    keypoints = 17
    return_heatmap = False
    pretrained = False

    models = [
        simplepose_resnet18_coco,
        simplepose_resnet50b_coco,
        simplepose_resnet101b_coco,
        simplepose_resnet152b_coco,
        simplepose_resneta50b_coco,
        simplepose_resneta101b_coco,
        simplepose_resneta152b_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (y.shape[0] == batch)
        if return_heatmap:
            if is_channels_first(data_format):
                assert ((y.shape[1] == keypoints) and (y.shape[2] == x.shape[2] // 4) and
                        (y.shape[3] == x.shape[3] // 4))
            else:
                assert ((y.shape[3] == keypoints) and (y.shape[1] == x.shape[1] // 4) and
                        (y.shape[2] == x.shape[2] // 4))
        else:
            assert ((y.shape[1] == keypoints) and (y.shape[2] == 3))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != simplepose_resnet18_coco or weight_count == 15376721)
        assert (model != simplepose_resnet50b_coco or weight_count == 33999697)
        assert (model != simplepose_resnet101b_coco or weight_count == 52991825)
        assert (model != simplepose_resnet152b_coco or weight_count == 68635473)
        assert (model != simplepose_resneta50b_coco or weight_count == 34018929)
        assert (model != simplepose_resneta101b_coco or weight_count == 53011057)
        assert (model != simplepose_resneta152b_coco or weight_count == 68654705)


if __name__ == "__main__":
    _test()
