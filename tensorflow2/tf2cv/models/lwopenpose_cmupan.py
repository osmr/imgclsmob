"""
    Lightweight OpenPose 2D/3D for CMU Panoptic, implemented in TensorFlow.
    Original paper: 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.
"""

__all__ = ['LwOpenPose', 'lwopenpose2d_mobilenet_cmupan_coco', 'lwopenpose3d_mobilenet_cmupan_coco',
           'LwopDecoderFinalBlock']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, dwsconv3x3_block, SimpleSequential, is_channels_first,\
    get_channel_axis


class LwopResBottleneck(nn.Layer):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False,
                 data_format="channels_last",
                 **kwargs):
        super(LwopResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor if squeeze_out else in_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_bias=use_bias,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=strides,
            use_bias=use_bias,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            activation=None,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class LwopResUnit(nn.Layer):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    activate : bool, default False
        Whether to activate the sum.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 use_bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False,
                 activate=False,
                 data_format="channels_last",
                 **kwargs):
        super(LwopResUnit, self).__init__(**kwargs)
        self.activate = activate
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        self.body = LwopResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            use_bias=use_bias,
            bottleneck_factor=bottleneck_factor,
            squeeze_out=squeeze_out,
            data_format=data_format,
            name="body")
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                use_bias=use_bias,
                activation=None,
                data_format=data_format,
                name="identity_conv")
        if self.activate:
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        if self.resize_identity:
            identity = self.identity_conv(x, training=training)
        else:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        if self.activate:
            x = self.activ(x)
        return x


class LwopEncoderFinalBlock(nn.Layer):
    """
    Lightweight OpenPose 2D/3D specific encoder final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(LwopEncoderFinalBlock, self).__init__(**kwargs)
        self.pre_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=True,
            use_bn=False,
            data_format=data_format,
            name="pre_conv")
        self.body = SimpleSequential(name="body")
        for i in range(3):
            self.body.add(dwsconv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                dw_use_bn=False,
                pw_use_bn=False,
                dw_activation=(lambda: nn.ELU()),
                pw_activation=(lambda: nn.ELU()),
                data_format=data_format,
                name="block{}".format(i + 1)))
        self.post_conv = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bias=True,
            use_bn=False,
            data_format=data_format,
            name="post_conv")

    def call(self, x, training=None):
        x = self.pre_conv(x, training=training)
        x = x + self.body(x, training=training)
        x = self.post_conv(x, training=training)
        return x


class LwopRefinementBlock(nn.Layer):
    """
    Lightweight OpenPose 2D/3D specific refinement block for decoder units.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(LwopRefinementBlock, self).__init__(**kwargs)
        self.pre_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=True,
            use_bn=False,
            data_format=data_format,
            name="pre_conv")
        self.body = SimpleSequential(name="body")
        self.body.add(conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="block1"))
        self.body.add(conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            padding=2,
            dilation=2,
            use_bias=True,
            data_format=data_format,
            name="block2"))

    def call(self, x, training=None):
        x = self.pre_conv(x, training=training)
        # print("--> x.shape={}".format(x.shape))
        y = self.body(x, training=training)
        # print("==> x.shape={}".format(x.shape))
        x = x + y
        return x


class LwopDecoderBend(nn.Layer):
    """
    Lightweight OpenPose 2D/3D specific decoder bend block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(LwopDecoderBend, self).__init__(**kwargs)
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_bias=True,
            use_bn=False,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class LwopDecoderInitBlock(nn.Layer):
    """
    Lightweight OpenPose 2D/3D specific decoder init block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 data_format="channels_last",
                 **kwargs):
        super(LwopDecoderInitBlock, self).__init__(**kwargs)
        self.data_format = data_format
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        bend_mid_channels = 512

        self.body = SimpleSequential(name="body")
        for i in range(3):
            self.body.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=in_channels,
                use_bias=True,
                use_bn=False,
                data_format=data_format,
                name="block{}".format(i + 1)))
        self.heatmap_bend = LwopDecoderBend(
            in_channels=in_channels,
            mid_channels=bend_mid_channels,
            out_channels=num_heatmap,
            data_format=data_format,
            name="heatmap_bend")
        self.paf_bend = LwopDecoderBend(
            in_channels=in_channels,
            mid_channels=bend_mid_channels,
            out_channels=num_paf,
            data_format=data_format,
            name="paf_bend")

    def call(self, x, training=None):
        y = self.body(x, training=training)
        heatmap = self.heatmap_bend(y, training=training)
        paf = self.paf_bend(y, training=training)
        y = tf.concat((x, heatmap, paf), axis=get_channel_axis(self.data_format))
        return y


class LwopDecoderUnit(nn.Layer):
    """
    Lightweight OpenPose 2D/3D specific decoder init.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 data_format="channels_last",
                 **kwargs):
        super(LwopDecoderUnit, self).__init__(**kwargs)
        self.data_format = data_format
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        self.features_channels = in_channels - num_heatmap - num_paf

        self.body = SimpleSequential(name="body")
        for i in range(5):
            self.body.add(LwopRefinementBlock(
                in_channels=in_channels,
                out_channels=self.features_channels,
                data_format=data_format,
                name="block{}".format(i + 1)))
            in_channels = self.features_channels
        self.heatmap_bend = LwopDecoderBend(
            in_channels=self.features_channels,
            mid_channels=self.features_channels,
            out_channels=num_heatmap,
            data_format=data_format,
            name="heatmap_bend")
        self.paf_bend = LwopDecoderBend(
            in_channels=self.features_channels,
            mid_channels=self.features_channels,
            out_channels=num_paf,
            data_format=data_format,
            name="paf_bend")

    def call(self, x, training=None):
        if is_channels_first(self.data_format):
            features = x[:, :self.features_channels]
        else:
            features = x[:, :, :, :self.features_channels]
        y = self.body(x, training=training)
        heatmap = self.heatmap_bend(y, training=training)
        paf = self.paf_bend(y, training=training)
        y = tf.concat((features, heatmap, paf), axis=get_channel_axis(self.data_format))
        return y


class LwopDecoderFeaturesBend(nn.Layer):
    """
    Lightweight OpenPose 2D/3D specific decoder 3D features bend.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(LwopDecoderFeaturesBend, self).__init__(**kwargs)
        self.body = SimpleSequential(name="body")
        for i in range(2):
            self.body.add(LwopRefinementBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                data_format=data_format,
                name="block{}".format(i + 1)))
            in_channels = mid_channels
        self.features_bend = LwopDecoderBend(
            in_channels=mid_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            data_format=data_format,
            name="features_bend")

    def call(self, x, training=None):
        x = self.body(x, training=training)
        x = self.features_bend(x, training=training)
        return x


class LwopDecoderFinalBlock(nn.Layer):
    """
    Lightweight OpenPose 2D/3D specific decoder final block for calcualation 3D poses.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    bottleneck_factor : int
        Bottleneck factor.
    calc_3d_features : bool
        Whether to calculate 3D features.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 bottleneck_factor,
                 calc_3d_features,
                 data_format="channels_last",
                 **kwargs):
        super(LwopDecoderFinalBlock, self).__init__(**kwargs)
        self.data_format = data_format
        self.num_heatmap_paf = 3 * keypoints
        self.calc_3d_features = calc_3d_features
        features_out_channels = self.num_heatmap_paf
        features_in_channels = in_channels - features_out_channels

        if self.calc_3d_features:
            self.body = SimpleSequential(name="body")
            for i in range(5):
                self.body.add(LwopResUnit(
                    in_channels=in_channels,
                    out_channels=features_in_channels,
                    bottleneck_factor=bottleneck_factor,
                    data_format=data_format,
                    name="block{}".format(i + 1)))
                in_channels = features_in_channels
            self.features_bend = LwopDecoderFeaturesBend(
                in_channels=features_in_channels,
                mid_channels=features_in_channels,
                out_channels=features_out_channels,
                data_format=data_format,
                name="features_bend")

    def call(self, x, training=None):
        if is_channels_first(self.data_format):
            heatmap_paf_2d = x[:, -self.num_heatmap_paf:]
        else:
            heatmap_paf_2d = x[:, :, :, -self.num_heatmap_paf:]
        if not self.calc_3d_features:
            return heatmap_paf_2d
        x = self.body(x, training=training)
        x = self.features_bend(x, training=training)
        y = tf.concat((heatmap_paf_2d, x), axis=get_channel_axis(self.data_format))
        return y


class LwOpenPose(tf.keras.Model):
    """
    Lightweight OpenPose 2D/3D model from 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    encoder_channels : list of list of int
        Number of output channels for each encoder unit.
    encoder_paddings : list of list of int
        Padding/dilation value for each encoder unit.
    encoder_init_block_channels : int
        Number of output channels for the encoder initial unit.
    encoder_final_block_channels : int
        Number of output channels for the encoder final unit.
    refinement_units : int
        Number of refinement blocks in the decoder.
    calc_3d_features : bool
        Whether to calculate 3D features.
    return_heatmap : bool, default True
        Whether to return only heatmap.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 192)
        Spatial size of the expected input image.
    keypoints : int, default 19
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 encoder_channels,
                 encoder_paddings,
                 encoder_init_block_channels,
                 encoder_final_block_channels,
                 refinement_units,
                 calc_3d_features,
                 return_heatmap=True,
                 in_channels=3,
                 in_size=(368, 368),
                 keypoints=19,
                 data_format="channels_last",
                 **kwargs):
        super(LwOpenPose, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.data_format = data_format
        self.return_heatmap = return_heatmap
        self.calc_3d_features = calc_3d_features
        num_heatmap_paf = 3 * keypoints

        self.encoder = SimpleSequential(name="encoder")
        backbone = SimpleSequential(name="backbone")
        backbone.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=encoder_init_block_channels,
            strides=2,
            data_format=data_format,
            name="init_block"))
        in_channels = encoder_init_block_channels
        for i, channels_per_stage in enumerate(encoder_channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if (j == 0) and (i != 0) else 1
                padding = encoder_paddings[i][j]
                stage.add(dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=padding,
                    dilation=padding,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            backbone.add(stage)
        self.encoder.add(backbone)
        self.encoder.add(LwopEncoderFinalBlock(
            in_channels=in_channels,
            out_channels=encoder_final_block_channels,
            data_format=data_format,
            name="final_block"))
        in_channels = encoder_final_block_channels

        self.decoder = SimpleSequential(name="decoder")
        self.decoder.add(LwopDecoderInitBlock(
            in_channels=in_channels,
            keypoints=keypoints,
            data_format=data_format,
            name="init_block"))
        in_channels = encoder_final_block_channels + num_heatmap_paf
        for i in range(refinement_units):
            self.decoder.add(LwopDecoderUnit(
                in_channels=in_channels,
                keypoints=keypoints,
                data_format=data_format,
                name="unit{}".format(i + 1)))
        self.decoder.add(LwopDecoderFinalBlock(
            in_channels=in_channels,
            keypoints=keypoints,
            bottleneck_factor=2,
            calc_3d_features=calc_3d_features,
            data_format=data_format,
            name="final_block"))

    def call(self, x, training=None):
        # print("**> x.shape={}".format(x.shape))
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        if self.return_heatmap:
            return x
        else:
            return x


def get_lwopenpose(calc_3d_features,
                   keypoints,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".tensorflow", "models"),
                   **kwargs):
    """
    Create Lightweight OpenPose 2D/3D model with specific parameters.

    Parameters:
    ----------
    calc_3d_features : bool, default False
        Whether to calculate 3D features.
    keypoints : int
        Number of keypoints.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    encoder_channels = [[64], [128, 128], [256, 256, 512, 512, 512, 512, 512, 512]]
    encoder_paddings = [[1], [1, 1], [1, 1, 1, 2, 1, 1, 1, 1]]
    encoder_init_block_channels = 32
    encoder_final_block_channels = 128
    refinement_units = 1

    net = LwOpenPose(
        encoder_channels=encoder_channels,
        encoder_paddings=encoder_paddings,
        encoder_init_block_channels=encoder_init_block_channels,
        encoder_final_block_channels=encoder_final_block_channels,
        refinement_units=refinement_units,
        calc_3d_features=calc_3d_features,
        keypoints=keypoints,
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


def lwopenpose2d_mobilenet_cmupan_coco(keypoints=19, data_format="channels_last", **kwargs):
    """
    Lightweight OpenPose 2D model on the base of MobileNet for CMU Panoptic from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    keypoints : int, default 19
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose(calc_3d_features=False, keypoints=keypoints, model_name="lwopenpose2d_mobilenet_cmupan_coco",
                          data_format=data_format, **kwargs)


def lwopenpose3d_mobilenet_cmupan_coco(keypoints=19, data_format="channels_last", **kwargs):
    """
    Lightweight OpenPose 3D model on the base of MobileNet for CMU Panoptic from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    keypoints : int, default 19
        Number of keypoints.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose(calc_3d_features=True, keypoints=keypoints, model_name="lwopenpose3d_mobilenet_cmupan_coco",
                          data_format=data_format, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    # os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size_ = (368, 368)
    keypoints = 19
    return_heatmap = True
    pretrained = False

    models = [
        (lwopenpose2d_mobilenet_cmupan_coco, "2d", in_size_),
        (lwopenpose3d_mobilenet_cmupan_coco, "3d", in_size_),
    ]

    for model, model_dim, in_size in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (y.shape[0] == batch)
        keypoints_ = 3 * keypoints if model_dim == "2d" else 6 * keypoints
        if is_channels_first(data_format):
            assert ((y.shape[1] == keypoints_) and (y.shape[2] == x.shape[2] // 8) and
                    (y.shape[3] == x.shape[3] // 8))
        else:
            assert ((y.shape[3] == keypoints_) and (y.shape[1] == x.shape[1] // 8) and
                    (y.shape[2] == x.shape[2] // 8))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lwopenpose2d_mobilenet_cmupan_coco or weight_count == 4091698)
        assert (model != lwopenpose3d_mobilenet_cmupan_coco or weight_count == 5085983)


if __name__ == "__main__":
    _test()
