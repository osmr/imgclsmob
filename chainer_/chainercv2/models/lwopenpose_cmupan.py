"""
    Lightweight OpenPose 2D/3D for CMU Panoptic, implemented in Chainer.
    Original paper: 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.
"""

__all__ = ['LwOpenPose', 'lwopenpose2d_mobilenet_cmupan_coco', 'lwopenpose3d_mobilenet_cmupan_coco',
           'LwopDecoderFinalBlock']

import os
import chainer.functions as F
from chainer import Chain
from chainer.serializers import load_npz
from .common import conv1x1, conv1x1_block, conv3x3_block, dwsconv3x3_block, SimpleSequential


class LwopResBottleneck(Chain):
    """
    Bottleneck block for residual path in the residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False):
        super(LwopResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor if squeeze_out else in_channels // bottleneck_factor

        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                use_bias=use_bias)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LwopResUnit(Chain):
    """
    ResNet-like residual unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    use_bias : bool, default True
        Whether the layer uses a bias vector.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    activate : bool, default False
        Whether to activate the sum.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 use_bias=True,
                 bottleneck_factor=2,
                 squeeze_out=False,
                 activate=False):
        super(LwopResUnit, self).__init__()
        self.activate = activate
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.body = LwopResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=use_bias,
                bottleneck_factor=bottleneck_factor,
                squeeze_out=squeeze_out)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_bias=use_bias,
                    activation=None)
            if self.activate:
                self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        if self.activate:
            x = self.activ(x)
        return x


class LwopEncoderFinalBlock(Chain):
    """
    Lightweight OpenPose 2D/3D specific encoder final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(LwopEncoderFinalBlock, self).__init__()
        with self.init_scope():
            self.pre_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=False)
            self.body = SimpleSequential()
            with self.body.init_scope():
                for i in range(3):
                    setattr(self.body, "block{}".format(i + 1), dwsconv3x3_block(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        dw_use_bn=False,
                        pw_use_bn=False,
                        dw_activation=(lambda: F.elu),
                        pw_activation=(lambda: F.elu)))
            self.post_conv = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=False)

    def __call__(self, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        x = self.post_conv(x)
        return x


class LwopRefinementBlock(Chain):
    """
    Lightweight OpenPose 2D/3D specific refinement block for decoder units.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(LwopRefinementBlock, self).__init__()
        with self.init_scope():
            self.pre_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=False)
            self.body = SimpleSequential()
            with self.body.init_scope():
                setattr(self.body, "block1", conv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    use_bias=True))
                setattr(self.body, "block2", conv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    pad=2,
                    dilate=2,
                    use_bias=True))

    def __call__(self, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        return x


class LwopDecoderBend(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels):
        super(LwopDecoderBend, self).__init__()
        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=True,
                use_bn=False)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LwopDecoderInitBlock(Chain):
    """
    Lightweight OpenPose 2D/3D specific decoder init block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    """
    def __init__(self,
                 in_channels,
                 keypoints):
        super(LwopDecoderInitBlock, self).__init__()
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        bend_mid_channels = 512

        with self.init_scope():
            self.body = SimpleSequential()
            with self.body.init_scope():
                for i in range(3):
                    setattr(self.body, "block{}".format(i + 1), conv3x3_block(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        use_bias=True,
                        use_bn=False))
            self.heatmap_bend = LwopDecoderBend(
                in_channels=in_channels,
                mid_channels=bend_mid_channels,
                out_channels=num_heatmap)
            self.paf_bend = LwopDecoderBend(
                in_channels=in_channels,
                mid_channels=bend_mid_channels,
                out_channels=num_paf)

    def __call__(self, x):
        y = self.body(x)
        heatmap = self.heatmap_bend(y)
        paf = self.paf_bend(y)
        y = F.concat((x, heatmap, paf), axis=1)
        return y


class LwopDecoderUnit(Chain):
    """
    Lightweight OpenPose 2D/3D specific decoder init.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    """
    def __init__(self,
                 in_channels,
                 keypoints):
        super(LwopDecoderUnit, self).__init__()
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        self.features_channels = in_channels - num_heatmap - num_paf

        with self.init_scope():
            self.body = SimpleSequential()
            with self.body.init_scope():
                for i in range(5):
                    setattr(self.body, "block{}".format(i + 1), LwopRefinementBlock(
                        in_channels=in_channels,
                        out_channels=self.features_channels))
                    in_channels = self.features_channels
            self.heatmap_bend = LwopDecoderBend(
                in_channels=self.features_channels,
                mid_channels=self.features_channels,
                out_channels=num_heatmap)
            self.paf_bend = LwopDecoderBend(
                in_channels=self.features_channels,
                mid_channels=self.features_channels,
                out_channels=num_paf)

    def __call__(self, x):
        features = x[:, :self.features_channels]
        y = self.body(x)
        heatmap = self.heatmap_bend(y)
        paf = self.paf_bend(y)
        y = F.concat((features, heatmap, paf), axis=1)
        return y


class LwopDecoderFeaturesBend(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels):
        super(LwopDecoderFeaturesBend, self).__init__()
        with self.init_scope():
            self.body = SimpleSequential()
            with self.body.init_scope():
                for i in range(2):
                    setattr(self.body, "block{}".format(i + 1), LwopRefinementBlock(
                        in_channels=in_channels,
                        out_channels=mid_channels))
                    in_channels = mid_channels
            self.features_bend = LwopDecoderBend(
                in_channels=mid_channels,
                mid_channels=mid_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.body(x)
        x = self.features_bend(x)
        return x


class LwopDecoderFinalBlock(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 bottleneck_factor,
                 calc_3d_features):
        super(LwopDecoderFinalBlock, self).__init__()
        self.num_heatmap_paf = 3 * keypoints
        self.calc_3d_features = calc_3d_features
        features_out_channels = self.num_heatmap_paf
        features_in_channels = in_channels - features_out_channels

        if self.calc_3d_features:
            with self.init_scope():
                self.body = SimpleSequential()
                with self.body.init_scope():
                    for i in range(5):
                        setattr(self.body, "block{}".format(i + 1), LwopResUnit(
                            in_channels=in_channels,
                            out_channels=features_in_channels,
                            bottleneck_factor=bottleneck_factor))
                        in_channels = features_in_channels
                self.features_bend = LwopDecoderFeaturesBend(
                    in_channels=features_in_channels,
                    mid_channels=features_in_channels,
                    out_channels=features_out_channels)

    def __call__(self, x):
        heatmap_paf_2d = x[:, -self.num_heatmap_paf:]
        if not self.calc_3d_features:
            return heatmap_paf_2d
        x = self.body(x)
        x = self.features_bend(x)
        y = F.concat((heatmap_paf_2d, x), axis=1)
        return y


class LwOpenPose(Chain):
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
                 keypoints=19):
        super(LwOpenPose, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap
        self.calc_3d_features = calc_3d_features
        num_heatmap_paf = 3 * keypoints

        with self.init_scope():
            self.encoder = SimpleSequential()
            with self.encoder.init_scope():
                backbone = SimpleSequential()
                with backbone.init_scope():
                    setattr(backbone, "init_block", conv3x3_block(
                        in_channels=in_channels,
                        out_channels=encoder_init_block_channels,
                        stride=2))
                    in_channels = encoder_init_block_channels
                    for i, channels_per_stage in enumerate(encoder_channels):
                        stage = SimpleSequential()
                        with stage.init_scope():
                            for j, out_channels in enumerate(channels_per_stage):
                                stride = 2 if (j == 0) and (i != 0) else 1
                                pad = encoder_paddings[i][j]
                                setattr(stage, "unit{}".format(j + 1), dwsconv3x3_block(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=stride,
                                    pad=pad,
                                    dilate=pad))
                                in_channels = out_channels
                        setattr(backbone, "stage{}".format(i + 1), stage)
                setattr(self.encoder, "backbone", backbone)
                setattr(self.encoder, "final_block", LwopEncoderFinalBlock(
                    in_channels=in_channels,
                    out_channels=encoder_final_block_channels))
                in_channels = encoder_final_block_channels

            self.decoder = SimpleSequential()
            with self.decoder.init_scope():
                setattr(self.decoder, "init_block", LwopDecoderInitBlock(
                    in_channels=in_channels,
                    keypoints=keypoints))
                in_channels = encoder_final_block_channels + num_heatmap_paf
                for i in range(refinement_units):
                    setattr(self.decoder, "unit{}".format(i + 1), LwopDecoderUnit(
                        in_channels=in_channels,
                        keypoints=keypoints))
                setattr(self.decoder, "final_block", LwopDecoderFinalBlock(
                    in_channels=in_channels,
                    keypoints=keypoints,
                    bottleneck_factor=2,
                    calc_3d_features=calc_3d_features))

    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.return_heatmap:
            return x
        else:
            return x


def get_lwopenpose(calc_3d_features,
                   keypoints,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def lwopenpose2d_mobilenet_cmupan_coco(keypoints=19, **kwargs):
    """
    Lightweight OpenPose 2D model on the base of MobileNet for CMU Panoptic from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    keypoints : int, default 19
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose(calc_3d_features=False, keypoints=keypoints, model_name="lwopenpose2d_mobilenet_cmupan_coco",
                          **kwargs)


def lwopenpose3d_mobilenet_cmupan_coco(keypoints=19, **kwargs):
    """
    Lightweight OpenPose 3D model on the base of MobileNet for CMU Panoptic from 'Real-time 2D Multi-Person Pose
    Estimation on CPU: Lightweight OpenPose,' https://arxiv.org/abs/1811.12004.

    Parameters:
    ----------
    keypoints : int, default 19
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose(calc_3d_features=True, keypoints=keypoints, model_name="lwopenpose3d_mobilenet_cmupan_coco",
                          **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (368, 368)
    keypoints = 19
    return_heatmap = True
    pretrained = False

    models = [
        (lwopenpose2d_mobilenet_cmupan_coco, "2d"),
        (lwopenpose3d_mobilenet_cmupan_coco, "3d"),
    ]

    for model, model_dim in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lwopenpose2d_mobilenet_cmupan_coco or weight_count == 4091698)
        assert (model != lwopenpose3d_mobilenet_cmupan_coco or weight_count == 5085983)

        batch = 14
        x = np.random.rand(batch, 3, in_size[0], in_size[1]).astype(np.float32)
        y = net(x)
        if model_dim == "2d":
            assert (y.shape == (batch, 3 * keypoints, in_size[0] // 8, in_size[0] // 8))
        else:
            assert (y.shape == (batch, 6 * keypoints, in_size[0] // 8, in_size[0] // 8))


if __name__ == "__main__":
    _test()
