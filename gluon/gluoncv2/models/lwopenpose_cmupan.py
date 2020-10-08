"""
    Lightweight OpenPose 2D/3D for CMU Panoptic, implemented in Gluon.
    Original paper: 'Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose,'
    https://arxiv.org/abs/1811.12004.
"""

__all__ = ['LwOpenPose', 'lwopenpose2d_mobilenet_cmupan_coco', 'lwopenpose3d_mobilenet_cmupan_coco',
           'LwopDecoderFinalBlock']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv1x1_block, conv3x3_block, dwsconv3x3_block


class LwopResBottleneck(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    squeeze_out : bool, default False
        Whether to squeeze the output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_bias=True,
                 bn_use_global_stats=False,
                 bottleneck_factor=2,
                 squeeze_out=False,
                 **kwargs):
        super(LwopResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor if squeeze_out else in_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                use_bias=use_bias,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                activation=None,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LwopResUnit(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
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
                 strides=1,
                 use_bias=True,
                 bn_use_global_stats=False,
                 bottleneck_factor=2,
                 squeeze_out=False,
                 activate=False,
                 **kwargs):
        super(LwopResUnit, self).__init__(**kwargs)
        self.activate = activate
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = LwopResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                use_bias=use_bias,
                bn_use_global_stats=bn_use_global_stats,
                bottleneck_factor=bottleneck_factor,
                squeeze_out=squeeze_out)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    use_bias=use_bias,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            if self.activate:
                self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        if self.activate:
            x = self.activ(x)
        return x


class LwopEncoderFinalBlock(HybridBlock):
    """
    Lightweight OpenPose 2D/3D specific encoder final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(LwopEncoderFinalBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.pre_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=False,
                bn_use_global_stats=bn_use_global_stats)
            self.body = nn.HybridSequential(prefix="")
            for i in range(3):
                self.body.add(dwsconv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dw_use_bn=False,
                    pw_use_bn=False,
                    bn_use_global_stats=bn_use_global_stats,
                    dw_activation=(lambda: nn.ELU()),
                    pw_activation=(lambda: nn.ELU())))
            self.post_conv = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=False,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        x = self.post_conv(x)
        return x


class LwopRefinementBlock(HybridBlock):
    """
    Lightweight OpenPose 2D/3D specific refinement block for decoder units.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(LwopRefinementBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.pre_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=False,
                bn_use_global_stats=bn_use_global_stats)
            self.body = nn.HybridSequential(prefix="")
            self.body.add(conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                use_bias=True,
                bn_use_global_stats=bn_use_global_stats))
            self.body.add(conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                padding=2,
                dilation=2,
                use_bias=True,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.pre_conv(x)
        x = x + self.body(x)
        return x


class LwopDecoderBend(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(LwopDecoderBend, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=True,
                use_bn=False,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LwopDecoderInitBlock(HybridBlock):
    """
    Lightweight OpenPose 2D/3D specific decoder init block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 bn_use_global_stats=False,
                 **kwargs):
        super(LwopDecoderInitBlock, self).__init__(**kwargs)
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        bend_mid_channels = 512

        with self.name_scope():
            self.body = nn.HybridSequential(prefix="")
            for i in range(3):
                self.body.add(conv3x3_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    use_bias=True,
                    use_bn=False,
                    bn_use_global_stats=bn_use_global_stats))
            self.heatmap_bend = LwopDecoderBend(
                in_channels=in_channels,
                mid_channels=bend_mid_channels,
                out_channels=num_heatmap,
                bn_use_global_stats=bn_use_global_stats)
            self.paf_bend = LwopDecoderBend(
                in_channels=in_channels,
                mid_channels=bend_mid_channels,
                out_channels=num_paf,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        y = self.body(x)
        heatmap = self.heatmap_bend(y)
        paf = self.paf_bend(y)
        y = F.concat(x, heatmap, paf, dim=1)
        return y


class LwopDecoderUnit(HybridBlock):
    """
    Lightweight OpenPose 2D/3D specific decoder init.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    keypoints : int
        Number of keypoints.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 bn_use_global_stats=False,
                 **kwargs):
        super(LwopDecoderUnit, self).__init__(**kwargs)
        num_heatmap = keypoints
        num_paf = 2 * keypoints
        self.features_channels = in_channels - num_heatmap - num_paf

        with self.name_scope():
            self.body = nn.HybridSequential(prefix="")
            for i in range(5):
                self.body.add(LwopRefinementBlock(
                    in_channels=in_channels,
                    out_channels=self.features_channels,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = self.features_channels
            self.heatmap_bend = LwopDecoderBend(
                in_channels=self.features_channels,
                mid_channels=self.features_channels,
                out_channels=num_heatmap,
                bn_use_global_stats=bn_use_global_stats)
            self.paf_bend = LwopDecoderBend(
                in_channels=self.features_channels,
                mid_channels=self.features_channels,
                out_channels=num_paf,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        features = F.slice_axis(x, axis=1, begin=0, end=self.features_channels)
        y = self.body(x)
        heatmap = self.heatmap_bend(y)
        paf = self.paf_bend(y)
        y = F.concat(features, heatmap, paf, dim=1)
        return y


class LwopDecoderFeaturesBend(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 **kwargs):
        super(LwopDecoderFeaturesBend, self).__init__(**kwargs)
        with self.name_scope():
            self.body = nn.HybridSequential(prefix="")
            for i in range(2):
                self.body.add(LwopRefinementBlock(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    bn_use_global_stats=bn_use_global_stats))
                in_channels = mid_channels
            self.features_bend = LwopDecoderBend(
                in_channels=mid_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.body(x)
        x = self.features_bend(x)
        return x


class LwopDecoderFinalBlock(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 keypoints,
                 bottleneck_factor,
                 calc_3d_features,
                 bn_use_global_stats=False,
                 **kwargs):
        super(LwopDecoderFinalBlock, self).__init__(**kwargs)
        self.num_heatmap_paf = 3 * keypoints
        self.calc_3d_features = calc_3d_features
        features_out_channels = self.num_heatmap_paf
        features_in_channels = in_channels - features_out_channels

        if self.calc_3d_features:
            with self.name_scope():
                self.body = nn.HybridSequential(prefix="")
                for i in range(5):
                    self.body.add(LwopResUnit(
                        in_channels=in_channels,
                        out_channels=features_in_channels,
                        bottleneck_factor=bottleneck_factor,
                        bn_use_global_stats=bn_use_global_stats))
                    in_channels = features_in_channels
                self.features_bend = LwopDecoderFeaturesBend(
                    in_channels=features_in_channels,
                    mid_channels=features_in_channels,
                    out_channels=features_out_channels,
                    bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        heatmap_paf_2d = F.slice_axis(x, axis=1, begin=-self.num_heatmap_paf, end=None)
        if not self.calc_3d_features:
            return heatmap_paf_2d
        x = self.body(x)
        x = self.features_bend(x)
        y = F.concat(heatmap_paf_2d, x, dim=1)
        return y


class LwOpenPose(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(368, 368),
                 keypoints=19,
                 **kwargs):
        super(LwOpenPose, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap
        self.calc_3d_features = calc_3d_features
        num_heatmap_paf = 3 * keypoints

        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix="")
            backbone = nn.HybridSequential(prefix="")
            backbone.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=encoder_init_block_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = encoder_init_block_channels
            for i, channels_per_stage in enumerate(encoder_channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        padding = encoder_paddings[i][j]
                        stage.add(dwsconv3x3_block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            padding=padding,
                            dilation=padding,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                backbone.add(stage)
            self.encoder.add(backbone)
            self.encoder.add(LwopEncoderFinalBlock(
                in_channels=in_channels,
                out_channels=encoder_final_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = encoder_final_block_channels

            self.decoder = nn.HybridSequential(prefix="")
            self.decoder.add(LwopDecoderInitBlock(
                in_channels=in_channels,
                keypoints=keypoints,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = encoder_final_block_channels + num_heatmap_paf
            for i in range(refinement_units):
                self.decoder.add(LwopDecoderUnit(
                    in_channels=in_channels,
                    keypoints=keypoints,
                    bn_use_global_stats=bn_use_global_stats))
            self.decoder.add(LwopDecoderFinalBlock(
                in_channels=in_channels,
                keypoints=keypoints,
                bottleneck_factor=2,
                calc_3d_features=calc_3d_features,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
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
                   ctx=cpu(),
                   root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_lwopenpose(calc_3d_features=True, keypoints=keypoints, model_name="lwopenpose3d_mobilenet_cmupan_coco",
                          **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lwopenpose2d_mobilenet_cmupan_coco or weight_count == 4091698)
        assert (model != lwopenpose3d_mobilenet_cmupan_coco or weight_count == 5085983)

        batch = 14
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        if model_dim == "2d":
            assert (y.shape == (batch, 3 * keypoints, in_size[0] // 8, in_size[0] // 8))
        else:
            assert (y.shape == (batch, 6 * keypoints, in_size[0] // 8, in_size[0] // 8))


if __name__ == "__main__":
    _test()
