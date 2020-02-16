"""
    SimplePose for COCO Keypoint, implemented in Gluon.
    Original paper: 'Simple Baselines for Human Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.
"""

__all__ = ['SimplePose', 'simplepose_resnet18_coco', 'simplepose_resnet50b_coco', 'simplepose_resnet101b_coco',
           'simplepose_resnet152b_coco', 'simplepose_resneta50b_coco', 'simplepose_resneta101b_coco',
           'simplepose_resneta152b_coco', 'HeatmapMaxDetBlock']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import get_activation_layer, BatchNormExtra, conv1x1
from .resnet import resnet18, resnet50b, resnet101b, resnet152b
from .resneta import resneta50b, resneta101b, resneta152b


class DeconvBlock(HybridBlock):
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
    out_padding : int or tuple/list of 2 int
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    activation : function or str or None, default nn.Activation('relu')
        Activation function or name of activation function.
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
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(DeconvBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.name_scope():
            self.conv = nn.Conv2DTranspose(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                output_padding=out_padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                in_channels=in_channels)
            if self.use_bn:
                self.bn = BatchNormExtra(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats,
                    cudnn_off=bn_cudnn_off)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class HeatmapMaxDetBlock(HybridBlock):
    """
    Heatmap maximum detector block.

    Parameters:
    ----------
    channels : int
        Number of channels.
    in_size : tuple of 2 int
        Spatial size of the input heatmap tensor.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    """
    def __init__(self,
                 channels,
                 in_size,
                 fixed_size,
                 **kwargs):
        super(HeatmapMaxDetBlock, self).__init__(**kwargs)
        self.channels = channels
        self.in_size = in_size
        self.fixed_size = fixed_size

    def hybrid_forward(self, F, x):
        heatmap = x
        vector_dim = 2
        batch = heatmap.shape[0]
        in_size = self.in_size if self.fixed_size else heatmap[2:]
        heatmap_vector = heatmap.reshape((0, 0, -3))
        indices = heatmap_vector.argmax(axis=vector_dim, keepdims=True)
        scores = heatmap_vector.max(axis=vector_dim, keepdims=True)
        scores_mask = (scores > 0.0)
        pts_x = (indices % in_size[1]) * scores_mask
        pts_y = (indices / in_size[1]).floor() * scores_mask
        pts = F.concat(pts_x, pts_y, scores, dim=vector_dim)
        for b in range(batch):
            for k in range(self.channels):
                hm = heatmap[b, k, :, :]
                px = int(pts[b, k, 0].asscalar())
                py = int(pts[b, k, 1].asscalar())
                if (0 < px < in_size[1] - 1) and (0 < py < in_size[0] - 1):
                    pts[b, k, 0] += (hm[py, px + 1] - hm[py, px - 1]).sign() * 0.25
                    pts[b, k, 1] += (hm[py + 1, px] - hm[py - 1, px]).sign() * 0.25
        return pts

    def __repr__(self):
        s = '{name}(channels={channels}, in_size={in_size}, fixed_size={fixed_size})'
        return s.format(
            name=self.__class__.__name__,
            channels=self.channels,
            in_size=self.in_size,
            fixed_size=self.fixed_size)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        num_flops = x.size + 26 * self.channels
        num_macs = 0
        return num_flops, num_macs


class SimplePose(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    return_heatmap : bool, default False
        Whether to return only heatmap.
    fixed_size : bool, default True
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 192)
        Spatial size of the expected input image.
    keypoints : int, default 17
        Number of keypoints.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels,
                 channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=True,
                 return_heatmap=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(256, 192),
                 keypoints=17,
                 **kwargs):
        super(SimplePose, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap

        with self.name_scope():
            self.backbone = backbone

            self.decoder = nn.HybridSequential(prefix="")
            in_channels = backbone_out_channels
            for out_channels in channels:
                self.decoder.add(DeconvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    strides=2,
                    padding=1,
                    bn_use_global_stats=bn_use_global_stats,
                    bn_cudnn_off=bn_cudnn_off))
                in_channels = out_channels
            self.decoder.add(conv1x1(
                in_channels=in_channels,
                out_channels=keypoints,
                use_bias=True))

            self.heatmap_max_det = HeatmapMaxDetBlock(
                channels=keypoints,
                in_size=(in_size[0] // 4, in_size[1] // 4),
                fixed_size=fixed_size)

    def hybrid_forward(self, F, x):
        x = self.backbone(x)
        heatmap = self.decoder(x)
        if self.return_heatmap:
            return heatmap
        else:
            keypoints = self.heatmap_max_det(heatmap)
            return keypoints


def get_simplepose(backbone,
                   backbone_out_channels,
                   keypoints,
                   bn_cudnn_off,
                   model_name=None,
                   pretrained=False,
                   ctx=cpu(),
                   root=os.path.join("~", ".mxnet", "models"),
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
    bn_cudnn_off : bool
        Whether to disable CUDNN batch normalization operator.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    channels = [256, 256, 256]

    net = SimplePose(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
        bn_cudnn_off=bn_cudnn_off,
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


def simplepose_resnet18_coco(pretrained_backbone=False, keypoints=17, bn_cudnn_off=True, **kwargs):
    """
    SimplePose model on the base of ResNet-18 for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone, bn_cudnn_off=bn_cudnn_off).features[:-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=512, keypoints=keypoints, bn_cudnn_off=bn_cudnn_off,
                          model_name="simplepose_resnet18_coco", **kwargs)


def simplepose_resnet50b_coco(pretrained_backbone=False, keypoints=17, bn_cudnn_off=True, **kwargs):
    """
    SimplePose model on the base of ResNet-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone, bn_cudnn_off=bn_cudnn_off).features[:-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints, bn_cudnn_off=bn_cudnn_off,
                          model_name="simplepose_resnet50b_coco", **kwargs)


def simplepose_resnet101b_coco(pretrained_backbone=False, keypoints=17, bn_cudnn_off=True, **kwargs):
    """
    SimplePose model on the base of ResNet-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone, bn_cudnn_off=bn_cudnn_off).features[:-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints, bn_cudnn_off=bn_cudnn_off,
                          model_name="simplepose_resnet101b_coco", **kwargs)


def simplepose_resnet152b_coco(pretrained_backbone=False, keypoints=17, bn_cudnn_off=True, **kwargs):
    """
    SimplePose model on the base of ResNet-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnet152b(pretrained=pretrained_backbone, bn_cudnn_off=bn_cudnn_off).features[:-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints, bn_cudnn_off=bn_cudnn_off,
                          model_name="simplepose_resnet152b_coco", **kwargs)


def simplepose_resneta50b_coco(pretrained_backbone=False, keypoints=17, bn_cudnn_off=True, **kwargs):
    """
    SimplePose model on the base of ResNet(A)-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resneta50b(pretrained=pretrained_backbone, bn_cudnn_off=bn_cudnn_off).features[:-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints, bn_cudnn_off=bn_cudnn_off,
                          model_name="simplepose_resneta50b_coco", **kwargs)


def simplepose_resneta101b_coco(pretrained_backbone=False, keypoints=17, bn_cudnn_off=True, **kwargs):
    """
    SimplePose model on the base of ResNet(A)-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resneta101b(pretrained=pretrained_backbone, bn_cudnn_off=bn_cudnn_off).features[:-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints, bn_cudnn_off=bn_cudnn_off,
                          model_name="simplepose_resneta101b_coco", **kwargs)


def simplepose_resneta152b_coco(pretrained_backbone=False, keypoints=17, bn_cudnn_off=True, **kwargs):
    """
    SimplePose model on the base of ResNet(A)-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    bn_cudnn_off : bool, default True
        Whether to disable CUDNN batch normalization operator.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resneta152b(pretrained=pretrained_backbone, bn_cudnn_off=bn_cudnn_off).features[:-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints, bn_cudnn_off=bn_cudnn_off,
                          model_name="simplepose_resneta152b_coco", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (256, 192)
    keypoints = 17
    return_heatmap = True
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
        assert (model != simplepose_resnet18_coco or weight_count == 15376721)
        assert (model != simplepose_resnet50b_coco or weight_count == 33999697)
        assert (model != simplepose_resnet101b_coco or weight_count == 52991825)
        assert (model != simplepose_resnet152b_coco or weight_count == 68635473)
        assert (model != simplepose_resneta50b_coco or weight_count == 34018929)
        assert (model != simplepose_resneta101b_coco or weight_count == 53011057)
        assert (model != simplepose_resneta152b_coco or weight_count == 68654705)

        batch = 14
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == keypoints))
        if return_heatmap:
            assert ((y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4))
        else:
            assert (y.shape[2] == 3)


if __name__ == "__main__":
    _test()
