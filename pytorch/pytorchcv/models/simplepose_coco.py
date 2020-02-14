"""
    SimplePose for COCO Keypoint, implemented in PyTorch.
    Original paper: 'Simple Baselines for Human Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.
"""

__all__ = ['SimplePose', 'simplepose_resnet18_coco', 'simplepose_resnet50b_coco', 'simplepose_resnet101b_coco',
           'simplepose_resnet152b_coco', 'simplepose_resneta50b_coco', 'simplepose_resneta101b_coco',
           'simplepose_resneta152b_coco']

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from .common import get_activation_layer, conv1x1
from .resnet import resnet18, resnet50b, resnet101b, resnet152b
from .resneta import resneta50b, resneta101b, resneta152b


def calc_keypoints_with_max_scores(heatmap):
    batch = heatmap.shape[0]
    num_keypoints = heatmap.shape[1]
    height = heatmap.shape[2]
    width = heatmap.shape[3]

    heatmap_vector = heatmap.view(batch, num_keypoints, -1)

    vector_dim = 2
    scores, indices = heatmap_vector.max(dim=vector_dim, keepdims=True)

    scores_mask = (scores > 0.0).int()

    keys_x = (indices % width) * scores_mask
    keys_y = (indices // width) * scores_mask

    keypoints = torch.cat((keys_x, keys_y), dim=vector_dim).float().cpu().detach().numpy()
    for b in range(batch):
        for k in range(num_keypoints):
            hm = heatmap[b, k, :, :]
            px = int(keys_x[b, k])
            py = int(keys_y[b, k])
            if (1 < px < width - 1) and (1 < py < height - 1):
                diff_x = float(hm[py, px + 1] - hm[py, px - 1])
                diff_y = float(hm[py + 1, px] - hm[py - 1, px])
                keypoints[b, k, 0] += np.sign(diff_x) * 0.25
                keypoints[b, k, 1] += np.sign(diff_y) * 0.25

    return keypoints, scores, batch, height, width


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape, np.float32)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def _calc_pose(heatmap, center, scale):
    center_ = center.cpu().detach().numpy()
    scale_ = scale.cpu().detach().numpy()

    keypoints, scores, batch, height, width = calc_keypoints_with_max_scores(heatmap)

    preds = np.zeros_like(keypoints)

    # Transform back
    for i in range(keypoints.shape[0]):
        preds[i] = transform_preds(keypoints[i], center_[i], scale_[i], [width, height])

    return preds, scores


class DeconvBlock(nn.Module):
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
    stride : int or tuple/list of 2 int
        Strides of the deconvolution.
    padding : int or tuple/list of 2 int
        Padding value for deconvolution layer.
    out_padding : int or tuple/list of 2 int
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for deconvolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 out_padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(DeconvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class SimplePose(nn.Module):
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
    return_heatmap_only : bool, default False
        Whether to return only heatmap.
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
                 return_heatmap_only=False,
                 in_channels=3,
                 in_size=(256, 192),
                 keypoints=17):
        super(SimplePose, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap_only = return_heatmap_only
        self.out_size = (in_size[0] // 4, in_size[1] // 4)

        self.backbone = backbone

        self.decoder = nn.Sequential()
        in_channels = backbone_out_channels
        for i, out_channels in enumerate(channels):
            self.decoder.add_module("unit{}".format(i + 1), DeconvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1))
            in_channels = out_channels

        self.final_block = conv1x1(
            in_channels=in_channels,
            out_channels=keypoints,
            bias=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        heatmap = self.final_block(x)
        if self.return_heatmap_only:
            return heatmap

        return heatmap
        # heatmap_vector = heatmap.reshape((0, 0, -3))
        # indices = heatmap_vector.argmax(axis=2, keepdims=True)
        # scores = heatmap_vector.max(axis=2, keepdims=True)
        # keys_x = indices % self.out_size[1]
        # keys_y = (indices / self.out_size[1]).floor()
        # keypoints = F.concat(keys_x, keys_y, dim=2)
        # keypoints = F.broadcast_mul(keypoints, scores.clip(0.0, 1.0e5))
        #
        # return heatmap, keypoints, scores

    @staticmethod
    def calc_pose(heatmap, center, scale):
        return _calc_pose(heatmap, center, scale)


def get_simplepose(backbone,
                   backbone_out_channels,
                   keypoints,
                   model_name=None,
                   pretrained=False,
                   root=os.path.join("~", ".torch", "models"),
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
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    channels = [256, 256, 256]

    net = SimplePose(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
        keypoints=keypoints,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def simplepose_resnet18_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose model on the base of ResNet-18 for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=512, keypoints=keypoints,
                          model_name="simplepose_resnet18_coco", **kwargs)


def simplepose_resnet50b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose model on the base of ResNet-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation and
    Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet50b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resnet50b_coco", **kwargs)


def simplepose_resnet101b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose model on the base of ResNet-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet101b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resnet101b_coco", **kwargs)


def simplepose_resnet152b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose model on the base of ResNet-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resnet152b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resnet152b_coco", **kwargs)


def simplepose_resneta50b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose model on the base of ResNet(A)-50b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resneta50b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resneta50b_coco", **kwargs)


def simplepose_resneta101b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose model on the base of ResNet(A)-101b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resneta101b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resneta101b_coco", **kwargs)


def simplepose_resneta152b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose model on the base of ResNet(A)-152b for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
    and Tracking,' https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    backbone = resneta152b(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simplepose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                          model_name="simplepose_resneta152b_coco", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    in_size = (256, 192)
    keypoints = 17
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

        net = model(pretrained=pretrained, in_size=in_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != simplepose_resnet18_coco or weight_count == 15376721)
        assert (model != simplepose_resnet50b_coco or weight_count == 33999697)
        assert (model != simplepose_resnet101b_coco or weight_count == 52991825)
        assert (model != simplepose_resnet152b_coco or weight_count == 68635473)
        assert (model != simplepose_resneta50b_coco or weight_count == 34018929)
        assert (model != simplepose_resneta101b_coco or weight_count == 53011057)
        assert (model != simplepose_resneta152b_coco or weight_count == 68654705)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == keypoints) and (y.shape[2] == x.shape[2] // 4) and
                (y.shape[3] == x.shape[3] // 4))

        center = torch.zeros((batch, 2))
        scale = torch.ones((batch, 2))
        z, _ = net.calc_pose(y, center, scale)
        assert (z.shape[0] == batch)


if __name__ == "__main__":
    _test()
