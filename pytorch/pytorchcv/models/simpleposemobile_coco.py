"""
    SimplePose(Mobile) for COCO Keypoint, implemented in PyTorch.
    Original paper: 'Simple Baselines for Human Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.
"""

__all__ = ['SimplePoseMobile', 'simplepose_mobile_resnet18_coco', 'simplepose_mobile_resnet50b_coco',
           'simplepose_mobile_mobilenet_w1_coco', 'simplepose_mobile_mobilenetv2b_w1_coco',
           'simplepose_mobile_mobilenetv3_small_w1_coco', 'simplepose_mobile_mobilenetv3_large_w1_coco']

import os
import torch
import torch.nn as nn
from .common import conv1x1, DucBlock, HeatmapMaxDetBlock
from .resnet import resnet18, resnet50b
from .mobilenet import mobilenet_w1
from .mobilenetv2 import mobilenetv2b_w1
from .mobilenetv3 import mobilenetv3_small_w1, mobilenetv3_large_w1


class SimplePoseMobile(nn.Module):
    """
    SimplePose(Mobile) model from 'Simple Baselines for Human Pose Estimation and Tracking,'
    https://arxiv.org/abs/1804.06208.

    Parameters:
    ----------
    backbone : nn.Sequential
        Feature extractor.
    backbone_out_channels : int
        Number of output channels for the backbone.
    channels : list of int
        Number of output channels for each decoder unit.
    decoder_init_block_channels : int
        Number of output channels for the initial unit of the decoder.
    return_heatmap : bool, default False
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
                 decoder_init_block_channels,
                 return_heatmap=False,
                 in_channels=3,
                 in_size=(256, 192),
                 keypoints=17):
        super(SimplePoseMobile, self).__init__()
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap

        self.backbone = backbone

        self.decoder = nn.Sequential()
        in_channels = backbone_out_channels
        self.decoder.add_module("init_block", conv1x1(
            in_channels=in_channels,
            out_channels=decoder_init_block_channels))
        in_channels = decoder_init_block_channels
        for i, out_channels in enumerate(channels):
            self.decoder.add_module("unit{}".format(i + 1), DucBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=2))
            in_channels = out_channels
        self.decoder.add_module("final_block", conv1x1(
            in_channels=in_channels,
            out_channels=keypoints))

        self.heatmap_max_det = HeatmapMaxDetBlock()

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        heatmap = self.decoder(x)
        if self.return_heatmap:
            return heatmap
        else:
            keypoints = self.heatmap_max_det(heatmap)
            return keypoints


def get_simpleposemobile(backbone,
                         backbone_out_channels,
                         keypoints,
                         model_name=None,
                         pretrained=False,
                         root=os.path.join("~", ".torch", "models"),
                         **kwargs):
    """
    Create SimplePose(Mobile) model with specific parameters.

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
    channels = [128, 64, 32]
    decoder_init_block_channels = 256

    net = SimplePoseMobile(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
        decoder_init_block_channels=decoder_init_block_channels,
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


def simplepose_mobile_resnet18_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose(Mobile) model on the base of ResNet-18 for COCO Keypoint from 'Simple Baselines for Human Pose Estimation
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
    backbone = resnet18(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simpleposemobile(backbone=backbone, backbone_out_channels=512, keypoints=keypoints,
                                model_name="simplepose_mobile_resnet18_coco", **kwargs)


def simplepose_mobile_resnet50b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose(Mobile) model on the base of ResNet-50b for COCO Keypoint from 'Simple Baselines for Human Pose
    Estimation and Tracking,' https://arxiv.org/abs/1804.06208.

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
    return get_simpleposemobile(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                                model_name="simplepose_mobile_resnet50b_coco", **kwargs)


def simplepose_mobile_mobilenet_w1_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose(Mobile) model on the base of 1.0 MobileNet-224 for COCO Keypoint from 'Simple Baselines for Human Pose
    Estimation and Tracking,' https://arxiv.org/abs/1804.06208.

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
    backbone = mobilenet_w1(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simpleposemobile(backbone=backbone, backbone_out_channels=1024, keypoints=keypoints,
                                model_name="simplepose_mobile_mobilenet_w1_coco", **kwargs)


def simplepose_mobile_mobilenetv2b_w1_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose(Mobile) model on the base of 1.0 MobileNetV2b-224 for COCO Keypoint from 'Simple Baselines for Human Pose
    Estimation and Tracking,' https://arxiv.org/abs/1804.06208.

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
    backbone = mobilenetv2b_w1(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simpleposemobile(backbone=backbone, backbone_out_channels=1280, keypoints=keypoints,
                                model_name="simplepose_mobile_mobilenetv2b_w1_coco", **kwargs)


def simplepose_mobile_mobilenetv3_small_w1_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose(Mobile) model on the base of MobileNetV3 Small 224/1.0 for COCO Keypoint from 'Simple Baselines for Human
    Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.

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
    backbone = mobilenetv3_small_w1(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simpleposemobile(backbone=backbone, backbone_out_channels=576, keypoints=keypoints,
                                model_name="simplepose_mobile_mobilenetv3_small_w1_coco", **kwargs)


def simplepose_mobile_mobilenetv3_large_w1_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    SimplePose(Mobile) model on the base of MobileNetV3 Large 224/1.0 for COCO Keypoint from 'Simple Baselines for Human
    Pose Estimation and Tracking,' https://arxiv.org/abs/1804.06208.

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
    backbone = mobilenetv3_large_w1(pretrained=pretrained_backbone).features
    del backbone[-1]
    return get_simpleposemobile(backbone=backbone, backbone_out_channels=960, keypoints=keypoints,
                                model_name="simplepose_mobile_mobilenetv3_large_w1_coco", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    in_size = (256, 192)
    keypoints = 17
    return_heatmap = False
    pretrained = False

    models = [
        simplepose_mobile_resnet18_coco,
        simplepose_mobile_resnet50b_coco,
        simplepose_mobile_mobilenet_w1_coco,
        simplepose_mobile_mobilenetv2b_w1_coco,
        simplepose_mobile_mobilenetv3_small_w1_coco,
        simplepose_mobile_mobilenetv3_large_w1_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != simplepose_mobile_resnet18_coco or weight_count == 12858208)
        assert (model != simplepose_mobile_resnet50b_coco or weight_count == 25582944)
        assert (model != simplepose_mobile_mobilenet_w1_coco or weight_count == 5019744)
        assert (model != simplepose_mobile_mobilenetv2b_w1_coco or weight_count == 4102176)
        assert (model != simplepose_mobile_mobilenetv3_small_w1_coco or weight_count == 2625088)
        assert (model != simplepose_mobile_mobilenetv3_large_w1_coco or weight_count == 4768336)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == keypoints))
        if return_heatmap:
            assert ((y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4))
        else:
            assert (y.shape[2] == 3)


if __name__ == "__main__":
    _test()
