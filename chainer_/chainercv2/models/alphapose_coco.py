"""
    AlphaPose for COCO Keypoint, implemented in Chainer.
    Original paper: 'RMPE: Regional Multi-person Pose Estimation,' https://arxiv.org/abs/1612.00137.
"""

__all__ = ['AlphaPose', 'alphapose_fastseresnet101b_coco']

import os
from chainer import Chain
from chainer.serializers import load_npz
from .common import conv3x3, PixelShuffle, DucBlock, HeatmapMaxDetBlock, SimpleSequential
from .fastseresnet import fastseresnet101b


class AlphaPose(Chain):
    """
    AlphaPose model from 'RMPE: Regional Multi-person Pose Estimation,' https://arxiv.org/abs/1612.00137.

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
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels,
                 channels,
                 return_heatmap=False,
                 in_channels=3,
                 in_size=(256, 192),
                 keypoints=17,
                 **kwargs):
        super(AlphaPose, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap

        with self.init_scope():
            self.backbone = backbone

            self.decoder = SimpleSequential()
            with self.decoder.init_scope():
                setattr(self.decoder, "init_block", PixelShuffle(scale_factor=2))
                in_channels = backbone_out_channels // 4
                for i, out_channels in enumerate(channels):
                    setattr(self.decoder, "unit{}".format(i + 1), DucBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=2))
                    in_channels = out_channels
                setattr(self.decoder, "final_block", conv3x3(
                    in_channels=in_channels,
                    out_channels=keypoints,
                    use_bias=True))

            self.heatmap_max_det = HeatmapMaxDetBlock()

    def __call__(self, x):
        x = self.backbone(x)
        # return self.decoder.el(0)(x)
        # return self.decoder.el(1)(self.decoder.el(0)(x))
        # print(y[0, 0].array)
        heatmap = self.decoder(x)
        if self.return_heatmap:
            return heatmap
        else:
            keypoints = self.heatmap_max_det(heatmap)
            return keypoints


def get_alphapose(backbone,
                  backbone_out_channels,
                  keypoints,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create AlphaPose model with specific parameters.

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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    channels = [256, 128]

    net = AlphaPose(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        channels=channels,
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


def alphapose_fastseresnet101b_coco(pretrained_backbone=False, keypoints=17, **kwargs):
    """
    AlphaPose model on the base of ResNet-101b for COCO Keypoint from 'RMPE: Regional Multi-person Pose Estimation,'
    https://arxiv.org/abs/1612.00137.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    keypoints : int, default 17
        Number of keypoints.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = fastseresnet101b(pretrained=pretrained_backbone).features
    del backbone.final_pool
    return get_alphapose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                         model_name="alphapose_fastseresnet101b_coco", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (256, 192)
    keypoints = 17
    return_heatmap = False
    pretrained = False

    models = [
        alphapose_fastseresnet101b_coco,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != alphapose_fastseresnet101b_coco or weight_count == 59569873)

        batch = 14
        x = np.random.rand(batch, 3, in_size[0], in_size[1]).astype(np.float32)
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == keypoints))
        if return_heatmap:
            assert ((y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4))
        else:
            assert (y.shape[2] == 3)


if __name__ == "__main__":
    _test()
