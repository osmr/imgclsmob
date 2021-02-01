"""
    AlphaPose for COCO Keypoint, implemented in TensorFlow.
    Original paper: 'RMPE: Regional Multi-person Pose Estimation,' https://arxiv.org/abs/1612.00137.
"""

__all__ = ['AlphaPose', 'alphapose_fastseresnet101b_coco']

import os
import tensorflow as tf
from .common import conv3x3, PixelShuffle, DucBlock, HeatmapMaxDetBlock, SimpleSequential, is_channels_first
from .fastseresnet import fastseresnet101b


class AlphaPose(tf.keras.Model):
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
        super(AlphaPose, self).__init__(**kwargs)
        assert (in_channels == 3)
        self.in_size = in_size
        self.keypoints = keypoints
        self.return_heatmap = return_heatmap
        self.data_format = data_format

        self.backbone = backbone
        self.backbone._name = "backbone"

        self.decoder = SimpleSequential(name="decoder")
        self.decoder.add(PixelShuffle(
            scale_factor=2,
            data_format=data_format,
            name="init_block"))
        in_channels = backbone_out_channels // 4
        for i, out_channels in enumerate(channels):
            self.decoder.add(DucBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=2,
                data_format=data_format,
                name="unit{}".format(i + 1)))
            in_channels = out_channels
        self.decoder.add(conv3x3(
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


def get_alphapose(backbone,
                  backbone_out_channels,
                  keypoints,
                  model_name=None,
                  data_format="channels_last",
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    channels = [256, 128]

    net = AlphaPose(
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


def alphapose_fastseresnet101b_coco(pretrained_backbone=False, keypoints=17, data_format="channels_last", **kwargs):
    """
    AlphaPose model on the base of ResNet-101b for COCO Keypoint from 'RMPE: Regional Multi-person Pose Estimation,'
    https://arxiv.org/abs/1612.00137.

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
    backbone = fastseresnet101b(pretrained=pretrained_backbone, data_format=data_format).features
    del backbone.children[-1]
    return get_alphapose(backbone=backbone, backbone_out_channels=2048, keypoints=keypoints,
                         model_name="alphapose_fastseresnet101b_coco", data_format=data_format, **kwargs)


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
        alphapose_fastseresnet101b_coco,
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
        assert (model != alphapose_fastseresnet101b_coco or weight_count == 59569873)


if __name__ == "__main__":
    _test()
