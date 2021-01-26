"""
    DANet for image segmentation, implemented in Chainer.
    Original paper: 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
"""

__all__ = ['DANet', 'danet_resnetd50b_cityscapes', 'danet_resnetd101b_cityscapes']


import os
import chainer.functions as F
from chainer import link
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from chainer.variable import Parameter
from chainer.initializers import _get_initializer
from .common import conv1x1, conv3x3_block
from .resnetd import resnetd50b, resnetd101b


class ScaleBlock(link.Link):
    """
    Simple scale block.

    Parameters:
    ----------
    initial_alpha : obj, default 0
        Initializer for the weights.
    """
    def __init__(self,
                 initial_alpha=0):
        super(ScaleBlock, self).__init__()
        with self.init_scope():
            alpha_initializer = _get_initializer(initial_alpha)
            self.alpha = Parameter(
                initializer=alpha_initializer,
                shape=(1,),
                name="alpha")

    def __call__(self, x):
        return self.alpha.data * x


class PosAttBlock(Chain):
    """
    Position attention block from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
    It captures long-range spatial contextual information.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 8
        Squeeze reduction value.
    """
    def __init__(self,
                 channels,
                 reduction=8):
        super(PosAttBlock, self).__init__()
        mid_channels = channels // reduction

        with self.init_scope():
            self.query_conv = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                use_bias=True)
            self.key_conv = conv1x1(
                in_channels=channels,
                out_channels=mid_channels,
                use_bias=True)
            self.value_conv = conv1x1(
                in_channels=channels,
                out_channels=channels,
                use_bias=True)
            self.scale = ScaleBlock()

    def __call__(self, x):
        batch, channels, height, width = x.shape
        proj_query = self.query_conv(x).reshape((batch, -1, height * width))
        proj_key = self.key_conv(x).reshape((batch, -1, height * width))
        proj_value = self.value_conv(x).reshape((batch, -1, height * width))

        energy = F.batch_matmul(proj_query, proj_key, transa=True)
        w = F.softmax(energy, axis=-1)

        y = F.batch_matmul(proj_value, w, transb=True)
        y = y.reshape((batch, -1, height, width))
        y = self.scale(y) + x
        return y


class ChaAttBlock(Chain):
    """
    Channel attention block from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
    It explicitly models interdependencies between channels.
    """
    def __init__(self):
        super(ChaAttBlock, self).__init__()
        with self.init_scope():
            self.scale = ScaleBlock()

    def __call__(self, x):
        batch, channels, height, width = x.shape
        proj_query = x.reshape((batch, -1, height * width))
        proj_key = x.reshape((batch, -1, height * width))
        proj_value = x.reshape((batch, -1, height * width))

        energy = F.batch_matmul(proj_query, proj_key, transb=True)
        energy_new = F.broadcast_to(F.max(energy, axis=-1, keepdims=True), shape=energy.shape) - energy
        w = F.softmax(energy_new, axis=-1)

        y = F.batch_matmul(w, proj_value)
        y = y.reshape((batch, -1, height, width))
        y = self.scale(y) + x
        return y


class DANetHeadBranch(Chain):
    """
    DANet head branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    pose_att : bool, default True
        Whether to use position attention instead of channel one.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pose_att=True):
        super(DANetHeadBranch, self).__init__()
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            if pose_att:
                self.att = PosAttBlock(mid_channels)
            else:
                self.att = ChaAttBlock()
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels)
            self.conv3 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)
            self.dropout = partial(
                F.dropout,
                ratio=dropout_rate)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.att(x)
        y = self.conv2(x)
        x = self.conv3(y)
        x = self.dropout(x)
        return x, y


class DANetHead(Chain):
    """
    DANet head block.

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
        super(DANetHead, self).__init__()
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        with self.init_scope():
            self.branch_pa = DANetHeadBranch(
                in_channels=in_channels,
                out_channels=out_channels,
                pose_att=True)
            self.branch_ca = DANetHeadBranch(
                in_channels=in_channels,
                out_channels=out_channels,
                pose_att=False)
            self.conv = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)
            self.dropout = partial(
                F.dropout,
                ratio=dropout_rate)

    def __call__(self, x):
        pa_x, pa_y = self.branch_pa(x)
        ca_x, ca_y = self.branch_ca(x)
        y = pa_y + ca_y
        x = self.conv(y)
        x = self.dropout(x)
        return x, pa_x, ca_x


class DANet(Chain):
    """
    DANet model from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.

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
    classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 classes=19):
        super(DANet, self).__init__()
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size

        with self.init_scope():
            self.backbone = backbone
            self.head = DANetHead(
                in_channels=backbone_out_channels,
                out_channels=classes)

    def __call__(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x, _ = self.backbone(x)
        x, y, z = self.head(x)
        x = F.resize_images(x, output_shape=in_size)
        if self.aux:
            y = F.resize_images(y, output_shape=in_size)
            z = F.resize_images(z, output_shape=in_size)
            return x, y, z
        else:
            return x


def get_danet(backbone,
              classes,
              aux=False,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".chainer", "models"),
              **kwargs):
    """
    Create DANet model with specific parameters.

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
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    net = DANet(
        backbone=backbone,
        classes=classes,
        aux=aux,
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


def danet_resnetd50b_cityscapes(pretrained_backbone=False, classes=19, aux=True, **kwargs):
    """
    DANet model on the base of ResNet(D)-50b for Cityscapes from 'Dual Attention Network for Scene Segmentation,'
    https://arxiv.org/abs/1809.02983.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_danet(backbone=backbone, classes=classes, aux=aux, model_name="danet_resnetd50b_cityscapes",
                     **kwargs)


def danet_resnetd101b_cityscapes(pretrained_backbone=False, classes=19, aux=True, **kwargs):
    """
    DANet model on the base of ResNet(D)-101b for Cityscapes from 'Dual Attention Network for Scene Segmentation,'
    https://arxiv.org/abs/1809.02983.

    Parameters:
    ----------
    pretrained_backbone : bool, default False
        Whether to load the pretrained weights for feature extractor.
    classes : int, default 19
        Number of segmentation classes.
    aux : bool, default True
        Whether to output an auxiliary result.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features
    del backbone.final_pool
    return get_danet(backbone=backbone, classes=classes, aux=aux, model_name="danet_resnetd101b_cityscapes",
                     **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (480, 480)
    aux = False
    pretrained = False

    models = [
        danet_resnetd50b_cityscapes,
        danet_resnetd101b_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != danet_resnetd50b_cityscapes or weight_count == 47586427)
        assert (model != danet_resnetd101b_cityscapes or weight_count == 66578555)

        batch = 2
        classes = 19
        x = np.zeros((batch, 3, in_size[0], in_size[1]), np.float32)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
