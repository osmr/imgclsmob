"""
    DANet for image segmentation, implemented in Gluon.
    Original paper: 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
"""

__all__ = ['DANet', 'danet_resnetd50b_cityscapes', 'danet_resnetd101b_cityscapes', 'ScaleBlock']


import os
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv3x3_block
from .resnetd import resnetd50b, resnetd101b


class ScaleBlock(HybridBlock):
    """
    Simple scale block.
    """
    def __init__(self,
                 **kwargs):
        super(ScaleBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get(
                "alpha",
                shape=(1,),
                init=mx.init.Zero(),
                allow_deferred_init=True)

    def hybrid_forward(self, F, x, alpha):
        return F.broadcast_mul(alpha, x)

    def __repr__(self):
        s = '{name}(alpha={alpha})'
        return s.format(
            name=self.__class__.__name__,
            gamma=self.alpha.shape[0])

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        num_flops = x.size
        num_macs = 0
        return num_flops, num_macs


class PosAttBlock(HybridBlock):
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
                 reduction=8,
                 **kwargs):
        super(PosAttBlock, self).__init__(**kwargs)
        mid_channels = channels // reduction

        with self.name_scope():
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

    def hybrid_forward(self, F, x):
        proj_query = self.query_conv(x).reshape((0, 0, -1))
        proj_key = self.key_conv(x).reshape((0, 0, -1))
        proj_value = self.value_conv(x).reshape((0, 0, -1))

        energy = F.batch_dot(proj_query, proj_key, transpose_a=True)
        w = F.softmax(energy)

        y = F.batch_dot(proj_value, w, transpose_b=True)
        y = F.reshape_like(y, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)
        y = self.scale(y) + x
        return y


class ChaAttBlock(HybridBlock):
    """
    Channel attention block from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
    It explicitly models interdependencies between channels.
    """
    def __init__(self,
                 **kwargs):
        super(ChaAttBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.scale = ScaleBlock()

    def hybrid_forward(self, F, x):
        proj_query = x.reshape((0, 0, -1))
        proj_key = x.reshape((0, 0, -1))
        proj_value = x.reshape((0, 0, -1))

        energy = F.batch_dot(proj_query, proj_key, transpose_b=True)
        energy_new = energy.max(axis=-1, keepdims=True).broadcast_like(energy) - energy
        w = F.softmax(energy_new)

        y = F.batch_dot(w, proj_value)
        y = F.reshape_like(y, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)
        y = self.scale(y) + x
        return y


class DANetHeadBranch(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pose_att=True,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DANetHeadBranch, self).__init__(**kwargs)
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            if pose_att:
                self.att = PosAttBlock(mid_channels)
            else:
                self.att = ChaAttBlock()
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv3 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)
            self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.att(x)
        y = self.conv2(x)
        x = self.conv3(y)
        x = self.dropout(x)
        return x, y


class DANetHead(HybridBlock):
    """
    DANet head block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 **kwargs):
        super(DANetHead, self).__init__(**kwargs)
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        with self.name_scope():
            self.branch_pa = DANetHeadBranch(
                in_channels=in_channels,
                out_channels=out_channels,
                pose_att=True,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.branch_ca = DANetHeadBranch(
                in_channels=in_channels,
                out_channels=out_channels,
                pose_att=False,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)
            self.conv = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)
            self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x):
        pa_x, pa_y = self.branch_pa(x)
        ca_x, ca_y = self.branch_ca(x)
        y = pa_y + ca_y
        x = self.conv(y)
        x = self.dropout(x)
        return x, pa_x, ca_x


class DANet(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    bn_cudnn_off : bool, default False
        Whether to disable CUDNN batch normalization operator.
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
                 bn_use_global_stats=False,
                 bn_cudnn_off=False,
                 in_channels=3,
                 in_size=(480, 480),
                 classes=19,
                 **kwargs):
        super(DANet, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size

        with self.name_scope():
            self.backbone = backbone
            self.head = DANetHead(
                in_channels=backbone_out_channels,
                out_channels=classes,
                bn_use_global_stats=bn_use_global_stats,
                bn_cudnn_off=bn_cudnn_off)

    def hybrid_forward(self, F, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        x, _ = self.backbone(x)
        x, y, z = self.head(x)
        x = F.contrib.BilinearResize2D(x, height=in_size[0], width=in_size[1])
        if self.aux:
            y = F.contrib.BilinearResize2D(y, height=in_size[0], width=in_size[1])
            z = F.contrib.BilinearResize2D(z, height=in_size[0], width=in_size[1])
            return x, y, z
        else:
            return x


def get_danet(backbone,
              classes,
              aux=False,
              model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx,
            ignore_extra=True)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,)).features[:-1]
    return get_danet(backbone=backbone, classes=classes, aux=aux, model_name="danet_resnetd101b_cityscapes",
                     **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (480, 480)
    aux = False
    pretrained = False

    models = [
        danet_resnetd50b_cityscapes,
        danet_resnetd101b_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux)

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
        assert (model != danet_resnetd50b_cityscapes or weight_count == 47586427)
        assert (model != danet_resnetd101b_cityscapes or weight_count == 66578555)

        batch = 14
        classes = 19
        x = mx.nd.zeros((batch, 3, in_size[0], in_size[1]), ctx=ctx)
        ys = net(x)
        y = ys[0] if aux else ys
        assert ((y.shape[0] == x.shape[0]) and (y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and
                (y.shape[3] == x.shape[3]))


if __name__ == "__main__":
    _test()
