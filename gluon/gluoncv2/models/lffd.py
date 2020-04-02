"""
    LFFD for face detection, implemented in Gluon.
    Original paper: 'LFFD: A Light and Fast Face Detector for Edge Devices,' https://arxiv.org/abs/1904.10633.
"""

__all__ = ['LFFD', 'lffd20x5s320v2_widerface', 'lffd25x8s560v1_widerface']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import conv3x3, conv1x1_block, conv3x3_block, MultiOutputSequential, ParallelConcurent
from .resnet import ResUnit
from .preresnet import PreResUnit


class LffdDetectionBranch(HybridBlock):
    """
    LFFD specific detection branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias,
                 use_bn,
                 **kwargs):
        super(LffdDetectionBranch, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=in_channels,
                use_bias=use_bias,
                use_bn=use_bn)
            self.conv2 = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LffdDetectionBlock(HybridBlock):
    """
    LFFD specific detection block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    use_bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 use_bias,
                 use_bn,
                 **kwargs):
        super(LffdDetectionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias,
                use_bn=use_bn)
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(LffdDetectionBranch(
                in_channels=mid_channels,
                out_channels=4,
                use_bias=use_bias,
                use_bn=use_bn))
            self.branches.add(LffdDetectionBranch(
                in_channels=mid_channels,
                out_channels=2,
                use_bias=use_bias,
                use_bn=use_bn))

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.branches(x)
        return x


class LFFD(HybridBlock):
    """
    LFFD model from 'LFFD: A Light and Fast Face Detector for Edge Devices,' https://arxiv.org/abs/1904.10633.

    Parameters:
    ----------
    enc_channels : list of int
        Number of output channels for each encoder stage.
    dec_channels : int
        Number of output channels for each decoder stage.
    init_block_channels : int
        Number of output channels for the initial encoder unit.
    layers : list of int
        Number of units in each encoder stage.
    int_bends : list of int
        Number of internal bends for each encoder stage.
    use_preresnet : bool
        Whether to use PreResnet backbone instead of ResNet.
    receptive_field_center_starts : list of int
        The start location of the first receptive field of each scale.
    receptive_field_strides : list of int
        Receptive field stride for each scale.
    bbox_factors : list of float
        A half of bbox upper bound for each scale.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (640, 640)
        Spatial size of the expected input image.
    """
    def __init__(self,
                 enc_channels,
                 dec_channels,
                 init_block_channels,
                 layers,
                 int_bends,
                 use_preresnet,
                 receptive_field_center_starts,
                 receptive_field_strides,
                 bbox_factors,
                 in_channels=3,
                 in_size=(480, 640),
                 **kwargs):
        super(LFFD, self).__init__(**kwargs)
        self.in_size = in_size
        self.receptive_field_center_starts = receptive_field_center_starts
        self.receptive_field_strides = receptive_field_strides
        self.bbox_factors = bbox_factors
        unit_class = PreResUnit if use_preresnet else ResUnit
        use_bias = True
        use_bn = False

        with self.name_scope():
            self.encoder = MultiOutputSequential(return_last=False)
            self.encoder.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                strides=2,
                padding=0,
                use_bias=use_bias,
                use_bn=use_bn))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(enc_channels):
                layers_per_stage = layers[i]
                int_bends_per_stage = int_bends[i]
                stage = MultiOutputSequential(prefix="stage{}_".format(i + 1), multi_output=False, dual_output=True)
                stage.add(conv3x3(
                    in_channels=in_channels,
                    out_channels=channels_per_stage,
                    strides=2,
                    padding=0,
                    use_bias=use_bias))
                for j in range(layers_per_stage):
                    unit = unit_class(
                        in_channels=channels_per_stage,
                        out_channels=channels_per_stage,
                        strides=1,
                        use_bias=use_bias,
                        use_bn=use_bn,
                        bottleneck=False)
                    if layers_per_stage - j <= int_bends_per_stage:
                        unit.do_output = True
                    stage.add(unit)
                final_activ = nn.Activation("relu")
                final_activ.do_output = True
                stage.add(final_activ)
                stage.do_output2 = True
                in_channels = channels_per_stage
                self.encoder.add(stage)

            self.decoder = ParallelConcurent()
            k = 0
            for i, channels_per_stage in enumerate(enc_channels):
                layers_per_stage = layers[i]
                int_bends_per_stage = int_bends[i]
                for j in range(layers_per_stage):
                    if layers_per_stage - j <= int_bends_per_stage:
                        self.decoder.add(LffdDetectionBlock(
                            in_channels=channels_per_stage,
                            mid_channels=dec_channels,
                            use_bias=use_bias,
                            use_bn=use_bn))
                        k += 1
                self.decoder.add(LffdDetectionBlock(
                    in_channels=channels_per_stage,
                    mid_channels=dec_channels,
                    use_bias=use_bias,
                    use_bn=use_bn))
                k += 1

    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_lffd(blocks,
             use_preresnet,
             model_name=None,
             pretrained=False,
             ctx=cpu(),
             root=os.path.join("~", ".mxnet", "models"),
             **kwargs):
    """
    Create LFFD model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    use_preresnet : bool
        Whether to use PreResnet backbone instead of ResNet.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if blocks == 20:
        layers = [3, 1, 1, 1, 1]
        enc_channels = [64, 64, 64, 128, 128]
        int_bends = [0, 0, 0, 0, 0]
        receptive_field_center_starts = [3, 7, 15, 31, 63]
        receptive_field_strides = [4, 8, 16, 32, 64]
        bbox_factors = [10.0, 20.0, 40.0, 80.0, 160.0]
    elif blocks == 25:
        layers = [4, 2, 1, 3]
        enc_channels = [64, 64, 128, 128]
        int_bends = [1, 1, 0, 2]
        receptive_field_center_starts = [3, 3, 7, 7, 15, 31, 31, 31]
        receptive_field_strides = [4, 4, 8, 8, 16, 32, 32, 32]
        bbox_factors = [7.5, 10.0, 20.0, 35.0, 55.0, 125.0, 200.0, 280.0]
    else:
        raise ValueError("Unsupported LFFD with number of blocks: {}".format(blocks))

    dec_channels = 128
    init_block_channels = 64

    net = LFFD(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        init_block_channels=init_block_channels,
        layers=layers,
        int_bends=int_bends,
        use_preresnet=use_preresnet,
        receptive_field_center_starts=receptive_field_center_starts,
        receptive_field_strides=receptive_field_strides,
        bbox_factors=bbox_factors,
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


def lffd20x5s320v2_widerface(**kwargs):
    """
    LFFD-320-20L-5S-V2 model for WIDER FACE from 'LFFD: A Light and Fast Face Detector for Edge Devices,'
    https://arxiv.org/abs/1904.10633.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_lffd(blocks=20, use_preresnet=True, model_name="lffd20x5s320v2_widerface", **kwargs)


def lffd25x8s560v1_widerface(**kwargs):
    """
    LFFD-560-25L-8S-V1 model for WIDER FACE from 'LFFD: A Light and Fast Face Detector for Edge Devices,'
    https://arxiv.org/abs/1904.10633.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_lffd(blocks=25, use_preresnet=False, model_name="lffd25x8s560v1_widerface", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    in_size = (480, 640)
    pretrained = False

    models = [
        (lffd20x5s320v2_widerface, 5),
        (lffd25x8s560v1_widerface, 8),
    ]

    for model, num_outs in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lffd20x5s320v2_widerface or weight_count == 1520606)
        assert (model != lffd25x8s560v1_widerface or weight_count == 2290608)

        batch = 14
        x = mx.nd.random.normal(shape=(batch, 3, in_size[0], in_size[1]), ctx=ctx)
        y = net(x)
        assert (len(y) == num_outs)


if __name__ == "__main__":
    _test()
