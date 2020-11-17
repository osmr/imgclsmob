"""
    LFFD for face detection, implemented in TensorFlow.
    Original paper: 'LFFD: A Light and Fast Face Detector for Edge Devices,' https://arxiv.org/abs/1904.10633.
"""

__all__ = ['LFFD', 'lffd20x5s320v2_widerface', 'lffd25x8s560v1_widerface']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv3x3, conv1x1_block, conv3x3_block, Concurrent, MultiOutputSequential, ParallelConcurent,\
    is_channels_first
from .resnet import ResUnit
from .preresnet import PreResUnit


class LffdDetectionBranch(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias,
                 use_bn,
                 data_format="channels_last",
                 **kwargs):
        super(LffdDetectionBranch, self).__init__(**kwargs)
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            activation=None,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class LffdDetectionBlock(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 use_bias,
                 use_bn,
                 data_format="channels_last",
                 **kwargs):
        super(LffdDetectionBlock, self).__init__(**kwargs)
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv")

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.add(LffdDetectionBranch(
            in_channels=mid_channels,
            out_channels=4,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="bbox_branch"))
        self.branches.add(LffdDetectionBranch(
            in_channels=mid_channels,
            out_channels=2,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="score_branch"))

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.branches(x, training=training)
        return x


class LFFD(tf.keras.Model):
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
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (640, 640)
        Spatial size of the expected input image.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 enc_channels,
                 dec_channels,
                 init_block_channels,
                 layers,
                 int_bends,
                 use_preresnet,
                 in_channels=3,
                 in_size=(640, 640),
                 data_format="channels_last",
                 **kwargs):
        super(LFFD, self).__init__(**kwargs)
        self.in_size = in_size
        self.data_format = data_format
        unit_class = PreResUnit if use_preresnet else ResUnit
        use_bias = True
        use_bn = False

        self.encoder = MultiOutputSequential(return_last=False)
        self.encoder.add(conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            strides=2,
            padding=0,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(enc_channels):
            layers_per_stage = layers[i]
            int_bends_per_stage = int_bends[i]
            stage = MultiOutputSequential(multi_output=False, dual_output=True, name="stage{}".format(i + 1))
            stage.add(conv3x3(
                in_channels=in_channels,
                out_channels=channels_per_stage,
                strides=2,
                padding=0,
                use_bias=use_bias,
                data_format=data_format,
                name="trans{}".format(i + 1)))
            for j in range(layers_per_stage):
                unit = unit_class(
                    in_channels=channels_per_stage,
                    out_channels=channels_per_stage,
                    strides=1,
                    use_bias=use_bias,
                    use_bn=use_bn,
                    bottleneck=False,
                    data_format=data_format,
                    name="unit{}".format(j + 1))
                if layers_per_stage - j <= int_bends_per_stage:
                    unit.do_output = True
                stage.add(unit)
            final_activ = nn.ReLU(name="final_activ")
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
                        use_bn=use_bn,
                        data_format=data_format,
                        name="unit{}".format(k + 1)))
                    k += 1
            self.decoder.add(LffdDetectionBlock(
                in_channels=channels_per_stage,
                mid_channels=dec_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                data_format=data_format,
                name="unit{}".format(k + 1)))
            k += 1

    def call(self, x, training=None):
        x = self.encoder(x, training=training)
        x = self.decoder(x, training=training)
        return x


def get_lffd(blocks,
             use_preresnet,
             model_name=None,
             pretrained=False,
             root=os.path.join("~", ".tensorflow", "models"),
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
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if blocks == 20:
        layers = [3, 1, 1, 1, 1]
        enc_channels = [64, 64, 64, 128, 128]
        int_bends = [0, 0, 0, 0, 0]
    elif blocks == 25:
        layers = [4, 2, 1, 3]
        enc_channels = [64, 64, 128, 128]
        int_bends = [1, 1, 0, 2]
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


def lffd20x5s320v2_widerface(**kwargs):
    """
    LFFD-320-20L-5S-V2 model for WIDER FACE from 'LFFD: A Light and Fast Face Detector for Edge Devices,'
    https://arxiv.org/abs/1904.10633.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
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
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_lffd(blocks=25, use_preresnet=False, model_name="lffd25x8s560v1_widerface", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (640, 640)
    pretrained = False

    models = [
        (lffd20x5s320v2_widerface, 5),
        (lffd25x8s560v1_widerface, 8),
    ]

    for model, num_outs in models:

        net = model(pretrained=pretrained)

        batch = 14
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        y = net(x)
        assert (len(y) == num_outs)

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lffd20x5s320v2_widerface or weight_count == 1520606)
        assert (model != lffd25x8s560v1_widerface or weight_count == 2290608)


if __name__ == "__main__":
    _test()
