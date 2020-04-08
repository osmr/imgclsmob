"""
    VOCA for speech-driven facial animation, implemented in Gluon.
    Original paper: 'Capture, Learning, and Synthesis of 3D Speaking Styles,' https://arxiv.org/abs/1905.03079.
"""

__all__ = ['VOCA', 'voca8flame']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import ConvBlock


class VocaEncoder(HybridBlock):
    """
    VOCA encoder.

    Parameters:
    ----------
    audio_features : int
        Number of audio features (characters/sounds).
    audio_window_size : int
        Size of audio window (for time related audio features).
    base_persons : int
        Number of base persons (subjects).
    encoder_features : int
        Number of encoder features.
    """
    def __init__(self,
                 audio_features,
                 audio_window_size,
                 base_persons,
                 encoder_features,
                 **kwargs):
        super(VocaEncoder, self).__init__(**kwargs)
        self.audio_window_size = audio_window_size
        channels = (32, 32, 64, 64)
        fc1_channels = 128

        with self.name_scope():
            self.bn = nn.BatchNorm(in_channels=1)

            in_channels = audio_features + base_persons
            self.branch = nn.HybridSequential(prefix="")
            with self.branch.name_scope():
                for out_channels in channels:
                    self.branch.add(ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 1),
                        strides=(2, 1),
                        padding=(1, 0),
                        use_bias=True,
                        use_bn=False))
                    in_channels = out_channels

            in_channels += base_persons
            self.fc1 = nn.Dense(
                units=fc1_channels,
                in_units=in_channels)
            self.fc2 = nn.Dense(
                units=encoder_features,
                in_units=fc1_channels)

    def hybrid_forward(self, F, x, pid):
        x = self.bn(x)
        x = x.swapaxes(1, 3)
        y = pid.expand_dims(-1).expand_dims(-1)
        y = y.tile(reps=(1, 1, self.audio_window_size, 1))
        x = F.concat(x, y, dim=1)
        x = self.branch(x)
        x = x.flatten()
        x = F.concat(x, pid, dim=1)
        x = self.fc1(x)
        x = x.tanh()
        x = self.fc2(x)
        return x


class VOCA(HybridBlock):
    """
    VOCA model from 'Capture, Learning, and Synthesis of 3D Speaking Styles,' https://arxiv.org/abs/1905.03079.

    Parameters:
    ----------
    audio_features : int, default 29
        Number of audio features (characters/sounds).
    audio_window_size : int, default 16
        Size of audio window (for time related audio features).
    base_persons : int, default 8
        Number of base persons (subjects).
    encoder_features : int, default 50
        Number of encoder features.
    vertices : int, default 5023
        Number of 3D geometry vertices.
    """
    def __init__(self,
                 audio_features=29,
                 audio_window_size=16,
                 base_persons=8,
                 encoder_features=50,
                 vertices=5023,
                 **kwargs):
        super(VOCA, self).__init__(**kwargs)
        self.base_persons = base_persons

        with self.name_scope():
            self.encoder = VocaEncoder(
                audio_features=audio_features,
                audio_window_size=audio_window_size,
                base_persons=base_persons,
                encoder_features=encoder_features)
            self.decoder = nn.Dense(
                units=(3 * vertices),
                in_units=encoder_features)

    def hybrid_forward(self, F, x, pid):
        pid = pid.one_hot(depth=self.base_persons)
        x = self.encoder(x, pid)
        x = self.decoder(x)
        x = x.reshape((0, 1, -1, 3))
        return x


def get_voca(base_persons,
             vertices,
             model_name=None,
             pretrained=False,
             ctx=cpu(),
             root=os.path.join("~", ".mxnet", "models"),
             **kwargs):
    """
    Create VOCA model with specific parameters.

    Parameters:
    ----------
    base_persons : int
        Number of base persons (subjects).
    vertices : int
        Number of 3D geometry vertices.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    net = VOCA(
        base_persons=base_persons,
        vertices=vertices,
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


def voca8flame(**kwargs):
    """
    VOCA-8-FLAME model for 8 base persons and FLAME topology from 'Capture, Learning, and Synthesis of 3D Speaking
    Styles,' https://arxiv.org/abs/1905.03079.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_voca(base_persons=8, vertices=5023, model_name="voca8flame", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        voca8flame,
    ]

    for model in models:

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
        assert (model != voca8flame or weight_count == 809563)

        batch = 14
        audio_features = 29
        audio_window_size = 16
        vertices = 5023

        x = mx.nd.random.normal(shape=(batch, 1, audio_window_size, audio_features), ctx=ctx)
        pid = mx.nd.array(np.full(shape=(batch,), fill_value=3), ctx=ctx)
        y = net(x, pid)
        assert (y.shape == (batch, 1, vertices, 3))


if __name__ == "__main__":
    _test()
