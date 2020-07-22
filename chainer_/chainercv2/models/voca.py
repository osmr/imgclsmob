"""
    VOCA for speech-driven facial animation, implemented in Chainer.
    Original paper: 'Capture, Learning, and Synthesis of 3D Speaking Styles,' https://arxiv.org/abs/1905.03079.
"""

__all__ = ['VOCA', 'voca8flame']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer.serializers import load_npz
from .common import ConvBlock, SimpleSequential


class VocaEncoder(Chain):
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
                 encoder_features):
        super(VocaEncoder, self).__init__()
        self.audio_window_size = audio_window_size
        channels = (32, 32, 64, 64)
        fc1_channels = 128

        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=1,
                eps=1e-5)

            in_channels = audio_features + base_persons
            self.branch = SimpleSequential()
            with self.branch.init_scope():
                for i, out_channels in enumerate(channels):
                    setattr(self.branch, "conv{}".format(i + 1), ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksize=(3, 1),
                        stride=(2, 1),
                        pad=(1, 0),
                        use_bias=True,
                        use_bn=False))
                    in_channels = out_channels

            in_channels += base_persons
            self.fc1 = L.Linear(
                in_size=in_channels,
                out_size=fc1_channels)
            self.fc2 = L.Linear(
                in_size=fc1_channels,
                out_size=encoder_features)

    def __call__(self, x, pid):
        x = self.bn(x)
        x = F.swapaxes(x, axis1=1, axis2=3)
        y = F.expand_dims(F.expand_dims(pid, axis=-1), axis=-1)
        y = F.tile(y, reps=(1, 1, self.audio_window_size, 1))
        x = F.concat((x, y), axis=1)
        x = self.branch(x)
        x = F.reshape(x, shape=(x.shape[0], -1))
        x = F.concat((x, pid), axis=1)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x


class VOCA(Chain):
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
                 vertices=5023):
        super(VOCA, self).__init__()
        self.base_persons = base_persons

        with self.init_scope():
            self.encoder = VocaEncoder(
                audio_features=audio_features,
                audio_window_size=audio_window_size,
                base_persons=base_persons,
                encoder_features=encoder_features)
            self.decoder = L.Linear(
                in_size=encoder_features,
                out_size=(3 * vertices))

    def __call__(self, x, pid):
        pid = self.xp.eye(self.base_persons, dtype=pid.dtype)[pid.astype("int32")]
        x = self.encoder(x, pid)
        x = self.decoder(x)
        x = F.reshape(x, shape=(x.shape[0], 1, -1, 3))
        return x


def get_voca(base_persons,
             vertices,
             model_name=None,
             pretrained=False,
             root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def voca8flame(**kwargs):
    """
    VOCA-8-FLAME model for 8 base persons and FLAME topology from 'Capture, Learning, and Synthesis of 3D Speaking
    Styles,' https://arxiv.org/abs/1905.03079.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_voca(base_persons=8, vertices=5023, model_name="voca8flame", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        voca8flame,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != voca8flame or weight_count == 809563)

        batch = 14
        audio_features = 29
        audio_window_size = 16
        vertices = 5023

        x = np.random.rand(batch, 1, audio_window_size, audio_features).astype(np.float32)
        pid = np.full(shape=(batch,), fill_value=3, dtype=np.float32)
        y = net(x, pid)
        assert (y.shape == (batch, 1, vertices, 3))


if __name__ == "__main__":
    _test()
