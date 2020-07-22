"""
    Neural Voice Puppetry Audio-to-Expression net for speech-driven facial animation, implemented in Chainer.
    Original paper: 'Neural Voice Puppetry: Audio-driven Facial Reenactment,' https://arxiv.org/abs/1912.05566.
"""

__all__ = ['NvpAttExp', 'nvpattexp116bazel76']

import os
from functools import partial
import chainer.functions as F
from chainer import Chain
from chainer.serializers import load_npz
from .common import DenseBlock, ConvBlock, ConvBlock1d, SelectableDense, SimpleSequential


class NvpAttExpEncoder(Chain):
    """
    Neural Voice Puppetry Audio-to-Expression encoder.

    Parameters:
    ----------
    audio_features : int
        Number of audio features (characters/sounds).
    audio_window_size : int
        Size of audio window (for time related audio features).
    seq_len : int, default
        Size of feature window.
    encoder_features : int
        Number of encoder features.
    """
    def __init__(self,
                 audio_features,
                 audio_window_size,
                 seq_len,
                 encoder_features,
                 **kwargs):
        super(NvpAttExpEncoder, self).__init__(**kwargs)
        self.audio_features = audio_features
        self.audio_window_size = audio_window_size
        self.seq_len = seq_len
        conv_channels = (32, 32, 64, 64)
        conv_slopes = (0.02, 0.02, 0.2, 0.2)
        fc_channels = (128, 64, encoder_features)
        fc_slopes = (0.02, 0.02, None)
        att_conv_channels = (16, 8, 4, 2, 1)
        att_conv_slopes = 0.02

        with self.init_scope():
            in_channels = audio_features
            self.conv_branch = SimpleSequential()
            with self.conv_branch.init_scope():
                for i, (out_channels, slope) in enumerate(zip(conv_channels, conv_slopes)):
                    setattr(self.conv_branch, "conv{}".format(i + 1), ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksize=(3, 1),
                        stride=(2, 1),
                        pad=(1, 0),
                        use_bias=True,
                        use_bn=False,
                        activation=partial(F.leaky_relu, slope=slope)))
                    in_channels = out_channels

            self.fc_branch = SimpleSequential()
            with self.fc_branch.init_scope():
                for i, (out_channels, slope) in enumerate(zip(fc_channels, fc_slopes)):
                    activation = partial(F.leaky_relu, slope=slope) if slope is not None else partial(F.tanh)
                    setattr(self.fc_branch, "fc{}".format(i + 1), DenseBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        use_bias=True,
                        use_bn=False,
                        activation=activation))
                    in_channels = out_channels

            self.att_conv_branch = SimpleSequential()
            with self.att_conv_branch.init_scope():
                for i, out_channels, in enumerate(att_conv_channels):
                    setattr(self.att_conv_branch, "att_conv{}".format(i + 1), ConvBlock1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ksize=3,
                        stride=1,
                        pad=1,
                        use_bias=True,
                        use_bn=False,
                        activation=partial(F.leaky_relu, slope=att_conv_slopes)))
                    in_channels = out_channels

            self.att_fc = DenseBlock(
                in_channels=seq_len,
                out_channels=seq_len,
                use_bias=True,
                use_bn=False,
                activation=partial(F.softmax, axis=1))

    def __call__(self, x):
        batch = x.shape[0]
        batch_seq_len = batch * self.seq_len

        x = F.reshape(x, shape=(batch_seq_len, 1, self.audio_window_size, self.audio_features))
        x = F.swapaxes(x, axis1=1, axis2=3)
        x = self.conv_branch(x)
        x = F.reshape(x, shape=(batch_seq_len, 1, -1))
        x = self.fc_branch(x)
        x = F.reshape(x, shape=(batch, self.seq_len, -1))
        x = F.swapaxes(x, axis1=1, axis2=2)

        y = x[:, :, (self.seq_len // 2)]

        w = self.att_conv_branch(x)
        w = F.reshape(w, shape=(batch, self.seq_len))
        w = self.att_fc(w)
        w = F.expand_dims(w, axis=-1)
        x = F.batch_matmul(x, w)
        x = F.squeeze(x, axis=-1)

        return x, y


class NvpAttExp(Chain):
    """
    Neural Voice Puppetry Audio-to-Expression model from 'Neural Voice Puppetry: Audio-driven Facial Reenactment,'
    https://arxiv.org/abs/1912.05566.

    Parameters:
    ----------
    audio_features : int, default 29
        Number of audio features (characters/sounds).
    audio_window_size : int, default 16
        Size of audio window (for time related audio features).
    seq_len : int, default 8
        Size of feature window.
    base_persons : int, default 116
        Number of base persons (identities).
    blendshapes : int, default 76
        Number of 3D model blendshapes.
    encoder_features : int, default 32
        Number of encoder features.
    """
    def __init__(self,
                 audio_features=29,
                 audio_window_size=16,
                 seq_len=8,
                 base_persons=116,
                 blendshapes=76,
                 encoder_features=32,
                 **kwargs):
        super(NvpAttExp, self).__init__(**kwargs)
        self.base_persons = base_persons

        with self.init_scope():
            self.encoder = NvpAttExpEncoder(
                audio_features=audio_features,
                audio_window_size=audio_window_size,
                seq_len=seq_len,
                encoder_features=encoder_features)
            self.decoder = SelectableDense(
                in_channels=encoder_features,
                out_channels=blendshapes,
                use_bias=False,
                num_options=base_persons)

    def __call__(self, x, pid):
        x, y = self.encoder(x)
        x = self.decoder(x, pid)
        y = self.decoder(y, pid)
        return x, y


def get_nvpattexp(base_persons,
                  blendshapes,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create Neural Voice Puppetry Audio-to-Expression model with specific parameters.

    Parameters:
    ----------
    base_persons : int
        Number of base persons (subjects).
    blendshapes : int
        Number of 3D model blendshapes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    net = NvpAttExp(
        base_persons=base_persons,
        blendshapes=blendshapes,
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


def nvpattexp116bazel76(**kwargs):
    """
    Neural Voice Puppetry Audio-to-Expression model for 116 base persons and Bazel topology with 76 blendshapes from
    'Neural Voice Puppetry: Audio-driven Facial Reenactment,' https://arxiv.org/abs/1912.05566.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_nvpattexp(base_persons=116, blendshapes=76, model_name="nvpattexp116bazel76", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        nvpattexp116bazel76,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nvpattexp116bazel76 or weight_count == 327397)

        batch = 14
        seq_len = 8
        audio_window_size = 16
        audio_features = 29
        blendshapes = 76

        x = np.random.rand(batch, seq_len, audio_window_size, audio_features).astype(np.float32)
        pid = np.full(shape=(batch,), fill_value=3, dtype=np.int64)
        y1, y2 = net(x, pid)
        assert (y1.shape == y2.shape == (batch, blendshapes))


if __name__ == "__main__":
    _test()
