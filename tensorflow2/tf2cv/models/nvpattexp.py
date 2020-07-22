"""
    Neural Voice Puppetry Audio-to-Expression net for speech-driven facial animation, implemented in TensorFlow.
    Original paper: 'Neural Voice Puppetry: Audio-driven Facial Reenactment,' https://arxiv.org/abs/1912.05566.
"""

__all__ = ['NvpAttExp', 'nvpattexp116bazel76']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import DenseBlock, ConvBlock, ConvBlock1d, SelectableDense, SimpleSequential, is_channels_first


class NvpAttExpEncoder(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 audio_features,
                 audio_window_size,
                 seq_len,
                 encoder_features,
                 data_format="channels_last",
                 **kwargs):
        super(NvpAttExpEncoder, self).__init__(**kwargs)
        self.audio_features = audio_features
        self.audio_window_size = audio_window_size
        self.seq_len = seq_len
        self.data_format = data_format
        conv_channels = (32, 32, 64, 64)
        conv_slopes = (0.02, 0.02, 0.2, 0.2)
        fc_channels = (128, 64, encoder_features)
        fc_slopes = (0.02, 0.02, None)
        att_conv_channels = (16, 8, 4, 2, 1)
        att_conv_slopes = 0.02

        in_channels = audio_features
        self.conv_branch = SimpleSequential(name="conv_branch")
        for i, (out_channels, slope) in enumerate(zip(conv_channels, conv_slopes)):
            self.conv_branch.add(ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 1),
                strides=(2, 1),
                padding=(1, 0),
                use_bias=True,
                use_bn=False,
                activation=nn.LeakyReLU(alpha=slope),
                data_format=data_format,
                name="conv{}".format(i + 1)))
            in_channels = out_channels

        self.fc_branch = SimpleSequential(name="fc_branch")
        for i, (out_channels, slope) in enumerate(zip(fc_channels, fc_slopes)):
            activation = nn.LeakyReLU(alpha=slope) if slope is not None else "tanh"
            self.fc_branch.add(DenseBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=False,
                activation=activation,
                data_format=data_format,
                name="fc{}".format(i + 1)))
            in_channels = out_channels

        self.att_conv_branch = SimpleSequential(name="att_conv_branch")
        for i, out_channels, in enumerate(att_conv_channels):
            self.att_conv_branch.add(ConvBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                strides=1,
                padding=1,
                use_bias=True,
                use_bn=False,
                activation=nn.LeakyReLU(alpha=att_conv_slopes),
                data_format=data_format,
                name="att_conv{}".format(i + 1)))
            in_channels = out_channels

        self.att_fc = DenseBlock(
            in_channels=seq_len,
            out_channels=seq_len,
            use_bias=True,
            use_bn=False,
            activation=nn.Softmax(axis=1),
            data_format=data_format,
            name="att_fc")

    def call(self, x, training=None):
        batch = x.shape[0]
        batch_seq_len = batch * self.seq_len

        if is_channels_first(self.data_format):
            x = tf.reshape(x, shape=(-1, 1, self.audio_window_size, self.audio_features))
            x = tf.transpose(x, perm=(0, 3, 2, 1))
            x = self.conv_branch(x)
            x = tf.squeeze(x, axis=-1)
            x = tf.reshape(x, shape=(batch_seq_len, 1, -1))
            x = self.fc_branch(x)
            x = tf.reshape(x, shape=(batch, self.seq_len, -1))
            x = tf.transpose(x, perm=(0, 2, 1))

            y = x[:, :, (self.seq_len // 2)]

            w = self.att_conv_branch(x)
            w = tf.squeeze(w, axis=1)
            w = self.att_fc(w)
            w = tf.expand_dims(w, axis=-1)
        else:
            x = tf.transpose(x, perm=(0, 3, 1, 2))
            x = tf.reshape(x, shape=(-1, 1, self.audio_window_size, self.audio_features))
            x = tf.transpose(x, perm=(0, 2, 3, 1))
            x = tf.transpose(x, perm=(0, 1, 3, 2))
            x = self.conv_branch(x)
            x = tf.squeeze(x, axis=1)
            x = self.fc_branch(x)
            x = tf.reshape(x, shape=(batch, self.seq_len, -1))

            y = x[:, (self.seq_len // 2), :]

            w = self.att_conv_branch(x)
            w = tf.squeeze(w, axis=-1)
            w = self.att_fc(w)
            w = tf.expand_dims(w, axis=-1)
            x = tf.transpose(x, perm=(0, 2, 1))

        x = tf.keras.backend.batch_dot(x, w)
        x = tf.squeeze(x, axis=-1)

        return x, y


class NvpAttExp(tf.keras.Model):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 audio_features=29,
                 audio_window_size=16,
                 seq_len=8,
                 base_persons=116,
                 blendshapes=76,
                 encoder_features=32,
                 data_format="channels_last",
                 **kwargs):
        super(NvpAttExp, self).__init__(**kwargs)
        self.base_persons = base_persons
        self.data_format = data_format

        self.encoder = NvpAttExpEncoder(
            audio_features=audio_features,
            audio_window_size=audio_window_size,
            seq_len=seq_len,
            encoder_features=encoder_features,
            data_format=data_format,
            name="encoder")
        self.decoder = SelectableDense(
            in_channels=encoder_features,
            out_channels=blendshapes,
            use_bias=False,
            num_options=base_persons,
            name="decoder")

    def call(self, x, pid, training=None):
        x, y = self.encoder(x, training=training)
        x = self.decoder(x, pid)
        y = self.decoder(y, pid)
        return x, y


def get_nvpattexp(base_persons,
                  blendshapes,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
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
    root : str, default '~/.tensorflow/models'
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
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def nvpattexp116bazel76(**kwargs):
    """
    Neural Voice Puppetry Audio-to-Expression model for 116 base persons and Bazel topology with 76 blendshapes from
    'Neural Voice Puppetry: Audio-driven Facial Reenactment,' https://arxiv.org/abs/1912.05566.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_nvpattexp(base_persons=116, blendshapes=76, model_name="nvpattexp116bazel76", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    # data_format = "channels_first"
    data_format = "channels_last"
    pretrained = False

    models = [
        nvpattexp116bazel76,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        seq_len = 8
        audio_window_size = 16
        audio_features = 29
        blendshapes = 76

        x = tf.random.normal((batch, seq_len, audio_window_size, audio_features) if is_channels_first(data_format) else
                             (batch, audio_window_size, audio_features, seq_len))
        pid = tf.fill(dims=(batch,), value=3)
        y1, y2 = net(x, pid)
        assert (y1.shape == y2.shape == (batch, blendshapes))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nvpattexp116bazel76 or weight_count == 327397)


if __name__ == "__main__":
    _test()
