"""
    VOCA for speech-driven facial animation, implemented in TensorFlow.
    Original paper: 'Capture, Learning, and Synthesis of 3D Speaking Styles,' https://arxiv.org/abs/1905.03079.
"""

__all__ = ['VOCA', 'voca8flame']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import BatchNorm, ConvBlock, SimpleSequential, flatten, get_channel_axis, is_channels_first


class VocaEncoder(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 audio_features,
                 audio_window_size,
                 base_persons,
                 encoder_features,
                 data_format="channels_last",
                 **kwargs):
        super(VocaEncoder, self).__init__(**kwargs)
        self.audio_window_size = audio_window_size
        self.data_format = data_format
        channels = (32, 32, 64, 64)
        fc1_channels = 128

        self.bn = BatchNorm(
            epsilon=1e-5,
            data_format=data_format,
            name="bn")

        in_channels = audio_features + base_persons
        self.branch = SimpleSequential(name="branch")
        for i, out_channels in enumerate(channels):
            self.branch.add(ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 1),
                strides=(2, 1),
                padding=(1, 0),
                use_bias=True,
                use_bn=False,
                data_format=data_format,
                name="conv{}".format(i + 1)))
            in_channels = out_channels

        in_channels += base_persons
        self.fc1 = nn.Dense(
            units=fc1_channels,
            input_dim=in_channels,
            name="fc1")
        self.fc2 = nn.Dense(
            units=encoder_features,
            input_dim=fc1_channels,
            name="fc2")

    def call(self, x, pid, training=None):
        x = self.bn(x, training=training)
        if is_channels_first(self.data_format):
            x = tf.transpose(x, perm=(0, 3, 2, 1))
            y = tf.expand_dims(tf.expand_dims(pid, -1), -1)
            y = tf.tile(y, multiples=(1, 1, self.audio_window_size, 1))
        else:
            x = tf.transpose(x, perm=(0, 1, 3, 2))
            y = tf.expand_dims(tf.expand_dims(pid, 1), 1)
            y = tf.tile(y, multiples=(1, self.audio_window_size, 1, 1))
        x = tf.concat([x, y], axis=get_channel_axis(self.data_format))
        x = self.branch(x)
        x = flatten(x, self.data_format)
        x = tf.concat([x, pid], axis=1)
        x = self.fc1(x)
        x = tf.math.tanh(x)
        x = self.fc2(x)
        return x


class VOCA(tf.keras.Model):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 audio_features=29,
                 audio_window_size=16,
                 base_persons=8,
                 encoder_features=50,
                 vertices=5023,
                 data_format="channels_last",
                 **kwargs):
        super(VOCA, self).__init__(**kwargs)
        self.base_persons = base_persons
        self.data_format = data_format

        self.encoder = VocaEncoder(
            audio_features=audio_features,
            audio_window_size=audio_window_size,
            base_persons=base_persons,
            encoder_features=encoder_features,
            data_format=data_format,
            name="encoder")
        self.decoder = nn.Dense(
            units=(3 * vertices),
            input_dim=encoder_features,
            name="decoder")

    def call(self, x, pid, training=None):
        pid = tf.one_hot(pid, depth=self.base_persons)
        x = self.encoder(x, pid, training=training)
        x = self.decoder(x)
        x = tf.reshape(x, shape=(x.get_shape().as_list()[0], 1, -1, 3)) if is_channels_first(self.data_format) else\
            tf.reshape(x, shape=(x.get_shape().as_list()[0], -1, 3, 1))
        return x


def get_voca(base_persons,
             vertices,
             model_name=None,
             pretrained=False,
             root=os.path.join("~", ".tensorflow", "models"),
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
    root : str, default '~/.tensorflow/models'
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
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def voca8flame(**kwargs):
    """
    VOCA-8-FLAME model for 8 base persons and FLAME topology from 'Capture, Learning, and Synthesis of 3D Speaking
    Styles,' https://arxiv.org/abs/1905.03079.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_voca(base_persons=8, vertices=5023, model_name="voca8flame", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    # data_format = "channels_first"
    data_format = "channels_last"
    pretrained = False

    models = [
        voca8flame,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        audio_features = 29
        audio_window_size = 16
        vertices = 5023

        x = tf.random.normal((batch, 1, audio_window_size, audio_features) if is_channels_first(data_format) else
                             (batch, audio_window_size, audio_features, 1))
        pid = tf.fill(dims=(batch,), value=3)
        y = net(x, pid)
        if is_channels_first(data_format):
            assert (y.shape == (batch, 1, vertices, 3))
        else:
            assert (y.shape == (batch, vertices, 3, 1))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != voca8flame or weight_count == 809563)


if __name__ == "__main__":
    _test()
