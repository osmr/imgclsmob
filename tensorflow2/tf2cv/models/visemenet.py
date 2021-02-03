"""
    VisemeNet for speech-driven facial animation, implemented in TensorFlow.
    Original paper: 'VisemeNet: Audio-Driven Animator-Centric Speech Animation,' https://arxiv.org/abs/1805.09488.
"""

__all__ = ['VisemeNet', 'visemenet20']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import DenseBlock, SimpleSequential


class VisemeDenseBranch(tf.keras.Model):
    """
    VisemeNet dense branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of middle/output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 data_format="channels_last",
                 **kwargs):
        super(VisemeDenseBranch, self).__init__(**kwargs)
        self.branch = SimpleSequential(name="branch")
        for i, out_channels in enumerate(out_channels_list[:-1]):
            self.branch.add(DenseBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=True,
                use_bn=True,
                data_format=data_format,
                name="block{}".format(i + 1)))
            in_channels = out_channels
        self.final_fc = nn.Dense(
            units=out_channels_list[-1],
            input_dim=in_channels,
            name="final_fc")

    def call(self, x, training=None):
        x = self.branch(x, training=training)
        y = self.final_fc(x)
        return y, x


class VisemeRnnBranch(nn.Layer):
    """
    VisemeNet RNN branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of middle/output channels.
    rnn_num_layers : int
        Number of RNN layers.
    dropout_rate : float
        Dropout rate.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 rnn_num_layers,
                 dropout_rate,
                 data_format="channels_last",
                 **kwargs):
        super(VisemeRnnBranch, self).__init__(**kwargs)
        assert (in_channels is not None)

        self.rnn = nn.RNN([nn.LSTMCell(
            units=out_channels_list[0],
            dropout=dropout_rate,
            name="rnn{}".format(i + 1)
        ) for i in range(rnn_num_layers)])
        self.fc_branch = VisemeDenseBranch(
            in_channels=out_channels_list[0],
            out_channels_list=out_channels_list[1:],
            data_format=data_format,
            name="fc_branch")

    def call(self, x, training=None):
        x = self.rnn(x, training=training)
        # x = x[:, -1, :]
        y, _ = self.fc_branch(x, training=training)
        return y


class VisemeNet(tf.keras.Model):
    """
    VisemeNet model from 'VisemeNet: Audio-Driven Animator-Centric Speech Animation,' https://arxiv.org/abs/1805.09488.

    Parameters:
    ----------
    audio_features : int, default 195
        Number of audio features (characters/sounds).
    audio_window_size : int, default 8
        Size of audio window (for time related audio features).
    stage2_window_size : int, default 64
        Size of window for stage #2.
    num_face_ids : int, default 76
        Number of face IDs.
    num_landmarks : int, default 76
        Number of landmarks.
    num_phonemes : int, default 21
        Number of phonemes.
    num_visemes : int, default 20
        Number of visemes.
    dropout_rate : float, default 0.5
        Dropout rate for RNNs.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 audio_features=195,
                 audio_window_size=8,
                 stage2_window_size=64,
                 num_face_ids=76,
                 num_landmarks=76,
                 num_phonemes=21,
                 num_visemes=20,
                 dropout_rate=0.5,
                 data_format="channels_last",
                 **kwargs):
        super(VisemeNet, self).__init__(**kwargs)
        stage1_rnn_hidden_size = 256
        stage1_fc_mid_channels = 256
        stage2_rnn_in_features = (audio_features + num_landmarks + stage1_fc_mid_channels) * \
                                 stage2_window_size // audio_window_size
        self.audio_window_size = audio_window_size
        self.stage2_window_size = stage2_window_size

        self.stage1_rnn = nn.RNN([nn.LSTMCell(
            units=stage1_rnn_hidden_size,
            dropout=dropout_rate,
            name="stage1_rnn{}".format(i + 1)
        ) for i in range(3)])

        self.lm_branch = VisemeDenseBranch(
            in_channels=(stage1_rnn_hidden_size + num_face_ids),
            out_channels_list=[stage1_fc_mid_channels, num_landmarks],
            data_format=data_format,
            name="lm_branch")
        self.ph_branch = VisemeDenseBranch(
            in_channels=(stage1_rnn_hidden_size + num_face_ids),
            out_channels_list=[stage1_fc_mid_channels, num_phonemes],
            data_format=data_format,
            name="ph_branch")

        self.cls_branch = VisemeRnnBranch(
            in_channels=stage2_rnn_in_features,
            out_channels_list=[256, 200, num_visemes],
            rnn_num_layers=1,
            dropout_rate=dropout_rate,
            data_format=data_format,
            name="cls_branch")
        self.reg_branch = VisemeRnnBranch(
            in_channels=stage2_rnn_in_features,
            out_channels_list=[256, 200, 100, num_visemes],
            rnn_num_layers=3,
            dropout_rate=dropout_rate,
            data_format=data_format,
            name="reg_branch")
        self.jali_branch = VisemeRnnBranch(
            in_channels=stage2_rnn_in_features,
            out_channels_list=[128, 200, 2],
            rnn_num_layers=3,
            dropout_rate=dropout_rate,
            data_format=data_format,
            name="jali_branch")

    def call(self, x, pid, training=None):
        y = self.stage1_rnn(x, training=training)
        # y = y[:, -1, :]
        y = tf.concat([y, tf.cast(pid, tf.float32)], axis=1)

        lm, _ = self.lm_branch(y, training=training)
        lm += tf.cast(pid, tf.float32)

        ph, ph1 = self.ph_branch(y, training=training)

        z = tf.concat([lm, ph1], axis=1)

        z2 = tf.concat([z, x[:, self.audio_window_size // 2, :]], axis=1)
        n_net2_input = z2.shape[1]
        z2 = tf.concat([tf.zeros((self.stage2_window_size // 2, n_net2_input)), z2], axis=0)
        z = tf.stack(
            [tf.reshape(
                z2[i:i + self.stage2_window_size],
                shape=(self.audio_window_size, n_net2_input * self.stage2_window_size // self.audio_window_size))
              for i in range(z2.shape[0] - self.stage2_window_size)],
            axis=0)
        cls = self.cls_branch(z, training=training)
        reg = self.reg_branch(z, training=training)
        jali = self.jali_branch(z, training=training)

        return cls, reg, jali


def get_visemenet(model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".tensorflow", "models"),
                  **kwargs):
    """
    Create VisemeNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = VisemeNet(
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


def visemenet20(**kwargs):
    """
    VisemeNet model for 20 visemes (without co-articulation rules) from 'VisemeNet: Audio-Driven Animator-Centric
    Speech Animation,' https://arxiv.org/abs/1805.09488.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_visemenet(model_name="visemenet20", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    pretrained = False

    models = [
        visemenet20,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 34
        audio_window_size = 8
        audio_features = 195
        num_face_ids = 76
        num_visemes = 20

        x = tf.random.normal((batch, audio_window_size, audio_features))
        pid = tf.fill(dims=(batch, num_face_ids), value=3)
        y1, y2, y3 = net(x, pid)
        assert (y1.shape[0] == y2.shape[0] == y3.shape[0])
        assert (y1.shape[-1] == y2.shape[-1] == num_visemes)
        assert (y3.shape[-1] == 2)

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != visemenet20 or weight_count == 14574303)
        assert (model != visemenet20 or weight_count == 14565599)
        print(net.summary())


if __name__ == "__main__":
    _test()
