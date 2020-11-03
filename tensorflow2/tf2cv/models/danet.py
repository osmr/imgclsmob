"""
    DANet for image segmentation, implemented in Gluon.
    Original paper: 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
"""

__all__ = ['DANet', 'danet_resnetd50b_cityscapes', 'danet_resnetd101b_cityscapes']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
from .common import conv1x1, conv3x3_block, is_channels_first, interpolate_im, get_im_size
from .resnetd import resnetd50b, resnetd101b


class ScaleBlock(nn.Layer):
    """
    Simple scale block.

    Parameters:
    ----------
    alpha_initializer : str, default 'zeros'
        Initializer function for the weights.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 alpha_initializer="zeros",
                 data_format="channels_last",
                 **kwargs):
        super(ScaleBlock, self).__init__(**kwargs)
        self.data_format = data_format
        self.alpha_initializer = initializers.get(alpha_initializer)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(1,),
            name="alpha",
            initializer=self.alpha_initializer,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=True)
        channel_axis = (1 if is_channels_first(self.data_format) else len(input_shape) - 1)
        axes = {}
        for i in range(1, len(input_shape)):
            if i != channel_axis:
                axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, x, training=None):
        return self.alpha * x

    def get_config(self):
        config = {
            "alpha_initializer": initializers.serialize(self.alpha_initializer),
            "data_format": self.data_format,
        }
        base_config = super(ScaleBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class PosAttBlock(nn.Layer):
    """
    Position attention block from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
    It captures long-range spatial contextual information.

    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 8
        Squeeze reduction value.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 reduction=8,
                 data_format="channels_last",
                 **kwargs):
        super(PosAttBlock, self).__init__(**kwargs)
        self.data_format = data_format
        mid_channels = channels // reduction

        self.query_conv = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            use_bias=True,
            data_format=data_format,
            name="query_conv")
        self.key_conv = conv1x1(
            in_channels=channels,
            out_channels=mid_channels,
            use_bias=True,
            data_format=data_format,
            name="key_conv")
        self.value_conv = conv1x1(
            in_channels=channels,
            out_channels=channels,
            use_bias=True,
            data_format=data_format,
            name="value_conv")
        self.scale = ScaleBlock(
            data_format=data_format,
            name="scale")
        self.softmax = nn.Softmax(axis=-1)

    def call(self, x, training=None):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        if not is_channels_first(self.data_format):
            proj_query = tf.transpose(proj_query, perm=(0, 3, 1, 2))
            proj_key = tf.transpose(proj_key, perm=(0, 3, 1, 2))
            proj_value = tf.transpose(proj_value, perm=(0, 3, 1, 2))

        batch, channels, height, width = proj_query.shape
        proj_query = tf.reshape(proj_query, shape=(batch, -1, height * width))
        proj_key = tf.reshape(proj_key, shape=(batch, -1, height * width))
        proj_value = tf.reshape(proj_value, shape=(batch, -1, height * width))

        energy = tf.keras.backend.batch_dot(tf.transpose(proj_query, perm=(0, 2, 1)), proj_key)
        w = self.softmax(energy)

        y = tf.keras.backend.batch_dot(proj_value, tf.transpose(w, perm=(0, 2, 1)))
        y = tf.reshape(y, shape=(batch, -1, height, width))

        if not is_channels_first(self.data_format):
            y = tf.transpose(y, perm=(0, 2, 3, 1))

        y = self.scale(y, training=training) + x
        return y


class ChaAttBlock(nn.Layer):
    """
    Channel attention block from 'Dual Attention Network for Scene Segmentation,' https://arxiv.org/abs/1809.02983.
    It explicitly models interdependencies between channels.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 data_format="channels_last",
                 **kwargs):
        super(ChaAttBlock, self).__init__(**kwargs)
        self.data_format = data_format

        self.scale = ScaleBlock(
            data_format=data_format,
            name="scale")
        self.softmax = nn.Softmax(axis=-1)

    def call(self, x, training=None):
        proj_query = x
        proj_key = x
        proj_value = x

        if not is_channels_first(self.data_format):
            proj_query = tf.transpose(proj_query, perm=(0, 3, 1, 2))
            proj_key = tf.transpose(proj_key, perm=(0, 3, 1, 2))
            proj_value = tf.transpose(proj_value, perm=(0, 3, 1, 2))

        batch, channels, height, width = proj_query.shape
        proj_query = tf.reshape(proj_query, shape=(batch, -1, height * width))
        proj_key = tf.reshape(proj_key, shape=(batch, -1, height * width))
        proj_value = tf.reshape(proj_value, shape=(batch, -1, height * width))

        energy = tf.keras.backend.batch_dot(proj_query, tf.transpose(proj_key, perm=(0, 2, 1)))
        energy_new = tf.broadcast_to(tf.math.reduce_max(energy, axis=-1, keepdims=True), shape=energy.shape) - energy
        w = self.softmax(energy_new)

        y = tf.keras.backend.batch_dot(w, proj_value)
        y = tf.reshape(y, shape=(batch, -1, height, width))

        if not is_channels_first(self.data_format):
            y = tf.transpose(y, perm=(0, 2, 3, 1))

        y = self.scale(y, training=training) + x
        return y


class DANetHeadBranch(nn.Layer):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 pose_att=True,
                 data_format="channels_last",
                 **kwargs):
        super(DANetHeadBranch, self).__init__(**kwargs)
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        if pose_att:
            self.att = PosAttBlock(
                mid_channels,
                data_format=data_format,
                name="att")
        else:
            self.att = ChaAttBlock(
                data_format=data_format,
                name="att")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="conv3")
        self.dropout = nn.Dropout(
            rate=dropout_rate,
            name="dropout")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.att(x, training=training)
        y = self.conv2(x, training=training)
        x = self.conv3(y)
        x = self.dropout(x, training=training)
        return x, y


class DANetHead(nn.Layer):
    """
    DANet head block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(DANetHead, self).__init__(**kwargs)
        mid_channels = in_channels // 4
        dropout_rate = 0.1

        self.branch_pa = DANetHeadBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            pose_att=True,
            data_format=data_format,
            name="branch_pa")
        self.branch_ca = DANetHeadBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            pose_att=False,
            data_format=data_format,
            name="branch_ca")
        self.conv = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=True,
            data_format=data_format,
            name="conv")
        self.dropout = nn.Dropout(
            rate=dropout_rate,
            name="dropout")

    def call(self, x, training=None):
        pa_x, pa_y = self.branch_pa(x, training=training)
        ca_x, ca_y = self.branch_ca(x, training=training)
        y = pa_y + ca_y
        x = self.conv(y)
        x = self.dropout(x, training=training)
        return x, pa_x, ca_x


class DANet(tf.keras.Model):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 backbone,
                 backbone_out_channels=2048,
                 aux=False,
                 fixed_size=True,
                 in_channels=3,
                 in_size=(480, 480),
                 classes=19,
                 data_format="channels_last",
                 **kwargs):
        super(DANet, self).__init__(**kwargs)
        assert (in_channels > 0)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.classes = classes
        self.aux = aux
        self.fixed_size = fixed_size
        self.data_format = data_format

        self.backbone = backbone
        self.head = DANetHead(
            in_channels=backbone_out_channels,
            out_channels=classes,
            data_format=data_format,
            name="head")

    def call(self, x, training=None):
        in_size = self.in_size if self.fixed_size else get_im_size(x, data_format=self.data_format)
        x, _ = self.backbone(x, training=training)
        x, y, z = self.head(x, training=training)
        x = interpolate_im(x, out_size=in_size, data_format=self.data_format)
        if self.aux:
            y = interpolate_im(y, out_size=in_size, data_format=self.data_format)
            z = interpolate_im(z, out_size=in_size, data_format=self.data_format)
            return x, y, z
        else:
            return x


def get_danet(backbone,
              classes,
              aux=False,
              model_name=None,
              data_format="channels_last",
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    net = DANet(
        backbone=backbone,
        classes=classes,
        aux=aux,
        data_format=data_format,
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
                local_model_store_dir_path=root),
            by_name=True,
            skip_mismatch=True)

    return net


def danet_resnetd50b_cityscapes(pretrained_backbone=False, classes=19, aux=True, data_format="channels_last", **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd50b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                          data_format=data_format).features
    backbone.children.pop()
    return get_danet(backbone=backbone, classes=classes, aux=aux, model_name="danet_resnetd50b_cityscapes",
                     data_format=data_format, **kwargs)


def danet_resnetd101b_cityscapes(pretrained_backbone=False, classes=19, aux=True, data_format="channels_last",
                                 **kwargs):
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
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    backbone = resnetd101b(pretrained=pretrained_backbone, ordinary_init=False, bends=(3,),
                           data_format=data_format).features
    backbone.children.pop()
    return get_danet(backbone=backbone, classes=classes, aux=aux, model_name="danet_resnetd101b_cityscapes",
                     data_format=data_format, **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    in_size = (480, 480)
    aux = False
    pretrained = False

    models = [
        danet_resnetd50b_cityscapes,
        danet_resnetd101b_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, aux=aux, data_format=data_format)

        batch = 14
        classes = 19
        x = tf.random.normal((batch, 3, in_size[0], in_size[1]) if is_channels_first(data_format) else
                             (batch, in_size[0], in_size[1], 3))
        ys = net(x)
        y = ys[0] if aux else ys
        assert (y.shape[0] == x.shape[0])
        if is_channels_first(data_format):
            assert ((y.shape[1] == classes) and (y.shape[2] == x.shape[2]) and (y.shape[3] == x.shape[3]))
        else:
            assert ((y.shape[3] == classes) and (y.shape[1] == x.shape[1]) and (y.shape[2] == x.shape[2]))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != danet_resnetd50b_cityscapes or weight_count == 47586427)
        assert (model != danet_resnetd101b_cityscapes or weight_count == 66578555)


if __name__ == "__main__":
    _test()
