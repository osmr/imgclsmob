
__all__ = ['ShuffleNetV2', 'oth_shufflenetv2_wd2', 'oth_shufflenetv2_w1']


import os
import tensorflow as tf
from tensorpack.models import Conv2D, BNReLU, BatchNorm, MaxPooling, AvgPooling, GlobalAvgPooling, FullyConnected,\
    layer_register
from tensorpack.tfutils import argscope
from .common_ import ImageNetModel, depthwise_conv, channel_shuffle


@layer_register()
def shufflenet_unit_v2(x,
                       out_channels,
                       downsample):
    in_channels = int(x.shape[1])
    mid_channels = out_channels // 2
    in_channels2 = in_channels // 2
    assert (in_channels % 2 == 0)

    if downsample:
        y1, x2 = x, x
        # shortcut_channel = int(x.shape[1])
        # assert (shortcut_channel == int(y1.shape[1]))

        y1 = depthwise_conv(
            'shortcut_dconv',
            y1,
            channels=in_channels,
            kernel_size=3,
            strides=2)
        y1 = BatchNorm('shortcut_dconv_bn', y1)
        y1 = Conv2D(
            'shortcut_conv',
            y1,
            filters=in_channels,
            kernel_size=1,
            activation=BNReLU)
    else:
        y1, x2 = tf.split(x, 2, axis=1)
        # shortcut_channel = int(x.shape[1] // 2)
        # assert (shortcut_channel == int(y1.shape[1]))

    y2_in_channels = (in_channels if downsample else in_channels2)
    y2 = Conv2D(
        'conv1',
        x2,
        filters=mid_channels,
        kernel_size=1,
        activation=BNReLU)
    y2 = depthwise_conv(
        'dconv',
        y2,
        channels=mid_channels,
        kernel_size=3,
        strides=(2 if downsample else 1))
    y2 = BatchNorm('dconv_bn', y2)

    y2_out_channels = out_channels - y2_in_channels
    y2 = Conv2D(
        'conv2',
        y2,
        filters=y2_out_channels,
        kernel_size=1,
        activation=BNReLU)

    output = tf.concat([y1, y2], axis=1)
    output = channel_shuffle(output, 2)
    return output


@layer_register(log_shape=True)
def shufflenet_stage(x,
                     out_channels,
                     units_per_stage):
    for i in range(units_per_stage):
        name = 'block{}'.format(i)
        x = shufflenet_unit_v2(
            name,
            x,
            out_channels=out_channels,
            downsample=(i == 0))
    return x


class ShuffleNetV2(ImageNetModel):

    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 classes=1000,
                 **kwargs):
        super(ShuffleNetV2, self).__init__(**kwargs)
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.final_block_channels = final_block_channels
        self.classes = classes
        self.weight_decay = 4e-5

    def get_logits(self, x, training=False):

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm],
                      data_format='channels_first'), argscope(Conv2D, use_bias=False):

            x = Conv2D(
                'conv1',
                x,
                filters=self.init_block_channels,
                kernel_size=3,
                strides=2,
                activation=BNReLU)
            x = MaxPooling(
                'pool1',
                x,
                pool_size=3,
                strides=2,
                padding='SAME')

            x = shufflenet_stage('stage2', x, out_channels=self.channels[0], units_per_stage=4)
            x = shufflenet_stage('stage3', x, out_channels=self.channels[1], units_per_stage=8)
            x = shufflenet_stage('stage4', x, out_channels=self.channels[2], units_per_stage=4)

            x = Conv2D(
                'conv5',
                x,
                filters=self.final_block_channels,
                kernel_size=1,
                activation=BNReLU)

            x = GlobalAvgPooling('gap', x)
            x = FullyConnected(
                'linear',
                x,
                units=self.classes)
            return x


def get_shufflenetv2(ratio,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join('~', '.tensorflow', 'models'),
                     **kwargs):
    """
    Create ShuffleNetV2 model with specific parameters.

    Parameters:
    ----------
    ratio : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 24
    final_block_channels = 1024
    channels = {
        0.5: [48, 96, 192],
        1.0: [116, 232, 464]
    }[ratio]
    net = ShuffleNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)
    return net.get_logits, None


def oth_shufflenetv2_wd2(**kwargs):
    """
    ShuffleNetV2 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2(ratio=0.5, model_name="shufflenetv2_wd2", **kwargs)


def oth_shufflenetv2_w1(**kwargs):
    """
    ShuffleNetV2 1.0x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2(ratio=1.0, model_name="shufflenetv2_w1", **kwargs)


def _load_model(sess,
                file_path,
                ignore_extra=True):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    sess: Session
        A Session to use to load the weights.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import numpy as np
    import tensorflow as tf

    assert sess is not None
    assert os.path.exists(file_path) and os.path.isfile(file_path)
    if file_path.endswith('.npy'):
        src_params = np.load(file_path, encoding='latin1').item()
    elif file_path.endswith('.npz'):
        src_params = dict(np.load(file_path))
    else:
        raise NotImplementedError
    dst_params = {v.name: v for v in tf.global_variables()}
    sess.run(tf.global_variables_initializer())
    for src_key in src_params.keys():
        if src_key in dst_params.keys():
            assert (src_params[src_key].shape == tuple(dst_params[src_key].get_shape().as_list()))
            sess.run(dst_params[src_key].assign(src_params[src_key]))
        elif not ignore_extra:
            raise Exception("The file `{}` is incompatible with the model".format(file_path))
        else:
            print("Key `{}` is ignored".format(src_key))


def _test():
    import numpy as np

    pretrained = False

    models = [
        oth_shufflenetv2_wd2,
    ]

    for model in models:

        net_lambda, net_file_path = model(pretrained=pretrained)

        x = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224),
            name='xx')
        y_net = net_lambda(x)

        weight_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("m={}, {}".format(model.__name__, weight_count))

        with tf.Session() as sess:
            if pretrained:
                _load_model(sess=sess, file_path=net_file_path)
            else:
                sess.run(tf.global_variables_initializer())
            x_value = np.zeros((1, 3, 224, 224), np.float32)

            _load_model(sess=sess, file_path="/home/semery/projects/imgclsmob_data/tf-shufflenetv2b_wd2/ShuffleNetV2-0.5x.npz")
            x_value = np.load('/home/semery/Downloads/x.npy')

            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
