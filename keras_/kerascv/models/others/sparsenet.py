'''SparseNet models for Keras.
# Reference
- [Sparsely Connected Convolutional Networks](https://arxiv.org/abs/1801.05895)
- [Github](https://github.com/lyken17/sparsenet)
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import warnings

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model, convert_dense_weights_data_format
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
import keras.backend as K


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017  # scale values

    return x


def SparseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
              bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
              include_top=True, weights=None, input_tensor=None,
              classes=10, activation='softmax'):
    '''Instantiate the SparseNet architecture,
        optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            depth: number or layers in the DenseNet
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block. Can be
                a single integer number or a list of numbers.
                If it is a list, length of list must match the length of
                `nb_layers_per_block`
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'imagenet' (pre-training on ImageNet)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
        # Returns
            A Keras model instance.
        '''

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top`'
                         ' as true, `classes` should be 1000')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=32,
    #                                   min_size=8,
    #                                   data_format=K.image_data_format(),
    #                                   require_flatten=include_top)
    in_channels = 3
    input_shape = (in_channels, 224, 224) if K.image_data_format() == 'channels_first' else (224, 224, in_channels)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                          growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                          dropout_rate, weight_decay, subsample_initial_block, activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')

    # load weights
    if weights == 'imagenet':
        weights_loaded = False

        if weights_loaded:
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

            if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

            print("Weights for the model were loaded successfully")

    return model


def SparseNetImageNet121(input_shape=None,
                         bottleneck=True,
                         reduction=0.5,
                         dropout_rate=0.0,
                         weight_decay=1e-4,
                         include_top=True,
                         weights=None,
                         input_tensor=None,
                         classes=1000,
                         activation='softmax'):
    return SparseNet(input_shape, depth=121, nb_dense_block=4, growth_rate=32, nb_filter=64,
                     nb_layers_per_block=[6, 12, 24, 16], bottleneck=bottleneck, reduction=reduction,
                     dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                     include_top=include_top, weights=weights, input_tensor=input_tensor,
                     classes=classes, activation=activation)


def SparseNetImageNet169(input_shape=None,
                         bottleneck=True,
                         reduction=0.5,
                         dropout_rate=0.0,
                         weight_decay=1e-4,
                         include_top=True,
                         weights=None,
                         input_tensor=None,
                         classes=1000,
                         activation='softmax'):
    return SparseNet(input_shape, depth=169, nb_dense_block=4, growth_rate=32, nb_filter=64,
                     nb_layers_per_block=[6, 12, 32, 32], bottleneck=bottleneck, reduction=reduction,
                     dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                     include_top=include_top, weights=weights, input_tensor=input_tensor,
                     classes=classes, activation=activation)


def SparseNetImageNet201(input_shape=None,
                         bottleneck=True,
                         reduction=0.5,
                         dropout_rate=0.0,
                         weight_decay=1e-4,
                         include_top=True,
                         weights=None,
                         input_tensor=None,
                         classes=1000,
                         activation='softmax'):
    return SparseNet(input_shape, depth=201, nb_dense_block=4, growth_rate=32, nb_filter=64,
                     nb_layers_per_block=[6, 12, 48, 32], bottleneck=bottleneck, reduction=reduction,
                     dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                     include_top=include_top, weights=weights, input_tensor=input_tensor,
                     classes=classes, activation=activation)


def SparseNetImageNet264(input_shape=None,
                         bottleneck=True,
                         reduction=0.5,
                         dropout_rate=0.0,
                         weight_decay=1e-4,
                         include_top=True,
                         weights=None,
                         input_tensor=None,
                         classes=1000,
                         activation='softmax'):
    return SparseNet(input_shape, depth=264, nb_dense_block=4, growth_rate=32, nb_filter=64,
                     nb_layers_per_block=[6, 12, 64, 48], bottleneck=bottleneck, reduction=reduction,
                     dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                     include_top=include_top, weights=weights, input_tensor=input_tensor,
                     classes=classes, activation=activation)


def SparseNetImageNet161(input_shape=None,
                         bottleneck=True,
                         reduction=0.5,
                         dropout_rate=0.0,
                         weight_decay=1e-4,
                         include_top=True,
                         weights=None,
                         input_tensor=None,
                         classes=1000,
                         activation='softmax'):
    return SparseNet(input_shape, depth=161, nb_dense_block=4, growth_rate=48, nb_filter=96,
                     nb_layers_per_block=[6, 12, 36, 24], bottleneck=bottleneck, reduction=reduction,
                     dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                     include_top=include_top, weights=weights, input_tensor=input_tensor,
                     classes=classes, activation=activation)


def _exponential_index_fetch(x_list):
    count = len(x_list)
    i = 1
    inputs = []
    while i <= count:
        inputs.append(x_list[count - i])
        i *= 2
    return inputs


def _conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('conv_block'):
        x = BatchNormalization(axis=concat_axis, momentum=0.1, epsilon=1e-5)(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

            x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1e-5, momentum=0.1)(x)
            x = Activation('relu')(x)

        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    return x


def _dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                 grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]
    channel_list = [nb_filter]

    for i in range(nb_layers):
        #nb_channels = sum(_exponential_index_fetch(channel_list))

        x = _conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(x)

        fetch_outputs = _exponential_index_fetch(x_list)
        x = concatenate(fetch_outputs, axis=concat_axis)

        channel_list.append(growth_rate)

    if grow_nb_filters:
        nb_filter = sum(_exponential_index_fetch(channel_list))

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def _transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('transition_block'):
        x = BatchNormalization(axis=concat_axis, epsilon=1e-5, momentum=0.1)(ip)
        x = Activation('relu')(x)
        x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def _create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                      nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                      subsample_initial_block=False, activation='softmax'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
        subsample_initial:
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                                                   'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)

            if bottleneck:
                count = count // 2

            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    if type(growth_rate) is list or type(growth_rate) is tuple:
        growth_rate = list(growth_rate)
        assert len(growth_rate) == len(nb_layers)
    else:
        growth_rate = [growth_rate for _ in range(len(nb_layers))]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = growth_rate[0]

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1e-5, momentum=0.1)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = _dense_block(x, nb_layers[block_idx], nb_filter, growth_rate[block_idx], bottleneck=bottleneck,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = _transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = _dense_block(x, final_nb_layer, nb_filter, growth_rate[-1], bottleneck=bottleneck,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1e-5, momentum=0.1)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(nb_classes, activation=activation)(x)

    return x


if __name__ == '__main__':
    # from keras.utils.vis_utils import plot_model
    # import tensorflow as tf
    # from keras import backend as K
    # sess = tf.Session()
    # K.set_session(sess)

    # model = SparseNet((32, 32, 3), depth=40, nb_dense_block=3,
    #                   growth_rate=24, bottleneck=False, reduction=0.0, weights=None)
    model = SparseNetImageNet121((3, 224, 224))
    model.summary()
    q = 10

    #writer = tf.summary.FileWriter('logs/', graph=sess.graph)
    #writer.close()

    #plot_model(model, 'sparse.png', show_shapes=True)

