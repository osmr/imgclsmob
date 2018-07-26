"""
Reference:
J. Redmon. Darknet: Open source neural networks in c.
"http://pjreddie.com/darknet/", 2013-2016 5
--> from: https://raw.githubusercontent.com/zhreshold/mxnet-yolo/master/symbol/symbol_darknet19.py
"""

import mxnet as mx

def conv_act_layer(from_layer, name, num_filter, kernel=(3, 3), pad=(1, 1), \
    stride=(1,1), act_type="relu", use_batchnorm=True):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad, \
        stride=stride, num_filter=num_filter, name="{}".format(name))
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="bn_{}".format(name))
    if act_type in ['elu', 'leaky', 'prelu', 'rrelu']:
        relu = mx.symbol.LeakyReLU(data=conv, act_type=act_type,
        name="{}_{}".format(act_type, name), slope=0.1)
    elif act_type in ['relu', 'sigmoid', 'softrelu', 'tanh']:
        relu = mx.symbol.Activation(data=conv, act_type=act_type, \
        name="{}_{}".format(act_type, name))
    else:
        assert isinstance(act_type, str)
        raise ValueError("Invalid activation type: " + str(act_type))
    return relu

def get_symbol(num_classes=1000, **kwargs):
    data = mx.sym.Variable(name='data')

    # group 1
    conv1 = conv_act_layer(data, 'conv1', 32, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    pool1 = mx.sym.Pooling(data=conv1, pool_type='max', kernel=(2, 2),
        stride=(2, 2), name='pool1')
    # group 2
    conv2 = conv_act_layer(pool1, 'conv2', 64, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    pool2 = mx.sym.Pooling(data=conv2, pool_type='max', kernel=(2, 2),
        stride=(2, 2), name='pool2')
    # group 3
    conv3_1 = conv_act_layer(pool2, 'conv3_1', 128, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    conv3_2 = conv_act_layer(conv3_1, 'conv3_2', 64, kernel=(1, 1), pad=(0, 0),
        act_type='leaky')
    conv3_3 = conv_act_layer(conv3_2, 'conv3_3', 128, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    pool3 = mx.sym.Pooling(data=conv3_3, pool_type='max', kernel=(2, 2),
        stride=(2, 2), name='pool3')
    # group 4
    conv4_1 = conv_act_layer(pool3, 'conv4_1', 256, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    conv4_2 = conv_act_layer(conv4_1, 'conv4_2', 128, kernel=(1, 1), pad=(0, 0),
        act_type='leaky')
    conv4_3 = conv_act_layer(conv4_2, 'conv4_3', 256, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    pool4 = mx.sym.Pooling(data=conv4_3, pool_type='max', kernel=(2, 2),
        stride=(2, 2), name='pool4')
    # group 5
    conv5_1 = conv_act_layer(pool4, 'conv5_1', 512, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    conv5_2 = conv_act_layer(conv5_1, 'conv5_2', 256, kernel=(1, 1), pad=(0, 0),
        act_type='leaky')
    conv5_3 = conv_act_layer(conv5_2, 'conv5_3', 512, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    conv5_4 = conv_act_layer(conv5_3, 'conv5_4', 256, kernel=(1, 1), pad=(0, 0),
        act_type='leaky')
    conv5_5 = conv_act_layer(conv5_4, 'conv5_5', 512, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    pool5 = mx.sym.Pooling(data=conv5_5, pool_type='max', kernel=(2, 2),
        stride=(2, 2), name='pool5')
    # group 6
    conv6_1 = conv_act_layer(pool5, 'conv6_1', 1024, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    conv6_2 = conv_act_layer(conv6_1, 'conv6_2', 512, kernel=(1, 1), pad=(0, 0),
        act_type='leaky')
    conv6_3 = conv_act_layer(conv6_2, 'conv6_3', 1024, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')
    conv6_4 = conv_act_layer(conv6_3, 'conv6_4', 512, kernel=(1, 1), pad=(0, 0),
        act_type='leaky')
    conv6_5 = conv_act_layer(conv6_4, 'conv6_5', 1024, kernel=(3, 3), pad=(1, 1),
        act_type='leaky')

    # class specific
    conv7 = mx.sym.Convolution(data=conv6_5, num_filter=num_classes, kernel=(1, 1),
        name='conv7')
    gpool = mx.sym.Pooling(data=conv7, pool_type='avg', global_pool=True,
        kernel=(7, 7), name='global_pool')
    return mx.symbol.SoftmaxOutput(data=gpool, name='softmax')