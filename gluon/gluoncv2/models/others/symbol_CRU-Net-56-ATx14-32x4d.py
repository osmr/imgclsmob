import mxnet as mx
from .symbol_advanced import *

k_sec  = {  2:  3, \
            3:  4, \
            4:  6, \
            5:  3  }
R    = 32
bw1  = 256
k_D  = [128, 160]

def get_before_pool():
    data = mx.symbol.Variable(name="data")

    conv1_x_x  = Conv(data=data,  num_filter=64,  kernel=(7, 7), name='conv1_x_1', pad=(3,3), stride=(2,2))
    conv1_x_x  = BN_AC(conv1_x_x, name='conv1_x_1__relu-sp')
    conv1_x_x  = mx.symbol.Pooling(data=conv1_x_x, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

    bw = bw1
    D  = k_D[0]*(bw/bw1)
    conv2_x_x  = ResidualFactory(     conv1_x_x,   D,   D,   bw,   R,   'conv2_x__1',            None, 'proj'  )
    for i_ly in range(2, k_sec[2]+1):
        conv2_x_x  = ResidualFactory( conv2_x_x,   D,   D,   bw,   R,  ('conv2_x__%d'% i_ly),    None, 'normal')


    bw = 2*bw
    D  = k_D[0]*(bw/bw1)
    conv3_x_x  = ResidualFactory(     conv2_x_x,   D,   D,   bw,   R,   'conv3_x__1',            None, 'down'  )
    for i_ly in range(2, k_sec[3]+1):
        conv3_x_x  = ResidualFactory( conv3_x_x,   D,   D,   bw,   R,  ('conv3_x__%d'% i_ly),    None, 'normal')


    bw = 2*bw
    D  = k_D[1]*(bw/bw1)
    weights    = DeclearWeights(D, bw, 'conv4_x__x')
    conv4_x_x  = CRUFactory(          conv3_x_x,   D,   D,   bw,   D,   'conv4_x__1',         weights, 'down',   True)
    for i_ly in range(2, k_sec[4]+1):
        conv4_x_x  = CRUFactory(      conv4_x_x,   D,   D,   bw,   D,  ('conv4_x__%d'% i_ly), weights, 'normal', True)


    bw = 2*bw
    D  = k_D[0]*(bw/bw1)
    conv5_x_x  = ResidualFactory(     conv4_x_x,   D,   D,   bw,   R,   'conv5_x__1',            None, 'down'  )
    for i_ly in range(2, k_sec[5]+1):
        conv5_x_x  = ResidualFactory( conv5_x_x,   D,   D,   bw,   R,  ('conv5_x__%d'% i_ly),    None, 'normal')


    conv5_x_x = BN_AC(conv5_x_x, name='conv5_x_x__relu-sp')
    return conv5_x_x
    

def get_linear(num_classes = 1000):
    before_pool = get_before_pool()
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1,1), name="pool5")
    flat5     = mx.symbol.Flatten(data=pool5, name='flatten')
    fc6       = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='fc6')
    return fc6


def get_symbol(num_classes = 1000):
    fc6       = get_linear(num_classes)
    softmax   = mx.symbol.SoftmaxOutput(data=fc6,  name='softmax')
    sys_out   = softmax
    return sys_out








