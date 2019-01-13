import mxnet as mx
from .symbol_basic import *

def DeclearWeights(R, bw, name_prefix):
    # ---------------
    dim0 = R  # out 
    dim1 = bw # in
    dim2 = 1
    dim3 = 1
    weights_dim3  = mx.symbol.Variable(name=('%s_%dx%d_bases[dim3]_weight' %(name_prefix,dim2,dim3)),
                                          shape=(dim0,dim1,dim2,dim3))
    # ---------------
    dim0 = R 
    dim1 = 1
    dim2 = 3
    dim3 = 3
    weights_dim21 = mx.symbol.Variable(name=('%s_%dx%d_bases[dim21]_weight' %(name_prefix,dim2,dim3)),
                                          shape=(dim0,dim1,dim2,dim3))
    # ---------------
    weights = {'dim3':weights_dim3, 'dim21':weights_dim21}
    return weights

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Resudual Units
def ResidualFactory(data, num_1x1_a, num_3x3_b, num_1x1_c, R, name, weights, _type='normal', acc=False):

    # type
    if _type is 'proj':
        key_stride = 1
        has_proj   = True
    if _type is 'down':
        key_stride = 2
        has_proj   = True
    if _type is 'normal':
        key_stride = 1
        has_proj   = False

    # PROJ
    if has_proj:
        data_o   = data
        w_weight = mx.symbol.Variable(name=('%s_c1x1-w_weight'% name))
        c1x1_w   = BN_AC_Conv( data=data, num_filter=num_1x1_c, kernel=(1, 1), name=('%s_c1x1-w(s/%d)' %(name, key_stride)), pad=(0, 0), w=w_weight, stride=(key_stride, key_stride))
    else:
        data_o = data
        c1x1_w = data

    # FUNC
    c1x1_a = BN_AC_Conv( data=data_o, num_filter=num_1x1_a, kernel=( 1,  1), name=('%s_c1x1-a'  % name),  pad=( 0,  0))
    c3x3_b = BN_AC_Conv( data=c1x1_a, num_filter=num_3x3_b, kernel=( 3,  3), name=('%s_c3x3-b'  % name),  pad=( 1,  1), \
                         stride=(key_stride,key_stride), num_group=R)
    c1x1_c = BN_AC_Conv( data=c3x3_b, num_filter=num_1x1_c, kernel=( 1,  1), name=('%s_c1x1-c'  % name),  pad=( 0,  0))

    # OUTPUTS
    summ   = mx.symbol.ElementWiseSum(*[c1x1_w, c1x1_c],                     name=('%s_ele-sum' % name))
    return summ


# - - - - - - - - - - - - - - - - - - - - - - -
# Collective Resudual Units
def CRUFactory(data, num_1x1_a, num_3x3_b, num_1x1_c, R, name, weights, _type='normal', acc=False):

    # type
    if _type is 'proj':
        key_stride = 1
        has_proj   = True
    if _type is 'down':
        key_stride = 2
        has_proj   = True
    if _type is 'normal':
        key_stride = 1
        has_proj   = False

    # PROJ
    if has_proj:
        w_weight   = mx.symbol.Variable(name=('%s_c1x1-w_weight'% name))
        data_o     = BN_AC_Conv( data=data, num_filter=num_1x1_c, kernel=(1, 1), name=('%s_c1x1-w(s/1)'   % name), pad=(0, 0), w=w_weight)
        if key_stride > 1:
            c1x1_w = BN_AC_Conv( data=data, num_filter=num_1x1_c, kernel=(1, 1), name=('%s_c1x1-w(s/key)' % name), pad=(0, 0), w=w_weight, stride=(key_stride, key_stride))
        else:
            c1x1_w = data_o
    else:
        data_o = data
        c1x1_w = data

    # FUNC
    c1x1_a = BN_AC_Conv( data=data_o, num_filter=num_1x1_a, kernel=( 1,  1), name=('%s_c1x1-a'    % name), pad=( 0,  0), w=weights['dim3'])
    c3x3_b =       Conv( data=c1x1_a, num_filter=num_3x3_b, kernel=( 3,  3), name=('%s_c3x3-b(1)' % name), pad=( 1,  1), w=weights['dim21'], \
                         stride=(key_stride,key_stride), num_group=R, acc=acc)
    c1x1_b = BN_AC_Conv( data=c3x3_b, num_filter=num_3x3_b, kernel=( 1,  1), name=('%s_c1x1-b(2)' % name), pad=( 0,  0))
    c1x1_c = BN_AC_Conv( data=c1x1_b, num_filter=num_1x1_c, kernel=( 1,  1), name=('%s_c1x1-c'    % name), pad=( 0,  0))

    # OUTPUTS
    summ   = mx.symbol.ElementWiseSum(*[c1x1_w, c1x1_c],                     name=('%s_ele-sum'   % name))
    return summ


