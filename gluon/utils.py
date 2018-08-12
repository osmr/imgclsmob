from .models.resnet import *
from .models.preresnet import *
from .models.densenet import *
from .models.squeezenet import *
from .models.darknet import *
from .models.mobilenet import *
from .models.mobilenetv2 import *
from .models.shufflenet import *
from .models.menet import *
from .models.squeezenext import *

from .models.nasnet import *

# import .models.menet1 as gl_meneta
# from .models.squeezenext1 import *

__all__ = ['get_model']


_models = {
    'resnet10': resnet10,
    'resnet12': resnet12,
    'resnet14': resnet14,
    'resnet16': resnet16,
    'resnet18_wd4': resnet18_wd4,
    'resnet18_wd2': resnet18_wd2,
    'resnet18_w3d4': resnet18_w3d4,

    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet50b': resnet50b,
    'resnet101': resnet101,
    'resnet101b': resnet101b,
    'resnet152': resnet152,
    'resnet152b': resnet152b,
    'resnet200': resnet200,
    'resnet200b': resnet200b,

    'preresnet10': preresnet10,
    'preresnet12': preresnet12,
    'preresnet14': preresnet14,
    'preresnet16': preresnet16,
    'preresnet18_wd4': preresnet18_wd4,
    'preresnet18_wd2': preresnet18_wd2,
    'preresnet18_w3d4': preresnet18_w3d4,

    'preresnet18': preresnet18,
    'preresnet34': preresnet34,
    'preresnet50': preresnet50,
    'preresnet50b': preresnet50b,
    'preresnet101': preresnet101,
    'preresnet101b': preresnet101b,
    'preresnet152': preresnet152,
    'preresnet152b': preresnet152b,
    'preresnet200': preresnet200,
    'preresnet200b': preresnet200b,

    'slk_densenet121': densenet121,
    'slk_densenet161': densenet161,
    'slk_densenet169': densenet169,
    'slk_densenet201': densenet201,

    'squeezenet_v1_0': squeezenet_v1_0,
    'squeezenet_v1_1': squeezenet_v1_1,
    'squeezeresnet_v1_0': squeezeresnet_v1_0,
    'squeezeresnet_v1_1': squeezeresnet_v1_1,

    'darknet_ref': darknet_ref,
    'darknet_tiny': darknet_tiny,
    'darknet19': darknet19,

    'mobilenet_w1': mobilenet_w1,
    'mobilenet_w3d4': mobilenet_w3d4,
    'mobilenet_wd2': mobilenet_wd2,
    'mobilenet_wd4': mobilenet_wd4,

    'fdmobilenet_w1': fdmobilenet_w1,
    'fdmobilenet_w3d4': fdmobilenet_w3d4,
    'fdmobilenet_wd2': fdmobilenet_wd2,
    'fdmobilenet_wd4': fdmobilenet_wd4,

    'mobilenetv2_w1': mobilenetv2_w1,
    'mobilenetv2_w3d4': mobilenetv2_w3d4,
    'mobilenetv2_wd2': mobilenetv2_wd2,
    'mobilenetv2_wd4': mobilenetv2_wd4,

    'shufflenet_g1_w1': shufflenet_g1_w1,
    'shufflenet_g2_w1': shufflenet_g2_w1,
    'shufflenet_g3_w1': shufflenet_g3_w1,
    'shufflenet_g4_w1': shufflenet_g4_w1,
    'shufflenet_g8_w1': shufflenet_g8_w1,
    'shufflenet_g1_w3d4': shufflenet_g1_w3d4,
    'shufflenet_g3_w3d4': shufflenet_g3_w3d4,
    'shufflenet_g1_wd2': shufflenet_g1_wd2,
    'shufflenet_g3_wd2': shufflenet_g3_wd2,
    'shufflenet_g1_wd4': shufflenet_g1_wd4,
    'shufflenet_g3_wd4': shufflenet_g3_wd4,

    'menet108_8x1_g3': menet108_8x1_g3,
    'menet128_8x1_g4': menet128_8x1_g4,
    'menet160_8x1_g8': menet160_8x1_g8,
    'menet228_12x1_g3': menet228_12x1_g3,
    'menet256_12x1_g4': menet256_12x1_g4,
    'menet348_12x1_g3': menet348_12x1_g3,
    'menet352_12x1_g8': menet352_12x1_g8,
    'menet456_24x1_g3': menet456_24x1_g3,

    'sqnxt23_w1': sqnxt23_w1,
    'sqnxt23_w3d2': sqnxt23_w3d2,
    'sqnxt23_w2': sqnxt23_w2,
    'sqnxt23v5_w1': sqnxt23v5_w1,
    'sqnxt23v5_w3d2': sqnxt23v5_w3d2,
    'sqnxt23v5_w2': sqnxt23v5_w2,

    'nasnet_a_mobile': nasnet_a_mobile,

    # 'sqnxt23_1_0': sqnxt23_1_0,
    # 'sqnxt23_1_5': sqnxt23_1_5,
    # 'sqnxt23_2_0': sqnxt23_2_0,
    # 'sqnxt23v5_1_0': sqnxt23v5_1_0,
    # 'sqnxt23v5_1_5': sqnxt23v5_1_5,
    # 'sqnxt23v5_2_0': sqnxt23v5_2_0,

    # 'menet108_8x1_g3a': gl_meneta.menet108_8x1_g3,
    # 'menet128_8x1_g4a': gl_meneta.menet128_8x1_g4,
    # 'menet160_8x1_g8a': gl_meneta.menet160_8x1_g8,
}


def get_model(name, **kwargs):
    try:
        from gluoncv.model_zoo import get_model as glcv_get_model
        net = glcv_get_model(name, **kwargs)
        return net
    except ValueError as e:
        upstream_supported = str(e)
    name = name.lower()
    if name not in _models:
        raise ValueError('{}\n\t{}'.format(upstream_supported, '\n\t'.join(sorted(_models.keys()))))
    net = _models[name](**kwargs)
    return net

