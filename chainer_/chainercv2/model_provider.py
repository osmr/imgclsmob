from .models.resnet import *
from .models.preresnet import *
from .models.resnext import *
from .models.senet import *
from .models.densenet import *
from .models.condensenet import *
from .models.dpn import *
from .models.darknet import *
from .models.squeezenet import *
from .models.squeezenext import *
from .models.shufflenet import *
from .models.shufflenetv2 import *
from .models.shufflenetv2b import *
from .models.menet import *
from .models.mobilenet import *
from .models.mobilenetv2 import *
from .models.xception import *
from .models.inceptionv3 import *
from .models.inceptionv4 import *
from .models.inceptionresnetv2 import *
from .models.polynet import *
from .models.nasnet import *

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

    'seresnet18': seresnet18,
    'seresnet34': seresnet34,
    'seresnet50': seresnet50,
    'seresnet50b': seresnet50b,
    'seresnet101': seresnet101,
    'seresnet101b': seresnet101b,
    'seresnet152': seresnet152,
    'seresnet152b': seresnet152b,
    'seresnet200': seresnet200,
    'seresnet200b': seresnet200b,

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

    'sepreresnet18': sepreresnet18,
    'sepreresnet34': sepreresnet34,
    'sepreresnet50': sepreresnet50,
    'sepreresnet50b': sepreresnet50b,
    'sepreresnet101': sepreresnet101,
    'sepreresnet101b': sepreresnet101b,
    'sepreresnet152': sepreresnet152,
    'sepreresnet152b': sepreresnet152b,
    'sepreresnet200': sepreresnet200,
    'sepreresnet200b': sepreresnet200b,

    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d,
    'seresnext50_32x4d': seresnext50_32x4d,
    'seresnext101_32x4d': seresnext101_32x4d,
    'seresnext101_64x4d': seresnext101_64x4d,

    'senet52': senet52,
    'senet103': senet103,
    'senet154': senet154,

    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,

    'condensenet74_c4_g4': condensenet74_c4_g4,
    'condensenet74_c8_g8': condensenet74_c8_g8,

    'dpn68': dpn68,
    'dpn68b': dpn68b,
    'dpn98': dpn98,
    'dpn107': dpn107,
    'dpn131': dpn131,

    'darknet_ref': darknet_ref,
    'darknet_tiny': darknet_tiny,
    'darknet19': darknet19,

    'squeezenet_v1_0': squeezenet_v1_0,
    'squeezenet_v1_1': squeezenet_v1_1,

    'squeezeresnet_v1_0': squeezeresnet_v1_0,
    'squeezeresnet_v1_1': squeezeresnet_v1_1,

    'sqnxt23_w1': sqnxt23_w1,
    'sqnxt23_w3d2': sqnxt23_w3d2,
    'sqnxt23_w2': sqnxt23_w2,
    'sqnxt23v5_w1': sqnxt23v5_w1,
    'sqnxt23v5_w3d2': sqnxt23v5_w3d2,
    'sqnxt23v5_w2': sqnxt23v5_w2,

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

    'shufflenetv2_wd2': shufflenetv2_wd2,
    'shufflenetv2_w1': shufflenetv2_w1,
    'shufflenetv2_w3d2': shufflenetv2_w3d2,
    'shufflenetv2_w2': shufflenetv2_w2,

    'shufflenetv2b_wd2': shufflenetv2b_wd2,
    'shufflenetv2b_w1': shufflenetv2b_w1,
    'shufflenetv2b_w3d2': shufflenetv2b_w3d2,
    'shufflenetv2b_w2': shufflenetv2b_w2,
    'shufflenetv2c_wd2': shufflenetv2c_wd2,
    'shufflenetv2c_w1': shufflenetv2c_w1,

    'menet108_8x1_g3': menet108_8x1_g3,
    'menet128_8x1_g4': menet128_8x1_g4,
    'menet160_8x1_g8': menet160_8x1_g8,
    'menet228_12x1_g3': menet228_12x1_g3,
    'menet256_12x1_g4': menet256_12x1_g4,
    'menet348_12x1_g3': menet348_12x1_g3,
    'menet352_12x1_g8': menet352_12x1_g8,
    'menet456_24x1_g3': menet456_24x1_g3,

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

    'xception': xception,
    'inceptionv3': inceptionv3,
    'inceptionv4': inceptionv4,
    'inceptionresnetv2': inceptionresnetv2,
    'polynet': polynet,

    'nasnet_4a1056': nasnet_4a1056,
    'nasnet_6a4032': nasnet_6a4032,
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    Chain
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError('Unsupported model: {}'.format(name))
    net = _models[name](**kwargs)
    return net
