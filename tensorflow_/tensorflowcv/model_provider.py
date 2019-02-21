from .models.alexnet import *
from .models.vgg import *
from .models.resnet import *
from .models.preresnet import *
from .models.resnext import *
from .models.seresnet import *
from .models.sepreresnet import *
from .models.seresnext import *
from .models.senet import *
from .models.densenet import *
from .models.darknet import *
from .models.darknet53 import *
from .models.channelnet import *
from .models.squeezenet import *
from .models.squeezenext import *
from .models.shufflenet import *
from .models.shufflenetv2 import *
from .models.shufflenetv2b import *
from .models.menet import *
from .models.mobilenet import *
from .models.mobilenetv2 import *
from .models.igcv3 import *
from .models.mnasnet import *

__all__ = ['get_model', 'init_variables_from_state_dict']


_models = {
    'alexnet': alexnet,

    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'bn_vgg11': bn_vgg11,
    'bn_vgg13': bn_vgg13,
    'bn_vgg16': bn_vgg16,
    'bn_vgg19': bn_vgg19,
    'bn_vgg11b': bn_vgg11b,
    'bn_vgg13b': bn_vgg13b,
    'bn_vgg16b': bn_vgg16b,
    'bn_vgg19b': bn_vgg19b,

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
    'preresnet269b': preresnet269b,

    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d,

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

    'darknet_ref': darknet_ref,
    'darknet_tiny': darknet_tiny,
    'darknet19': darknet19,
    'darknet53': darknet53,

    'channelnet': channelnet,

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

    'igcv3_w1': igcv3_w1,
    'igcv3_w3d4': igcv3_w3d4,
    'igcv3_wd2': igcv3_wd2,
    'igcv3_wd4': igcv3_wd4,

    'mnasnet': mnasnet,
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
    HybridBlock
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError('Unsupported model: {}'.format(name))
    net = _models[name](**kwargs)
    return net


def init_variables_from_state_dict(sess,
                                   state_dict,
                                   ignore_extra=True):
    """
    Initialize model variables from state dictionary.

    Parameters
    ----------
    sess: Session
        A Session to use to load the weights.
    state_dict : dict
        Dictionary with values of model variables.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    from .models.model_store import init_variables_from_state_dict
    init_variables_from_state_dict(
        sess=sess,
        state_dict=state_dict,
        ignore_extra=ignore_extra)
