from .models.alexnet import *
from .models.zfnet import *
from .models.vgg import *
from .models.bninception import *
from .models.resnet import *
from .models.preresnet import *
from .models.resnext import *
from .models.seresnet import *
from .models.sepreresnet import *
from .models.seresnext import *
from .models.senet import *
from .models.ibnresnet import *
from .models.ibnbresnet import *
from .models.ibnresnext import *
from .models.ibndensenet import *
from .models.airnet import *
from .models.airnext import *
from .models.bamresnet import *
from .models.cbamresnet import *
from .models.resattnet import *
from .models.sknet import *
from .models.diaresnet import *
from .models.diapreresnet import *
from .models.pyramidnet import *
from .models.diracnetv2 import *
from .models.sharesnet import *
from .models.densenet import *
from .models.condensenet import *
from .models.sparsenet import *
from .models.peleenet import *
from .models.wrn import *
from .models.drn import *
from .models.dpn import *
from .models.darknet import *
from .models.darknet53 import *
from .models.channelnet import *
from .models.isqrtcovresnet import *
from .models.revnet import *
from .models.irevnet import *
from .models.bagnet import *
from .models.dla import *
from .models.msdnet import *
from .models.fishnet import *
from .models.espnetv2 import *
from .models.xdensenet import *
from .models.squeezenet import *
from .models.squeezenext import *
from .models.shufflenet import *
from .models.shufflenetv2 import *
from .models.shufflenetv2b import *
from .models.menet import *
from .models.mobilenet import *
from .models.mobilenetv2 import *
from .models.mobilenetv3 import *
from .models.igcv3 import *
from .models.mnasnet import *
from .models.darts import *
from .models.proxylessnas import *
from .models.fbnet import *
from .models.xception import *
from .models.inceptionv3 import *
from .models.inceptionv4 import *
from .models.inceptionresnetv2 import *
from .models.polynet import *
from .models.nasnet import *
from .models.pnasnet import *
from .models.spnasnet import *
from .models.efficientnet import *
from .models.mixnet import *

from .models.nin_cifar import *
from .models.resnet_cifar import *
from .models.preresnet_cifar import *
from .models.resnext_cifar import *
from .models.seresnet_cifar import *
from .models.sepreresnet_cifar import *
from .models.pyramidnet_cifar import *
from .models.densenet_cifar import *
from .models.xdensenet_cifar import *
from .models.wrn_cifar import *
from .models.wrn1bit_cifar import *
from .models.ror_cifar import *
from .models.rir_cifar import *
from .models.msdnet_cifar10 import *
from .models.resdropresnet_cifar import *
from .models.shakeshakeresnet_cifar import *
from .models.shakedropresnet_cifar import *
from .models.fractalnet_cifar import *
from .models.diaresnet_cifar import *
from .models.diapreresnet_cifar import *

from .models.octresnet import *

from .models.resnetd import *

from .models.resnet_cub import *
from .models.seresnet_cub import *
from .models.mobilenet_cub import *
from .models.proxylessnas_cub import *
from .models.ntsnet_cub import *

from .models.fcn8sd import *
from .models.pspnet import *
from .models.deeplabv3 import *

from .models.superpointnet import *

# from .models.others.oth_gen_efficientnet import *

__all__ = ['get_model', 'trained_model_metainfo_list']


_models = {
    'alexnet': alexnet,
    'alexnetb': alexnetb,

    'zfnet': zfnet,
    'zfnetb': zfnetb,

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

    'bninception': bninception,

    'resnet10': resnet10,
    'resnet12': resnet12,
    'resnet14': resnet14,
    'resnetbc14b': resnetbc14b,
    'resnet16': resnet16,
    'resnet18_wd4': resnet18_wd4,
    'resnet18_wd2': resnet18_wd2,
    'resnet18_w3d4': resnet18_w3d4,
    'resnet18': resnet18,
    'resnet26': resnet26,
    'resnetbc26b': resnetbc26b,
    'resnet34': resnet34,
    'resnetbc38b': resnetbc38b,
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
    'preresnetbc14b': preresnetbc14b,
    'preresnet16': preresnet16,
    'preresnet18_wd4': preresnet18_wd4,
    'preresnet18_wd2': preresnet18_wd2,
    'preresnet18_w3d4': preresnet18_w3d4,
    'preresnet18': preresnet18,
    'preresnet26': preresnet26,
    'preresnetbc26b': preresnetbc26b,
    'preresnet34': preresnet34,
    'preresnetbc38b': preresnetbc38b,
    'preresnet50': preresnet50,
    'preresnet50b': preresnet50b,
    'preresnet101': preresnet101,
    'preresnet101b': preresnet101b,
    'preresnet152': preresnet152,
    'preresnet152b': preresnet152b,
    'preresnet200': preresnet200,
    'preresnet200b': preresnet200b,
    'preresnet269b': preresnet269b,

    'resnext14_16x4d': resnext14_16x4d,
    'resnext14_32x2d': resnext14_32x2d,
    'resnext14_32x4d': resnext14_32x4d,
    'resnext26_16x4d': resnext26_16x4d,
    'resnext26_32x2d': resnext26_32x2d,
    'resnext26_32x4d': resnext26_32x4d,
    'resnext38_32x4d': resnext38_32x4d,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d,

    'seresnet10': seresnet10,
    'seresnet12': seresnet12,
    'seresnet14': seresnet14,
    'seresnet16': seresnet16,
    'seresnet18': seresnet18,
    'seresnet26': seresnet26,
    'seresnetbc26b': seresnetbc26b,
    'seresnet34': seresnet34,
    'seresnetbc38b': seresnetbc38b,
    'seresnet50': seresnet50,
    'seresnet50b': seresnet50b,
    'seresnet101': seresnet101,
    'seresnet101b': seresnet101b,
    'seresnet152': seresnet152,
    'seresnet152b': seresnet152b,
    'seresnet200': seresnet200,
    'seresnet200b': seresnet200b,

    'sepreresnet10': sepreresnet10,
    'sepreresnet12': sepreresnet12,
    'sepreresnet14': sepreresnet14,
    'sepreresnet16': sepreresnet16,
    'sepreresnet18': sepreresnet18,
    'sepreresnet26': sepreresnet26,
    'sepreresnetbc26b': sepreresnetbc26b,
    'sepreresnet34': sepreresnet34,
    'sepreresnetbc38b': sepreresnetbc38b,
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

    'senet16': senet16,
    'senet28': senet28,
    'senet40': senet40,
    'senet52': senet52,
    'senet103': senet103,
    'senet154': senet154,

    'ibn_resnet50': ibn_resnet50,
    'ibn_resnet101': ibn_resnet101,
    'ibn_resnet152': ibn_resnet152,

    'ibnb_resnet50': ibnb_resnet50,
    'ibnb_resnet101': ibnb_resnet101,
    'ibnb_resnet152': ibnb_resnet152,

    'ibn_resnext50_32x4d': ibn_resnext50_32x4d,
    'ibn_resnext101_32x4d': ibn_resnext101_32x4d,
    'ibn_resnext101_64x4d': ibn_resnext101_64x4d,

    'ibn_densenet121': ibn_densenet121,
    'ibn_densenet161': ibn_densenet161,
    'ibn_densenet169': ibn_densenet169,
    'ibn_densenet201': ibn_densenet201,

    'airnet50_1x64d_r2': airnet50_1x64d_r2,
    'airnet50_1x64d_r16': airnet50_1x64d_r16,
    'airnet101_1x64d_r2': airnet101_1x64d_r2,

    'airnext50_32x4d_r2': airnext50_32x4d_r2,
    'airnext101_32x4d_r2': airnext101_32x4d_r2,
    'airnext101_32x4d_r16': airnext101_32x4d_r16,

    'bam_resnet18': bam_resnet18,
    'bam_resnet34': bam_resnet34,
    'bam_resnet50': bam_resnet50,
    'bam_resnet101': bam_resnet101,
    'bam_resnet152': bam_resnet152,

    'cbam_resnet18': cbam_resnet18,
    'cbam_resnet34': cbam_resnet34,
    'cbam_resnet50': cbam_resnet50,
    'cbam_resnet101': cbam_resnet101,
    'cbam_resnet152': cbam_resnet152,

    'resattnet56': resattnet56,
    'resattnet92': resattnet92,
    'resattnet128': resattnet128,
    'resattnet164': resattnet164,
    'resattnet200': resattnet200,
    'resattnet236': resattnet236,
    'resattnet452': resattnet452,

    'sknet50': sknet50,
    'sknet101': sknet101,
    'sknet152': sknet152,

    'diaresnet10': diaresnet10,
    'diaresnet12': diaresnet12,
    'diaresnet14': diaresnet14,
    'diaresnetbc14b': diaresnetbc14b,
    'diaresnet16': diaresnet16,
    'diaresnet18': diaresnet18,
    'diaresnet26': diaresnet26,
    'diaresnetbc26b': diaresnetbc26b,
    'diaresnet34': diaresnet34,
    'diaresnetbc38b': diaresnetbc38b,
    'diaresnet50': diaresnet50,
    'diaresnet50b': diaresnet50b,
    'diaresnet101': diaresnet101,
    'diaresnet101b': diaresnet101b,
    'diaresnet152': diaresnet152,
    'diaresnet152b': diaresnet152b,
    'diaresnet200': diaresnet200,
    'diaresnet200b': diaresnet200b,

    'diapreresnet10': diapreresnet10,
    'diapreresnet12': diapreresnet12,
    'diapreresnet14': diapreresnet14,
    'diapreresnetbc14b': diapreresnetbc14b,
    'diapreresnet16': diapreresnet16,
    'diapreresnet18': diapreresnet18,
    'diapreresnet26': diapreresnet26,
    'diapreresnetbc26b': diapreresnetbc26b,
    'diapreresnet34': diapreresnet34,
    'diapreresnetbc38b': diapreresnetbc38b,
    'diapreresnet50': diapreresnet50,
    'diapreresnet50b': diapreresnet50b,
    'diapreresnet101': diapreresnet101,
    'diapreresnet101b': diapreresnet101b,
    'diapreresnet152': diapreresnet152,
    'diapreresnet152b': diapreresnet152b,
    'diapreresnet200': diapreresnet200,
    'diapreresnet200b': diapreresnet200b,
    'diapreresnet269b': diapreresnet269b,

    'pyramidnet101_a360': pyramidnet101_a360,

    'diracnet18v2': diracnet18v2,
    'diracnet34v2': diracnet34v2,

    'sharesnet18': sharesnet18,
    'sharesnet34': sharesnet34,
    'sharesnet50': sharesnet50,
    'sharesnet50b': sharesnet50b,
    'sharesnet101': sharesnet101,
    'sharesnet101b': sharesnet101b,
    'sharesnet152': sharesnet152,
    'sharesnet152b': sharesnet152b,

    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,

    'condensenet74_c4_g4': condensenet74_c4_g4,
    'condensenet74_c8_g8': condensenet74_c8_g8,

    'sparsenet121': sparsenet121,
    'sparsenet161': sparsenet161,
    'sparsenet169': sparsenet169,
    'sparsenet201': sparsenet201,
    'sparsenet264': sparsenet264,

    'peleenet': peleenet,

    'wrn50_2': wrn50_2,

    'drnc26': drnc26,
    'drnc42': drnc42,
    'drnc58': drnc58,
    'drnd22': drnd22,
    'drnd38': drnd38,
    'drnd54': drnd54,
    'drnd105': drnd105,

    'dpn68': dpn68,
    'dpn68b': dpn68b,
    'dpn98': dpn98,
    'dpn107': dpn107,
    'dpn131': dpn131,

    'darknet_ref': darknet_ref,
    'darknet_tiny': darknet_tiny,
    'darknet19': darknet19,
    'darknet53': darknet53,

    'channelnet': channelnet,

    'revnet38': revnet38,
    'revnet110': revnet110,
    'revnet164': revnet164,

    'irevnet301': irevnet301,

    'bagnet9': bagnet9,
    'bagnet17': bagnet17,
    'bagnet33': bagnet33,

    'dla34': dla34,
    'dla46c': dla46c,
    'dla46xc': dla46xc,
    'dla60': dla60,
    'dla60x': dla60x,
    'dla60xc': dla60xc,
    'dla102': dla102,
    'dla102x': dla102x,
    'dla102x2': dla102x2,
    'dla169': dla169,

    'msdnet22': msdnet22,

    'fishnet99': fishnet99,
    'fishnet150': fishnet150,

    'espnetv2_wd2': espnetv2_wd2,
    'espnetv2_w1': espnetv2_w1,
    'espnetv2_w5d4': espnetv2_w5d4,
    'espnetv2_w3d2': espnetv2_w3d2,
    'espnetv2_w2': espnetv2_w2,

    'xdensenet121_2': xdensenet121_2,
    'xdensenet161_2': xdensenet161_2,
    'xdensenet169_2': xdensenet169_2,
    'xdensenet201_2': xdensenet201_2,

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

    'mobilenetv3_small_w7d20': mobilenetv3_small_w7d20,
    'mobilenetv3_small_wd2': mobilenetv3_small_wd2,
    'mobilenetv3_small_w3d4': mobilenetv3_small_w3d4,
    'mobilenetv3_small_w1': mobilenetv3_small_w1,
    'mobilenetv3_small_w5d4': mobilenetv3_small_w5d4,
    'mobilenetv3_large_w7d20': mobilenetv3_large_w7d20,
    'mobilenetv3_large_wd2': mobilenetv3_large_wd2,
    'mobilenetv3_large_w3d4': mobilenetv3_large_w3d4,
    'mobilenetv3_large_w1': mobilenetv3_large_w1,
    'mobilenetv3_large_w5d4': mobilenetv3_large_w5d4,

    'igcv3_w1': igcv3_w1,
    'igcv3_w3d4': igcv3_w3d4,
    'igcv3_wd2': igcv3_wd2,
    'igcv3_wd4': igcv3_wd4,

    'mnasnet': mnasnet,

    'darts': darts,

    'proxylessnas_cpu': proxylessnas_cpu,
    'proxylessnas_gpu': proxylessnas_gpu,
    'proxylessnas_mobile': proxylessnas_mobile,
    'proxylessnas_mobile14': proxylessnas_mobile14,

    'fbnet_cb': fbnet_cb,

    'xception': xception,
    'inceptionv3': inceptionv3,
    'inceptionv4': inceptionv4,
    'inceptionresnetv2': inceptionresnetv2,
    'polynet': polynet,

    'nasnet_4a1056': nasnet_4a1056,
    'nasnet_6a4032': nasnet_6a4032,

    'pnasnet5large': pnasnet5large,

    'spnasnet': spnasnet,

    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'efficientnet_b0b': efficientnet_b0b,
    'efficientnet_b1b': efficientnet_b1b,
    'efficientnet_b2b': efficientnet_b2b,
    'efficientnet_b3b': efficientnet_b3b,
    'efficientnet_b4b': efficientnet_b4b,
    'efficientnet_b5b': efficientnet_b5b,
    'efficientnet_b6b': efficientnet_b6b,
    'efficientnet_b7b': efficientnet_b7b,

    'mixnet_s': mixnet_s,
    'mixnet_m': mixnet_m,
    'mixnet_l': mixnet_l,

    'nin_cifar10': nin_cifar10,
    'nin_cifar100': nin_cifar100,
    'nin_svhn': nin_svhn,

    'resnet20_cifar10': resnet20_cifar10,
    'resnet20_cifar100': resnet20_cifar100,
    'resnet20_svhn': resnet20_svhn,
    'resnet56_cifar10': resnet56_cifar10,
    'resnet56_cifar100': resnet56_cifar100,
    'resnet56_svhn': resnet56_svhn,
    'resnet110_cifar10': resnet110_cifar10,
    'resnet110_cifar100': resnet110_cifar100,
    'resnet110_svhn': resnet110_svhn,
    'resnet164bn_cifar10': resnet164bn_cifar10,
    'resnet164bn_cifar100': resnet164bn_cifar100,
    'resnet164bn_svhn': resnet164bn_svhn,
    'resnet272bn_cifar10': resnet272bn_cifar10,
    'resnet272bn_cifar100': resnet272bn_cifar100,
    'resnet272bn_svhn': resnet272bn_svhn,
    'resnet542bn_cifar10': resnet542bn_cifar10,
    'resnet542bn_cifar100': resnet542bn_cifar100,
    'resnet542bn_svhn': resnet542bn_svhn,
    'resnet1001_cifar10': resnet1001_cifar10,
    'resnet1001_cifar100': resnet1001_cifar100,
    'resnet1001_svhn': resnet1001_svhn,
    'resnet1202_cifar10': resnet1202_cifar10,
    'resnet1202_cifar100': resnet1202_cifar100,
    'resnet1202_svhn': resnet1202_svhn,

    'preresnet20_cifar10': preresnet20_cifar10,
    'preresnet20_cifar100': preresnet20_cifar100,
    'preresnet20_svhn': preresnet20_svhn,
    'preresnet56_cifar10': preresnet56_cifar10,
    'preresnet56_cifar100': preresnet56_cifar100,
    'preresnet56_svhn': preresnet56_svhn,
    'preresnet110_cifar10': preresnet110_cifar10,
    'preresnet110_cifar100': preresnet110_cifar100,
    'preresnet110_svhn': preresnet110_svhn,
    'preresnet164bn_cifar10': preresnet164bn_cifar10,
    'preresnet164bn_cifar100': preresnet164bn_cifar100,
    'preresnet164bn_svhn': preresnet164bn_svhn,
    'preresnet272bn_cifar10': preresnet272bn_cifar10,
    'preresnet272bn_cifar100': preresnet272bn_cifar100,
    'preresnet272bn_svhn': preresnet272bn_svhn,
    'preresnet542bn_cifar10': preresnet542bn_cifar10,
    'preresnet542bn_cifar100': preresnet542bn_cifar100,
    'preresnet542bn_svhn': preresnet542bn_svhn,
    'preresnet1001_cifar10': preresnet1001_cifar10,
    'preresnet1001_cifar100': preresnet1001_cifar100,
    'preresnet1001_svhn': preresnet1001_svhn,
    'preresnet1202_cifar10': preresnet1202_cifar10,
    'preresnet1202_cifar100': preresnet1202_cifar100,
    'preresnet1202_svhn': preresnet1202_svhn,

    'resnext20_16x4d_cifar10': resnext20_16x4d_cifar10,
    'resnext20_16x4d_cifar100': resnext20_16x4d_cifar100,
    'resnext20_16x4d_svhn': resnext20_16x4d_svhn,
    'resnext20_32x2d_cifar10': resnext20_32x2d_cifar10,
    'resnext20_32x2d_cifar100': resnext20_32x2d_cifar100,
    'resnext20_32x2d_svhn': resnext20_32x2d_svhn,
    'resnext20_32x4d_cifar10': resnext20_32x4d_cifar10,
    'resnext20_32x4d_cifar100': resnext20_32x4d_cifar100,
    'resnext20_32x4d_svhn': resnext20_32x4d_svhn,
    'resnext29_32x4d_cifar10': resnext29_32x4d_cifar10,
    'resnext29_32x4d_cifar100': resnext29_32x4d_cifar100,
    'resnext29_32x4d_svhn': resnext29_32x4d_svhn,
    'resnext29_16x64d_cifar10': resnext29_16x64d_cifar10,
    'resnext29_16x64d_cifar100': resnext29_16x64d_cifar100,
    'resnext29_16x64d_svhn': resnext29_16x64d_svhn,
    'resnext272_1x64d_cifar10': resnext272_1x64d_cifar10,
    'resnext272_1x64d_cifar100': resnext272_1x64d_cifar100,
    'resnext272_1x64d_svhn': resnext272_1x64d_svhn,
    'resnext272_2x32d_cifar10': resnext272_2x32d_cifar10,
    'resnext272_2x32d_cifar100': resnext272_2x32d_cifar100,
    'resnext272_2x32d_svhn': resnext272_2x32d_svhn,

    'seresnet20_cifar10': seresnet20_cifar10,
    'seresnet20_cifar100': seresnet20_cifar100,
    'seresnet20_svhn': seresnet20_svhn,
    'seresnet56_cifar10': seresnet56_cifar10,
    'seresnet56_cifar100': seresnet56_cifar100,
    'seresnet56_svhn': seresnet56_svhn,
    'seresnet110_cifar10': seresnet110_cifar10,
    'seresnet110_cifar100': seresnet110_cifar100,
    'seresnet110_svhn': seresnet110_svhn,
    'seresnet164bn_cifar10': seresnet164bn_cifar10,
    'seresnet164bn_cifar100': seresnet164bn_cifar100,
    'seresnet164bn_svhn': seresnet164bn_svhn,
    'seresnet272bn_cifar10': seresnet272bn_cifar10,
    'seresnet272bn_cifar100': seresnet272bn_cifar100,
    'seresnet272bn_svhn': seresnet272bn_svhn,
    'seresnet542bn_cifar10': seresnet542bn_cifar10,
    'seresnet542bn_cifar100': seresnet542bn_cifar100,
    'seresnet542bn_svhn': seresnet542bn_svhn,
    'seresnet1001_cifar10': seresnet1001_cifar10,
    'seresnet1001_cifar100': seresnet1001_cifar100,
    'seresnet1001_svhn': seresnet1001_svhn,
    'seresnet1202_cifar10': seresnet1202_cifar10,
    'seresnet1202_cifar100': seresnet1202_cifar100,
    'seresnet1202_svhn': seresnet1202_svhn,

    'sepreresnet20_cifar10': sepreresnet20_cifar10,
    'sepreresnet20_cifar100': sepreresnet20_cifar100,
    'sepreresnet20_svhn': sepreresnet20_svhn,
    'sepreresnet56_cifar10': sepreresnet56_cifar10,
    'sepreresnet56_cifar100': sepreresnet56_cifar100,
    'sepreresnet56_svhn': sepreresnet56_svhn,
    'sepreresnet110_cifar10': sepreresnet110_cifar10,
    'sepreresnet110_cifar100': sepreresnet110_cifar100,
    'sepreresnet110_svhn': sepreresnet110_svhn,
    'sepreresnet164bn_cifar10': sepreresnet164bn_cifar10,
    'sepreresnet164bn_cifar100': sepreresnet164bn_cifar100,
    'sepreresnet164bn_svhn': sepreresnet164bn_svhn,
    'sepreresnet272bn_cifar10': sepreresnet272bn_cifar10,
    'sepreresnet272bn_cifar100': sepreresnet272bn_cifar100,
    'sepreresnet272bn_svhn': sepreresnet272bn_svhn,
    'sepreresnet542bn_cifar10': sepreresnet542bn_cifar10,
    'sepreresnet542bn_cifar100': sepreresnet542bn_cifar100,
    'sepreresnet542bn_svhn': sepreresnet542bn_svhn,
    'sepreresnet1001_cifar10': sepreresnet1001_cifar10,
    'sepreresnet1001_cifar100': sepreresnet1001_cifar100,
    'sepreresnet1001_svhn': sepreresnet1001_svhn,
    'sepreresnet1202_cifar10': sepreresnet1202_cifar10,
    'sepreresnet1202_cifar100': sepreresnet1202_cifar100,
    'sepreresnet1202_svhn': sepreresnet1202_svhn,

    'pyramidnet110_a48_cifar10': pyramidnet110_a48_cifar10,
    'pyramidnet110_a48_cifar100': pyramidnet110_a48_cifar100,
    'pyramidnet110_a48_svhn': pyramidnet110_a48_svhn,
    'pyramidnet110_a84_cifar10': pyramidnet110_a84_cifar10,
    'pyramidnet110_a84_cifar100': pyramidnet110_a84_cifar100,
    'pyramidnet110_a84_svhn': pyramidnet110_a84_svhn,
    'pyramidnet110_a270_cifar10': pyramidnet110_a270_cifar10,
    'pyramidnet110_a270_cifar100': pyramidnet110_a270_cifar100,
    'pyramidnet110_a270_svhn': pyramidnet110_a270_svhn,
    'pyramidnet164_a270_bn_cifar10': pyramidnet164_a270_bn_cifar10,
    'pyramidnet164_a270_bn_cifar100': pyramidnet164_a270_bn_cifar100,
    'pyramidnet164_a270_bn_svhn': pyramidnet164_a270_bn_svhn,
    'pyramidnet200_a240_bn_cifar10': pyramidnet200_a240_bn_cifar10,
    'pyramidnet200_a240_bn_cifar100': pyramidnet200_a240_bn_cifar100,
    'pyramidnet200_a240_bn_svhn': pyramidnet200_a240_bn_svhn,
    'pyramidnet236_a220_bn_cifar10': pyramidnet236_a220_bn_cifar10,
    'pyramidnet236_a220_bn_cifar100': pyramidnet236_a220_bn_cifar100,
    'pyramidnet236_a220_bn_svhn': pyramidnet236_a220_bn_svhn,
    'pyramidnet272_a200_bn_cifar10': pyramidnet272_a200_bn_cifar10,
    'pyramidnet272_a200_bn_cifar100': pyramidnet272_a200_bn_cifar100,
    'pyramidnet272_a200_bn_svhn': pyramidnet272_a200_bn_svhn,

    'densenet40_k12_cifar10': densenet40_k12_cifar10,
    'densenet40_k12_cifar100': densenet40_k12_cifar100,
    'densenet40_k12_svhn': densenet40_k12_svhn,
    'densenet40_k12_bc_cifar10': densenet40_k12_bc_cifar10,
    'densenet40_k12_bc_cifar100': densenet40_k12_bc_cifar100,
    'densenet40_k12_bc_svhn': densenet40_k12_bc_svhn,
    'densenet40_k24_bc_cifar10': densenet40_k24_bc_cifar10,
    'densenet40_k24_bc_cifar100': densenet40_k24_bc_cifar100,
    'densenet40_k24_bc_svhn': densenet40_k24_bc_svhn,
    'densenet40_k36_bc_cifar10': densenet40_k36_bc_cifar10,
    'densenet40_k36_bc_cifar100': densenet40_k36_bc_cifar100,
    'densenet40_k36_bc_svhn': densenet40_k36_bc_svhn,
    'densenet100_k12_cifar10': densenet100_k12_cifar10,
    'densenet100_k12_cifar100': densenet100_k12_cifar100,
    'densenet100_k12_svhn': densenet100_k12_svhn,
    'densenet100_k24_cifar10': densenet100_k24_cifar10,
    'densenet100_k24_cifar100': densenet100_k24_cifar100,
    'densenet100_k24_svhn': densenet100_k24_svhn,
    'densenet100_k12_bc_cifar10': densenet100_k12_bc_cifar10,
    'densenet100_k12_bc_cifar100': densenet100_k12_bc_cifar100,
    'densenet100_k12_bc_svhn': densenet100_k12_bc_svhn,
    'densenet190_k40_bc_cifar10': densenet190_k40_bc_cifar10,
    'densenet190_k40_bc_cifar100': densenet190_k40_bc_cifar100,
    'densenet190_k40_bc_svhn': densenet190_k40_bc_svhn,
    'densenet250_k24_bc_cifar10': densenet250_k24_bc_cifar10,
    'densenet250_k24_bc_cifar100': densenet250_k24_bc_cifar100,
    'densenet250_k24_bc_svhn': densenet250_k24_bc_svhn,

    'xdensenet40_2_k24_bc_cifar10': xdensenet40_2_k24_bc_cifar10,
    'xdensenet40_2_k24_bc_cifar100': xdensenet40_2_k24_bc_cifar100,
    'xdensenet40_2_k24_bc_svhn': xdensenet40_2_k24_bc_svhn,
    'xdensenet40_2_k36_bc_cifar10': xdensenet40_2_k36_bc_cifar10,
    'xdensenet40_2_k36_bc_cifar100': xdensenet40_2_k36_bc_cifar100,
    'xdensenet40_2_k36_bc_svhn': xdensenet40_2_k36_bc_svhn,

    'wrn16_10_cifar10': wrn16_10_cifar10,
    'wrn16_10_cifar100': wrn16_10_cifar100,
    'wrn16_10_svhn': wrn16_10_svhn,
    'wrn28_10_cifar10': wrn28_10_cifar10,
    'wrn28_10_cifar100': wrn28_10_cifar100,
    'wrn28_10_svhn': wrn28_10_svhn,
    'wrn40_8_cifar10': wrn40_8_cifar10,
    'wrn40_8_cifar100': wrn40_8_cifar100,
    'wrn40_8_svhn': wrn40_8_svhn,

    'wrn20_10_1bit_cifar10': wrn20_10_1bit_cifar10,
    'wrn20_10_1bit_cifar100': wrn20_10_1bit_cifar100,
    'wrn20_10_1bit_svhn': wrn20_10_1bit_svhn,
    'wrn20_10_32bit_cifar10': wrn20_10_32bit_cifar10,
    'wrn20_10_32bit_cifar100': wrn20_10_32bit_cifar100,
    'wrn20_10_32bit_svhn': wrn20_10_32bit_svhn,

    'ror3_56_cifar10': ror3_56_cifar10,
    'ror3_56_cifar100': ror3_56_cifar100,
    'ror3_56_svhn': ror3_56_svhn,
    'ror3_110_cifar10': ror3_110_cifar10,
    'ror3_110_cifar100': ror3_110_cifar100,
    'ror3_110_svhn': ror3_110_svhn,
    'ror3_164_cifar10': ror3_164_cifar10,
    'ror3_164_cifar100': ror3_164_cifar100,
    'ror3_164_svhn': ror3_164_svhn,

    'rir_cifar10': rir_cifar10,
    'rir_cifar100': rir_cifar100,
    'rir_svhn': rir_svhn,

    'msdnet22_cifar10': msdnet22_cifar10,

    'resdropresnet20_cifar10': resdropresnet20_cifar10,
    'resdropresnet20_cifar100': resdropresnet20_cifar100,
    'resdropresnet20_svhn': resdropresnet20_svhn,

    'shakeshakeresnet20_2x16d_cifar10': shakeshakeresnet20_2x16d_cifar10,
    'shakeshakeresnet20_2x16d_cifar100': shakeshakeresnet20_2x16d_cifar100,
    'shakeshakeresnet20_2x16d_svhn': shakeshakeresnet20_2x16d_svhn,
    'shakeshakeresnet26_2x32d_cifar10': shakeshakeresnet26_2x32d_cifar10,
    'shakeshakeresnet26_2x32d_cifar100': shakeshakeresnet26_2x32d_cifar100,
    'shakeshakeresnet26_2x32d_svhn': shakeshakeresnet26_2x32d_svhn,

    'shakedropresnet20_cifar10': shakedropresnet20_cifar10,
    'shakedropresnet20_cifar100': shakedropresnet20_cifar100,
    'shakedropresnet20_svhn': shakedropresnet20_svhn,

    'fractalnet_cifar10': fractalnet_cifar10,
    'fractalnet_cifar100': fractalnet_cifar100,

    'diaresnet20_cifar10': diaresnet20_cifar10,
    'diaresnet20_cifar100': diaresnet20_cifar100,
    'diaresnet20_svhn': diaresnet20_svhn,
    'diaresnet56_cifar10': diaresnet56_cifar10,
    'diaresnet56_cifar100': diaresnet56_cifar100,
    'diaresnet56_svhn': diaresnet56_svhn,
    'diaresnet110_cifar10': diaresnet110_cifar10,
    'diaresnet110_cifar100': diaresnet110_cifar100,
    'diaresnet110_svhn': diaresnet110_svhn,
    'diaresnet164bn_cifar10': diaresnet164bn_cifar10,
    'diaresnet164bn_cifar100': diaresnet164bn_cifar100,
    'diaresnet164bn_svhn': diaresnet164bn_svhn,
    'diaresnet1001_cifar10': diaresnet1001_cifar10,
    'diaresnet1001_cifar100': diaresnet1001_cifar100,
    'diaresnet1001_svhn': diaresnet1001_svhn,
    'diaresnet1202_cifar10': diaresnet1202_cifar10,
    'diaresnet1202_cifar100': diaresnet1202_cifar100,
    'diaresnet1202_svhn': diaresnet1202_svhn,

    'diapreresnet20_cifar10': diapreresnet20_cifar10,
    'diapreresnet20_cifar100': diapreresnet20_cifar100,
    'diapreresnet20_svhn': diapreresnet20_svhn,
    'diapreresnet56_cifar10': diapreresnet56_cifar10,
    'diapreresnet56_cifar100': diapreresnet56_cifar100,
    'diapreresnet56_svhn': diapreresnet56_svhn,
    'diapreresnet110_cifar10': diapreresnet110_cifar10,
    'diapreresnet110_cifar100': diapreresnet110_cifar100,
    'diapreresnet110_svhn': diapreresnet110_svhn,
    'diapreresnet164bn_cifar10': diapreresnet164bn_cifar10,
    'diapreresnet164bn_cifar100': diapreresnet164bn_cifar100,
    'diapreresnet164bn_svhn': diapreresnet164bn_svhn,
    'diapreresnet1001_cifar10': diapreresnet1001_cifar10,
    'diapreresnet1001_cifar100': diapreresnet1001_cifar100,
    'diapreresnet1001_svhn': diapreresnet1001_svhn,
    'diapreresnet1202_cifar10': diapreresnet1202_cifar10,
    'diapreresnet1202_cifar100': diapreresnet1202_cifar100,
    'diapreresnet1202_svhn': diapreresnet1202_svhn,

    'isqrtcovresnet18': isqrtcovresnet18,
    'isqrtcovresnet34': isqrtcovresnet34,
    'isqrtcovresnet50': isqrtcovresnet50,
    'isqrtcovresnet50b': isqrtcovresnet50b,
    'isqrtcovresnet101': isqrtcovresnet101,
    'isqrtcovresnet101b': isqrtcovresnet101b,

    'resnetd50b': resnetd50b,
    'resnetd101b': resnetd101b,
    'resnetd152b': resnetd152b,

    'octresnet10_ad2': octresnet10_ad2,
    'octresnet50b_ad2': octresnet50b_ad2,

    'resnet10_cub': resnet10_cub,
    'resnet12_cub': resnet12_cub,
    'resnet14_cub': resnet14_cub,
    'resnetbc14b_cub': resnetbc14b_cub,
    'resnet16_cub': resnet16_cub,
    'resnet18_cub': resnet18_cub,
    'resnet26_cub': resnet26_cub,
    'resnetbc26b_cub': resnetbc26b_cub,
    'resnet34_cub': resnet34_cub,
    'resnetbc38b_cub': resnetbc38b_cub,
    'resnet50_cub': resnet50_cub,
    'resnet50b_cub': resnet50b_cub,
    'resnet101_cub': resnet101_cub,
    'resnet101b_cub': resnet101b_cub,
    'resnet152_cub': resnet152_cub,
    'resnet152b_cub': resnet152b_cub,
    'resnet200_cub': resnet200_cub,
    'resnet200b_cub': resnet200b_cub,

    'seresnet10_cub': seresnet10_cub,
    'seresnet12_cub': seresnet12_cub,
    'seresnet14_cub': seresnet14_cub,
    'seresnetbc14b_cub': seresnetbc14b_cub,
    'seresnet16_cub': seresnet16_cub,
    'seresnet18_cub': seresnet18_cub,
    'seresnet26_cub': seresnet26_cub,
    'seresnetbc26b_cub': seresnetbc26b_cub,
    'seresnet34_cub': seresnet34_cub,
    'seresnetbc38b_cub': seresnetbc38b_cub,
    'seresnet50_cub': seresnet50_cub,
    'seresnet50b_cub': seresnet50b_cub,
    'seresnet101_cub': seresnet101_cub,
    'seresnet101b_cub': seresnet101b_cub,
    'seresnet152_cub': seresnet152_cub,
    'seresnet152b_cub': seresnet152b_cub,
    'seresnet200_cub': seresnet200_cub,
    'seresnet200b_cub': seresnet200b_cub,

    'mobilenet_w1_cub': mobilenet_w1_cub,
    'mobilenet_w3d4_cub': mobilenet_w3d4_cub,
    'mobilenet_wd2_cub': mobilenet_wd2_cub,
    'mobilenet_wd4_cub': mobilenet_wd4_cub,

    'fdmobilenet_w1_cub': fdmobilenet_w1_cub,
    'fdmobilenet_w3d4_cub': fdmobilenet_w3d4_cub,
    'fdmobilenet_wd2_cub': fdmobilenet_wd2_cub,
    'fdmobilenet_wd4_cub': fdmobilenet_wd4_cub,

    'proxylessnas_cpu_cub': proxylessnas_cpu_cub,
    'proxylessnas_gpu_cub': proxylessnas_gpu_cub,
    'proxylessnas_mobile_cub': proxylessnas_mobile_cub,
    'proxylessnas_mobile14_cub': proxylessnas_mobile14_cub,

    'ntsnet_cub': ntsnet_cub,

    'fcn8sd_resnetd50b_voc': fcn8sd_resnetd50b_voc,
    'fcn8sd_resnetd101b_voc': fcn8sd_resnetd101b_voc,
    'fcn8sd_resnetd50b_coco': fcn8sd_resnetd50b_coco,
    'fcn8sd_resnetd101b_coco': fcn8sd_resnetd101b_coco,
    'fcn8sd_resnetd50b_ade20k': fcn8sd_resnetd50b_ade20k,
    'fcn8sd_resnetd101b_ade20k': fcn8sd_resnetd101b_ade20k,
    'fcn8sd_resnetd50b_cityscapes': fcn8sd_resnetd50b_cityscapes,
    'fcn8sd_resnetd101b_cityscapes': fcn8sd_resnetd101b_cityscapes,

    'pspnet_resnetd50b_voc': pspnet_resnetd50b_voc,
    'pspnet_resnetd101b_voc': pspnet_resnetd101b_voc,
    'pspnet_resnetd50b_coco': pspnet_resnetd50b_coco,
    'pspnet_resnetd101b_coco': pspnet_resnetd101b_coco,
    'pspnet_resnetd50b_ade20k': pspnet_resnetd50b_ade20k,
    'pspnet_resnetd101b_ade20k': pspnet_resnetd101b_ade20k,
    'pspnet_resnetd50b_cityscapes': pspnet_resnetd50b_cityscapes,
    'pspnet_resnetd101b_cityscapes': pspnet_resnetd101b_cityscapes,

    'deeplabv3_resnetd50b_voc': deeplabv3_resnetd50b_voc,
    'deeplabv3_resnetd101b_voc': deeplabv3_resnetd101b_voc,
    'deeplabv3_resnetd152b_voc': deeplabv3_resnetd152b_voc,
    'deeplabv3_resnetd50b_coco': deeplabv3_resnetd50b_coco,
    'deeplabv3_resnetd101b_coco': deeplabv3_resnetd101b_coco,
    'deeplabv3_resnetd152b_coco': deeplabv3_resnetd152b_coco,
    'deeplabv3_resnetd50b_ade20k': deeplabv3_resnetd50b_ade20k,
    'deeplabv3_resnetd101b_ade20k': deeplabv3_resnetd101b_ade20k,
    'deeplabv3_resnetd50b_cityscapes': deeplabv3_resnetd50b_cityscapes,
    'deeplabv3_resnetd101b_cityscapes': deeplabv3_resnetd101b_cityscapes,

    'superpointnet': superpointnet,
}

trained_model_metainfo_list = (
    ('alexnet', 'AlexNet', '1404.5997', 224, 0.875, 200, 'pytorch'),
    ('alexnetb', 'AlexNet-b', '1404.5997', 224, 0.875, 200, 'pytorch'),
    ('zfnet', 'ZFNet', '1311.2901', 224, 0.875, 200, 'pytorch'),
    ('zfnetb', 'ZFNet-b', '1311.2901', 224, 0.875, 200, 'pytorch'),
    ('vgg11', 'VGG-11', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('vgg13', 'VGG-13', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('vgg16', 'VGG-16', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('vgg19', 'VGG-19', '1409.1556', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('bn_vgg11', 'BN-VGG-11', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('bn_vgg13', 'BN-VGG-13', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('bn_vgg16', 'BN-VGG-16', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('bn_vgg19', 'BN-VGG-19', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('bn_vgg11b', 'BN-VGG-11b', '1409.1556', 224, 0.875, 200, 'pytorch'),
    ('bn_vgg13b', 'BN-VGG-13b', '1409.1556', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('bn_vgg16b', 'BN-VGG-16b', '1409.1556', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('bn_vgg19b', 'BN-VGG-19b', '1409.1556', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('bninception', 'BN-Inception', '1502.03167', 224, 0.875, 200, 'pytorch'),
    ('resnet10', 'ResNet-10', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet12', 'ResNet-12', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet14', 'ResNet-14', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet16', 'ResNet-16', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet18', 'ResNet-18', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet26', 'ResNet-26', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnetbc26b', 'ResNet-BC-26b', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet34', 'ResNet-34', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnetbc38b', 'ResNet-BC-38b', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet50', 'ResNet-50', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet50b', 'ResNet-50b', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet101', 'ResNet-101', '1512.03385', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('resnet101b', 'ResNet-101b', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('resnet152', 'ResNet-152', '1512.03385', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('resnet152b', 'ResNet-152b', '1512.03385', 224, 0.875, 200, 'pytorch'),
    ('preresnet10', 'PrepResNet-10', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet12', 'PreResNet-12', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet14', 'PreResNet-14', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet16', 'PreResNet-16', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet18', 'PreResNet-18', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet26', 'PreResNet-26', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnetbc26b', 'PreResNet-BC-26b', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet34', 'PreResNet-34', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnetbc38b', 'PreResNet-BC-38b', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet50', 'PreResNet-50', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet50b', 'PreResNet-50b', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet101', 'PreResNet-101', '1603.05027', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('preresnet101b', 'PreResNet-101b', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet152', 'PreResNet-152', '1603.05027', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('preresnet152b', 'PreResNet-152b', '1603.05027', 224, 0.875, 200, 'pytorch'),
    ('preresnet200b', 'PreResNet-200b', '1603.05027', 224, 0.875, 200, 'pytorch, from [tornadomeet/ResNet]'),
    ('preresnet269b', 'PreResNet-269b', '1603.05027', 224, 0.875, 200, 'pytorch, from [soeaver/mxnet-model]'),
    ('resnext14_16x4d', 'ResNeXt-14 (16x4d)', '1611.05431', 224, 0.875, 200, 'pytorch'),
    ('resnext14_32x2d', 'ResNeXt-14 (32x2d)', '1611.05431', 224, 0.875, 200, 'pytorch'),
    ('resnext14_32x4d', 'ResNeXt-14 (32x4d)', '1611.05431', 224, 0.875, 200, 'pytorch'),
    ('resnext26_32x2d', 'ResNeXt-26 (32x2d)', '1611.05431', 224, 0.875, 200, 'pytorch'),
    ('resnext26_32x4d', 'ResNeXt-26 (32x4d)', '1611.05431', 224, 0.875, 200, 'pytorch'),
    ('resnext101_32x4d', 'ResNeXt-101 (32x4d)', '1611.05431', 224, 0.875, 200,
     'pytorch, from [Cadene/pretrained...pytorch]'),
    ('resnext101_64x4d', 'ResNeXt-101 (64x4d)', '1611.05431', 224, 0.875, 200,
     'pytorch, from [Cadene/pretrained...pytorch]'),
    ('seresnet10', 'SE-ResNet-10', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('seresnet18', 'SE-ResNet-18', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('seresnet26', 'SE-ResNet-26', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('seresnetbc26b', 'SE-ResNet-BC-26b', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('seresnetbc38b', 'SE-ResNet-BC-38b', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('seresnet50', 'SE-ResNet-50', '1709.01507', 224, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('seresnet50b', 'SE-ResNet-50b', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('seresnet101', 'SE-ResNet-101', '1709.01507', 224, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('seresnet152', 'SE-ResNet-152', '1709.01507', 224, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('sepreresnet10', 'SE-PreResNet-10', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('sepreresnet18', 'SE-PreResNet-18', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('sepreresnetbc26b', 'SE-PreResNet-BC-26b', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('sepreresnetbc38b', 'SE-PreResNet-BC-38b', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('seresnext50_32x4d', 'SE-ResNeXt-50 (32x4d)', '1709.01507', 224, 0.875, 200,
     'pytorch, from [Cadene/pretrained...pytorch]'),
    ('seresnext101_32x4d', 'SE-ResNeXt-101 (32x4d)', '1709.01507', 224, 0.875, 200,
     'pytorch, from [Cadene/pretrained...pytorch]'),
    ('senet16', 'SENet-16', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('senet28', 'SENet-28', '1709.01507', 224, 0.875, 200, 'pytorch'),
    ('senet154', 'SENet-154', '1709.01507', 224, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('ibn_resnet50', 'IBN-ResNet-50', '1807.09441', 224, 0.875, 200, 'pytorch,  from [XingangPan/IBN-Net]'),
    ('ibn_resnet101', 'IBN-ResNet-101', '1807.09441', 224, 0.875, 200, 'pytorch,  from [XingangPan/IBN-Net]'),
    ('ibnb_resnet50', 'IBN(b)-ResNet-50', '1807.09441', 224, 0.875, 200, 'pytorch,  from [XingangPan/IBN-Net]'),
    ('ibn_resnext101_32x4d', 'IBN-ResNeXt-101 (32x4d)', '1807.09441', 224, 0.875, 200,
     'pytorch,  from [XingangPan/IBN-Net]'),
    ('ibn_densenet121', 'IBN-DenseNet-121', '1807.09441', 224, 0.875, 200, 'pytorch,  from [XingangPan/IBN-Net]'),
    ('ibn_densenet169', 'IBN-DenseNet-169', '1807.09441', 224, 0.875, 200, 'pytorch,  from [XingangPan/IBN-Net]'),
    ('airnet50_1x64d_r2', 'AirNet50-1x64d (r=2)', '', 224, 0.875, 200, 'pytorch, from [soeaver/AirNet-PyTorch]'),
    ('airnet50_1x64d_r16', 'AirNet50-1x64d (r=16)', '', 224, 0.875, 200, 'pytorch, from [soeaver/AirNet-PyTorch]'),
    ('airnext50_32x4d_r2', 'AirNeXt50-32x4d (r=2)', '', 224, 0.875, 200, 'pytorch, from [soeaver/AirNet-PyTorch]'),
    ('bam_resnet50', 'BAM-ResNet-50', '1807.06514', 224, 0.875, 200, 'pytorch, from [Jongchan/attention-module]'),
    ('cbam_resnet50', 'CBAM-ResNet-50', '1807.06521', 224, 0.875, 200, 'pytorch, from [Jongchan/attention-module]'),
    ('pyramidnet101_a360', 'PyramidNet-101 (a=360)', '1610.02915', 224, 0.875, 200,
     'pytorch, from [dyhan0920/Pyramid...PyTorch]'),
    ('diracnet18v2', 'DiracNetV2-18', '1706.00388', 224, 0.875, 200, 'pytorch, from [szagoruyko/diracnets]'),
    ('diracnet34v2', 'DiracNetV2-34', '1706.00388', 224, 0.875, 200, 'pytorch, from [szagoruyko/diracnets]'),
    ('densenet121', 'DenseNet-121', '1608.06993', 224, 0.875, 200, 'pytorch'),
    ('densenet161', 'DenseNet-161', '1608.06993', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('densenet169', 'DenseNet-169', '1608.06993', 224, 0.875, 200, 'pytorch'),
    ('densenet201', 'DenseNet-201', '1608.06993', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('condensenet74_c4_g4', 'CondenseNet-74 (C=G=4)', '1711.09224', 224, 0.875, 200,
     'pytorch, from [ShichenLiu/CondenseNet]'),
    ('condensenet74_c8_g8', 'CondenseNet-74 (C=G=8)', '1711.09224', 224, 0.875, 200,
     'pytorch, from [ShichenLiu/CondenseNet]'),
    ('peleenet', 'PeleeNet', '1804.06882', 224, 0.875, 200, 'pytorch'),
    ('wrn50_2', 'WRN-50-2', '1605.07146', 224, 0.875, 200, 'pytorch, from [szagoruyko/functional-zoo]'),
    ('drnc26', 'DRN-C-26', '1705.09914', 224, 0.875, 200, 'pytorch, from [fyu/drn]'),
    ('drnc42', 'DRN-C-42', '1705.09914', 224, 0.875, 200, 'pytorch, from [fyu/drn]'),
    ('drnc58', 'DRN-C-58', '1705.09914', 224, 0.875, 200, 'pytorch, from [fyu/drn]'),
    ('drnd22', 'DRN-D-22', '1705.09914', 224, 0.875, 200, 'pytorch, from [fyu/drn]'),
    ('drnd38', 'DRN-D-38', '1705.09914', 224, 0.875, 200, 'pytorch, from [fyu/drn]'),
    ('drnd54', 'DRN-D-54', '1705.09914', 224, 0.875, 200, 'pytorch, from [fyu/drn]'),
    ('drnd105', 'DRN-D-105', '1705.09914', 224, 0.875, 200, 'pytorch, from [fyu/drn]'),
    ('dpn68', 'DPN-68', '1707.01629', 224, 0.875, 200, 'pytorch'),
    ('dpn98', 'DPN-98', '1707.01629', 224, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('dpn131', 'DPN-131', '1707.01629', 224, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('darknet_tiny', 'DarkNet Tiny', '', 224, 0.875, 200, 'pytorch'),
    ('darknet_ref', 'DarkNet Ref', '', 224, 0.875, 200, 'pytorch'),
    ('darknet53', 'DarkNet-53', '1804.02767', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('irevnet301', 'i-RevNet-301', '1802.07088', 224, 0.875, 200, 'pytorch, from [jhjacobsen/pytorch-i-revnet]'),
    ('bagnet9', 'BagNet-9', '', 224, 0.875, 200, 'pytorch, from [wielandbrendel/bag...models]'),
    ('bagnet17', 'BagNet-17', '', 224, 0.875, 200, 'pytorch, from [wielandbrendel/bag...models]'),
    ('bagnet33', 'BagNet-33', '', 224, 0.875, 200, 'pytorch, from [wielandbrendel/bag...models]'),
    ('dla34', 'DLA-34', '1707.06484', 224, 0.875, 200, 'pytorch, from [ucbdrive/dla]'),
    ('dla46c', 'DLA-46-C', '1707.06484', 224, 0.875, 200, 'pytorch'),
    ('dla46xc', 'DLA-X-46-C', '1707.06484', 224, 0.875, 200, 'pytorch'),
    ('dla60', 'DLA-60', '1707.06484', 224, 0.875, 200, 'pytorch, from [ucbdrive/dla]'),
    ('dla60x', 'DLA-X-60', '1707.06484', 224, 0.875, 200, 'pytorch, from [ucbdrive/dla]'),
    ('dla60xc', 'DLA-X-60-C', '1707.06484', 224, 0.875, 200, 'pytorch'),
    ('dla102', 'DLA-102', '1707.06484', 224, 0.875, 200, 'pytorch, from [ucbdrive/dla]'),
    ('dla102x', 'DLA-X-102', '1707.06484', 224, 0.875, 200, 'pytorch, from [ucbdrive/dla]'),
    ('dla102x2', 'DLA-X2-102', '1707.06484', 224, 0.875, 200, 'pytorch, from [ucbdrive/dla]'),
    ('dla169', 'DLA-169', '1707.06484', 224, 0.875, 200, 'pytorch, from [ucbdrive/dla]'),
    ('fishnet150', 'FishNet-150', '', 224, 0.875, 200, 'pytorch, from [kevin-ssy/FishNet]'),
    ('espnetv2_wd2', 'ESPNetv2 x0.5', '1811.11431', 224, 0.875, 200, 'pytorch, from [sacmehta/ESPNetv2]'),
    ('espnetv2_w1', 'ESPNetv2 x1.0', '1811.11431', 224, 0.875, 200, 'pytorch, from [sacmehta/ESPNetv2]'),
    ('espnetv2_w5d4', 'ESPNetv2 x1.25', '1811.11431', 224, 0.875, 200, 'pytorch, from [sacmehta/ESPNetv2]'),
    ('espnetv2_w3d2', 'ESPNetv2 x1.5', '1811.11431', 224, 0.875, 200, 'pytorch, from [sacmehta/ESPNetv2]'),
    ('espnetv2_w2', 'ESPNetv2 x2.0', '1811.11431', 224, 0.875, 200, 'pytorch, from [sacmehta/ESPNetv2]'),
    ('squeezenet_v1_0', 'SqueezeNet v1.0', '1602.07360', 224, 0.875, 200, 'pytorch'),
    ('squeezenet_v1_1', 'SqueezeNet v1.1', '1602.07360', 224, 0.875, 200, 'pytorch'),
    ('squeezeresnet_v1_0', 'SqueezeResNet v1.0', '1602.07360', 224, 0.875, 200, 'pytorch'),
    ('squeezeresnet_v1_1', 'SqueezeResNet v1.1', '1602.07360', 224, 0.875, 200, 'pytorch'),
    ('sqnxt23_w1', '1.0-SqNxt-23', '1803.10615', 224, 0.875, 200, 'pytorch'),
    ('sqnxt23v5_w1', '1.0-SqNxt-23v5', '1803.10615', 224, 0.875, 200, 'pytorch'),
    ('sqnxt23_w3d2', '1.5-SqNxt-23', '1803.10615', 224, 0.875, 200, 'pytorch'),
    ('sqnxt23v5_w3d2', '1.5-SqNxt-23v5', '1803.10615', 224, 0.875, 200, 'pytorch'),
    ('sqnxt23_w2', '2.0-SqNxt-23', '1803.10615', 224, 0.875, 200, 'pytorch'),
    ('sqnxt23v5_w2', '2.0-SqNxt-23v5', '1803.10615', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g1_wd4', 'ShuffleNet x0.25 (g=1)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g3_wd4', 'ShuffleNet x0.25 (g=3)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g1_wd2', 'ShuffleNet x0.5 (g=1)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g3_wd2', 'ShuffleNet x0.5 (g=3)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g1_w3d4', 'ShuffleNet x0.75 (g=1)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g3_w3d4', 'ShuffleNet x0.75 (g=3)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g1_w1', 'ShuffleNet x1.0 (g=1)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g2_w1', 'ShuffleNet x1.0 (g=2)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g3_w1', 'ShuffleNet x1.0 (g=3)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g4_w1', 'ShuffleNet x1.0 (g=4)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenet_g8_w1', 'ShuffleNet x1.0 (g=8)', '1707.01083', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2_wd2', 'ShuffleNetV2 x0.5', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2_w1', 'ShuffleNetV2 x1.0', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2_w3d2', 'ShuffleNetV2 x1.5', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2_w2', 'ShuffleNetV2 x2.0', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2b_wd2', 'ShuffleNetV2b x0.5', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2b_w1', 'ShuffleNetV2b x1.0', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2b_w3d2', 'ShuffleNetV2b x1.5', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('shufflenetv2b_w2', 'ShuffleNetV2b x2.0', '1807.11164', 224, 0.875, 200, 'pytorch'),
    ('menet108_8x1_g3', '108-MENet-8x1 (g=3)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('menet128_8x1_g4', '128-MENet-8x1 (g=4)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('menet160_8x1_g8', '160-MENet-8x1 (g=8)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('menet228_12x1_g3', '228-MENet-12x1 (g=3)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('menet256_12x1_g4', '256-MENet-12x1 (g=4)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('menet348_12x1_g3', '348-MENet-12x1 (g=3)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('menet352_12x1_g8', '352-MENet-12x1 (g=8)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('menet456_24x1_g3', '456-MENet-24x1 (g=3)', '1803.09127', 224, 0.875, 200, 'pytorch'),
    ('mobilenet_wd4', 'MobileNet x0.25', '1704.04861', 224, 0.875, 200, 'pytorch'),
    ('mobilenet_wd2', 'MobileNet x0.5', '1704.04861', 224, 0.875, 200, 'pytorch'),
    ('mobilenet_w3d4', 'MobileNet x0.75', '1704.04861', 224, 0.875, 200, 'pytorch'),
    ('mobilenet_w1', 'MobileNet x1.0', '1704.04861', 224, 0.875, 200, 'pytorch'),
    ('fdmobilenet_wd4', 'FD-MobileNet x0.25', '1802.03750', 224, 0.875, 200, 'pytorch'),
    ('fdmobilenet_wd2', 'FD-MobileNet x0.5', '1802.03750', 224, 0.875, 200, 'pytorch'),
    ('fdmobilenet_w3d4', 'FD-MobileNet x0.75', '1802.03750', 224, 0.875, 200, 'pytorch'),
    ('fdmobilenet_w1', 'FD-MobileNet x1.0', '1802.03750', 224, 0.875, 200, 'pytorch'),
    ('mobilenetv2_wd4', 'MobileNetV2 x0.25', '1801.04381', 224, 0.875, 200, 'pytorch'),
    ('mobilenetv2_wd2', 'MobileNetV2 x0.5', '1801.04381', 224, 0.875, 200, 'pytorch'),
    ('mobilenetv2_w3d4', 'MobileNetV2 x0.75', '1801.04381', 224, 0.875, 200, 'pytorch'),
    ('mobilenetv2_w1', 'MobileNetV2 x1.0', '1801.04381', 224, 0.875, 200, 'pytorch'),
    ('mobilenetv3_large_w1', 'MobileNetV3 L/224/1.0', '1905.02244', 224, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('igcv3_wd4', 'IGCV3 x0.25', '1806.00178', 224, 0.875, 200, 'pytorch'),
    ('igcv3_wd2', 'IGCV3 x0.5', '1806.00178', 224, 0.875, 200, 'pytorch'),
    ('igcv3_w3d4', 'IGCV3 x0.75', '1806.00178', 224, 0.875, 200, 'pytorch'),
    ('igcv3_w1', 'IGCV3 x1.0', '1806.00178', 224, 0.875, 200, 'pytorch'),
    ('mnasnet', 'MnasNet', '1807.11626', 224, 0.875, 200, 'pytorch, from [zeusees/Mnasnet...Model]'),
    ('darts', 'DARTS', '1806.09055', 224, 0.875, 200, 'pytorch, from [quark0/darts]'),
    ('proxylessnas_cpu', 'ProxylessNAS CPU', '1812.00332', 224, 0.875, 200, 'pytorch, from [MIT-HAN-LAB/ProxylessNAS]'),
    ('proxylessnas_gpu', 'ProxylessNAS GPU', '1812.00332', 224, 0.875, 200, 'pytorch'),
    ('proxylessnas_mobile', 'ProxylessNAS Mobile', '1812.00332', 224, 0.875, 200,
     'pytorch, from [MIT-HAN-LAB/ProxylessNAS]'),
    ('proxylessnas_mobile14', 'ProxylessNAS Mob-14', '1812.00332', 224, 0.875, 200, 'pytorch'),
    ('fbnet_cb', 'FBNet-Cb', '1812.03443', 224, 0.875, 200, 'pytorch, from [rwightman/pyt...models]'),
    ('xception', 'Xception', '1610.02357', 299, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('inceptionv3', 'InceptionV3', '1512.00567', 299, 0.875, 200, 'pytorch, from [dmlc/gluon-cv]'),
    ('inceptionv4', 'InceptionV4', '1602.07261', 299, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('inceptionresnetv2', 'InceptionResNetV', '1602.07261', 299, 0.875, 200,
     'pytorch, from [Cadene/pretrained...pytorch]'),
    ('polynet', 'PolyNet', '1611.05725', 331, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('nasnet_4a1056', 'NASNet-A 4@1056', '1707.07012', 224, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('nasnet_6a4032', 'NASNet-A 6@4032', '1707.07012', 331, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('pnasnet5large', 'PNASNet-5-Large', '1712.00559', 331, 0.875, 200, 'pytorch, from [Cadene/pretrained...pytorch]'),
    ('spnasnet', 'SPNASNet', '1904.02877', 224, 0.875, 200, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b0', 'EfficientNet-B0', '1905.11946', 224, 0.875, 200, 'pytorch'),
    ('efficientnet_b1', 'EfficientNet-B1', '1905.11946', 240, 0.882, 200, 'pytorch'),
    ('efficientnet_b0b', 'EfficientNet-B0b', '1905.11946', 224, 0.875, 200, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b1b', 'EfficientNet-B1b', '1905.11946', 240, 0.882, 200, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b2b', 'EfficientNet-B2b', '1905.11946', 260, 0.890, 100, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b3b', 'EfficientNet-B3b', '1905.11946', 300, 0.904, 90, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b4b', 'EfficientNet-B4b', '1905.11946', 380, 0.922, 80, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b5b', 'EfficientNet-B5b', '1905.11946', 456, 0.934, 70, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b6b', 'EfficientNet-B6b', '1905.11946', 528, 0.942, 60, 'pytorch, from [rwightman/pyt...models]'),
    ('efficientnet_b7b', 'EfficientNet-B7b', '1905.11946', 600, 0.949, 50, 'pytorch, from [rwightman/pyt...models]'),
    ('mixnet_s', 'MixNet-S', '1907.09595', 224, 0.875, 200, 'pytorch, from [rwightman/pyt...models]'),
    ('mixnet_m', 'MixNet-M', '1907.09595', 224, 0.875, 200, 'pytorch, from [rwightman/pyt...models]'),
    ('mixnet_l', 'MixNet-L', '1907.09595', 224, 0.875, 200, 'pytorch, from [rwightman/pyt...models]'),
)


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net
