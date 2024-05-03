"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file', 'load_model', 'download_model']

import os
import zipfile
import logging
import hashlib
import warnings
import numpy as np
import h5py
from keras import backend as K
from keras.engine.saving import load_attributes_from_hdf5_group

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '1788', 'b00ce627a6c74fe471eb7aebf906fcfa79387861', 'v0.0.394'),
    ('alexnetb', '1853', '045e80b5a055c0006215b1f416c0e4b03455b3e5', 'v0.0.384'),
    ('zfnet', '1715', '3226638b9270c0f3a2ad5302d56b3f9d47706b88', 'v0.0.395'),
    ('zfnetb', '1483', '6ff6768e463fdd333cef5ecb821a456cff6debb8', 'v0.0.400'),
    ('vgg11', '1016', 'c6bc31d0f1f1575081107f4ea8e2ecec3132bfb4', 'v0.0.381'),
    ('vgg13', '0950', 'f0e5bed7cb64111b0fdf73875a33500e8f78a365', 'v0.0.388'),
    ('vgg16', '0832', 'baf4278d9d75dbb76b459e7f30f2d1e18c44ae1b', 'v0.0.401'),
    ('vgg19', '0767', '315c0bc8ddcdfd90b503a6fa197a596c1b23b897', 'v0.0.420'),
    ('bn_vgg11', '0934', '96a967baaf97ebf8c5802f60aabbc30fc59d9027', 'v0.0.339'),
    ('bn_vgg13', '0887', 'd4a3da4039babf9ec89348ed3a9f11bb2d899c25', 'v0.0.353'),
    ('bn_vgg16', '0757', '2960ba135dda2678514e0ed6d1118adc9ff4e9dc', 'v0.0.359'),
    ('bn_vgg19', '0689', 'aaee8cb7d2c70a4db5dc3c665158b835e00c3fbc', 'v0.0.360'),
    ('bn_vgg11b', '0975', '8a35fd728d1e35570ffaedffd4d8fc8f968f4bfe', 'v0.0.407'),
    ('bn_vgg13b', '1016', 'b26cafd39447f039a8124dda8a177b2dc72d98f3', 'v0.0.123'),
    ('bn_vgg16b', '0865', '2272fdd110106e830920bfdd5999fa58737f20e4', 'v0.0.123'),
    ('bn_vgg19b', '0814', '852e2ca228821f3ea1d32a12ce47a9a001236f5e', 'v0.0.123'),
    ('resnet10', '1385', '0a7d3ca6c6616d0a55ebfb0faabe56af980509f5', 'v0.0.248'),
    ('resnet12', '1303', '3ba378deed1b148fff66a83c2ef195a0acaca563', 'v0.0.253'),
    ('resnet14', '1220', 'b7cfec5936dad4f7c56620c331d5546be57a3ab3', 'v0.0.256'),
    ('resnetbc14b', '1116', 'defe7c1982bf0b60b6043792210631ff36448f30', 'v0.0.309'),
    ('resnet16', '1088', 'cc0968d30689e278ba68cd80c8b76a430c8a24a3', 'v0.0.259'),
    ('resnet18_wd4', '1741', '6d84323b46771d914075db6665ca1f164a0936f7', 'v0.0.262'),
    ('resnet18_wd2', '1283', '8e70ce72e5e9ab6925aac0b4342156d3ba462523', 'v0.0.263'),
    ('resnet18_w3d4', '1066', 'afa3a2391bfb24bc46a19ad0332fc9824418a8a7', 'v0.0.266'),
    ('resnet18', '0952', '0817d05847105afe00b42c99ab4dc8031f196fbe', 'v0.0.153'),
    ('resnet26', '0837', 'b3c764c0a35c1e2bca5d0526a72fc2b007f72f97', 'v0.0.305'),
    ('resnetbc26b', '0759', 'a1916fd0f4ffa2d116c20be61d7f78c52aaea58b', 'v0.0.313'),
    ('resnet34', '0744', 'd366daf86c928d3aa2efd63328fa37e918d6fa32', 'v0.0.291'),
    ('resnetbc38b', '0672', '703a75434656b7892a121993353ac3cefbeb91e4', 'v0.0.328'),
    ('resnet50', '0604', '8e1e86d39b65517592152eccf6e6d2eca7cf2a9b', 'v0.0.329'),
    ('resnet50b', '0610', '8a54fb83791e86cdc190fc9a35cfab30c70d394b', 'v0.0.308'),
    ('resnet101', '0599', 'ab4289478d017d3b929011c26fbcf8b54dd8ce07', 'v0.0.49'),
    ('resnet101b', '0511', '84e8ef696cfec4990365c9b0f6afa641a6330357', 'v0.0.357'),
    ('resnet152', '0535', '43ecb2b0cc2dccd771aea77b674c64a69d449164', 'v0.0.144'),
    ('resnet152b', '0479', 'a0dd484cb7afdb2a813858df487c484f17543683', 'v0.0.378'),
    ('preresnet10', '1401', '2349a7c822eac8821120aff6417de0bba99d7966', 'v0.0.249'),
    ('preresnet12', '1322', '32f2f50c15e6320202fdaf40790e03f4f469a281', 'v0.0.257'),
    ('preresnet14', '1219', 'b123205e636c19708eb4a311808c6894c28794cb', 'v0.0.260'),
    ('preresnetbc14b', '1151', '8989bc9fea0e2acb6b31e1b891078fd9e55559d5', 'v0.0.315'),
    ('preresnet16', '1081', 'ec02b7995fedb58cda1c20c236fa44be0d7b434d', 'v0.0.261'),
    ('preresnet18_wd4', '1778', '13ecb34c8031d30d01c19542b17b96c83fba1f32', 'v0.0.272'),
    ('preresnet18_wd2', '1319', '694dbc5bb6f20657478d56f280cac67673103c23', 'v0.0.273'),
    ('preresnet18_w3d4', '1068', '13000951d0737fb3c180dbf4f8c6c116de1c5086', 'v0.0.274'),
    ('preresnet18', '0952', 'b88bf7670642b313929649a20b2a07e4cbe3b35a', 'v0.0.140'),
    ('preresnet26', '0834', 'be46d91ce5f85b2fadcd77e0a126600221dbd826', 'v0.0.316'),
    ('preresnetbc26b', '0786', 'f6ab507bce438a1cfb033558ec85ec78ff248d99', 'v0.0.325'),
    ('preresnet34', '0751', 'fcccbc33435c60f9257f50c5bb8b2ea0fb626535', 'v0.0.300'),
    ('preresnetbc38b', '0633', 'b6793dec9fa0893cad19ec346ed3651a01d75a87', 'v0.0.348'),
    ('preresnet50', '0620', '91bd3a6071d230d061eda0e6af22eff0a782b47c', 'v0.0.330'),
    ('preresnet50b', '0632', 'd3f20f4ea7dc030bb6be59898a79522525263d05', 'v0.0.307'),
    ('preresnet101', '0575', '5dff088de44ce782ac72b4c5fbc03de83b379d1c', 'v0.0.50'),
    ('preresnet101b', '0540', 'e70bed8e7abd2a50fb7a74464fc1da06e83b8ab1', 'v0.0.351'),
    ('preresnet152', '0531', 'a5ac128d79e3e6eb01a4a5eeb571e252482edbc7', 'v0.0.50'),
    ('preresnet152b', '0500', '360cd64056ab0d0d00de059cc748ad7e54ebf258', 'v0.0.386'),
    ('preresnet200b', '0564', '9172d4c02aef8c6ff1504dcf3c299518325afae0', 'v0.0.50'),
    ('preresnet269b', '0556', 'bdd89388474c482c432d3af5d5c4231b33e68588', 'v0.0.239'),
    ('resnext14_16x4d', '1224', '146ff5dae72156ed07a7a0a679ae419d1ece78b5', 'v0.0.370'),
    ('resnext14_32x2d', '1246', '3af87217c5a811d5f303986c7f9a27955daed304', 'v0.0.371'),
    ('resnext14_32x4d', '1110', '86af26f7ab4c0c7f30dea890cbd66e79444b16cb', 'v0.0.327'),
    ('resnext26_32x2d', '0850', '0e54facd6ad17180075862cf032fed1a30e6e034', 'v0.0.373'),
    ('resnext26_32x4d', '0720', 'a5e34838cc78ff16c4b3e1bf1c00acaa4f205d53', 'v0.0.332'),
    ('resnext50_32x4d', '0546', '1c9906b02b3194c568ccf13a478fc7e81a8edb29', 'v0.0.417'),
    ('resnext101_32x4d', '0492', '24e9dbdb2350ad74c2e054c3260b2db1c860ea05', 'v0.0.417'),
    ('resnext101_64x4d', '0483', 'a6b4bdefff3bac5c435d4b3a2cd46eae298be209', 'v0.0.417'),
    ('seresnet10', '1329', 'f70cf6c73471f3878641414d95dc979a7acbb221', 'v0.0.354'),
    ('seresnet18', '0920', 'bb27e27345afaee959e42eada2e79f91ffa2fb22', 'v0.0.355'),
    ('seresnet26', '0803', 'd689714732e408238ee85de95e06aeb56aed4002', 'v0.0.363'),
    ('seresnetbc26b', '0682', 'ba3e51706b787ba4034bbb74d2b60afeb16cc2e8', 'v0.0.366'),
    ('seresnetbc38b', '0575', '536881363a6bbc41cda686dff4881c9a2acd086c', 'v0.0.374'),
    ('seresnet50', '0643', 'fabfa4062a7724ea31752434a687e1837eb30932', 'v0.0.52'),
    ('seresnet50b', '0533', 'bc9d11ec3038951cac6e03c33c3abd61eb61e9a4', 'v0.0.387'),
    ('seresnet101', '0588', '933d34159345f5cf9a663504f03cd423b527aeac', 'v0.0.52'),
    ('seresnet152', '0577', 'd25ced7d6369f3d14ed2cfe54fb70bc4be9c68e0', 'v0.0.52'),
    ('sepreresnet10', '1306', '6096e4d9873949faf31ffbed58a321d8b2396ba7', 'v0.0.377'),
    ('sepreresnet18', '0938', 'd0bf29b9a7d489a5be3a3a802be7c9bb87a5df5f', 'v0.0.380'),
    ('sepreresnetbc26b', '0636', 'cc11e087d240944f6e5e8952460ba2f417d91950', 'v0.0.399'),
    ('sepreresnetbc38b', '0563', 'f4b96ed792b0f92473c8f43763cf6b6340d19960', 'v0.0.409'),
    ('seresnext50_32x4d', '0505', '077f048f2b4e4fd1946c6c3f85a07b9566dc6271', 'v0.0.418'),
    ('seresnext101_32x4d', '0460', '08ea8055b2b3d8c5c2eafcb200355968649c8f52', 'v0.0.418'),
    ('seresnext101_64x4d', '0466', '28ff2d1f7f77569101515fdfb93298feb936d33a', 'v0.0.418'),
    ('senet16', '0806', '8a634c501ee89777cfd0af9ec8b953e7ebc1a5de', 'v0.0.341'),
    ('senet28', '0591', '33c65063c8889f065cd92cee5abe4fee3a129eec', 'v0.0.356'),
    ('senet154', '0465', '962aeede627d5196eaf0cf8c25b6f7281f62e9ea', 'v0.0.54'),
    ('densenet121', '0684', '7c6d506aa37ffdbab6fbe8ee45f8ef8d9b505fa2', 'v0.0.314'),
    ('densenet161', '0618', '070fcb455db45c45aeb67fa4fb0fda4a89b7ef45', 'v0.0.55'),
    ('densenet169', '0605', '7b3b7888c19a672d914800bdebc701ce6bb9f360', 'v0.0.406'),
    ('densenet201', '0635', 'cf3afbb259163bb76eee519f9d43ddbdf0a583b9', 'v0.0.55'),
    ('darknet_tiny', '1746', '147e949b779914331f740badc82339a2fb5bcb11', 'v0.0.69'),
    ('darknet_ref', '1668', '2ef080bb6f470e5ffb0c625ff3047de97cfeb6e2', 'v0.0.64'),
    ('darknet53', '0556', 'd6c6e7dcb96bd6d6789f35c41ac9abb4474b4bf1', 'v0.0.150'),
    ('squeezenet_v1_0', '1756', 'a489092344c0214c402655210e031a1441bd70d1', 'v0.0.128'),
    ('squeezenet_v1_1', '1739', 'b9a8f9eae7a48d053895fe4a362d1d8eb592e994', 'v0.0.88'),
    ('squeezeresnet_v1_0', '1780', 'fb9a54aac20d59f73111fee0745e144b183a66d9', 'v0.0.178'),
    ('squeezeresnet_v1_1', '1784', '43ee9cbbb91046f5316ee14e227f8323b1801b51', 'v0.0.70'),
    ('sqnxt23_w1', '1862', 'cab60636597912e7861d7b6618ecb390b90545ec', 'v0.0.171'),
    ('sqnxt23v5_w1', '1757', '96b94e1dfa1872f96f9b1ce99546a0613bfb1775', 'v0.0.172'),
    ('sqnxt23_w3d2', '1330', 'e52625a000e7a0b02fdf01c64b18a8a21c10b7cd', 'v0.0.210'),
    ('sqnxt23v5_w3d2', '1284', 'fd150fcca3fb73242650ba5e705cc2947def075e', 'v0.0.212'),
    ('sqnxt23_w2', '1066', 'a34e73b9645874532b42bf4a12765080d4c53fb1', 'v0.0.240'),
    ('sqnxt23v5_w2', '1028', '13c5a59866483b958bb116a60001b31f783022a4', 'v0.0.216'),
    ('shufflenet_g1_wd4', '3676', 'cb39b77366909eb13b736497c6eb239efb69e4ac', 'v0.0.134'),
    ('shufflenet_g3_wd4', '3615', '21150468a44c548845b2304700445485407670c7', 'v0.0.135'),
    ('shufflenet_g1_wd2', '2238', '76709a36a9fb8feb2c9ac50fecfcbccdc2bf77ec', 'v0.0.174'),
    ('shufflenet_g3_wd2', '2060', '173a725c1a8b66be6f5b044f0994634113cff8b0', 'v0.0.167'),
    ('shufflenet_g1_w3d4', '1675', '56aa41794ba19d865c06dba56fd73f676dec1f48', 'v0.0.218'),
    ('shufflenet_g3_w3d4', '1609', '34e28781782082e73a06c7230b6c87caacf58945', 'v0.0.219'),
    ('shufflenet_g1_w1', '1350', 'f44c8a1823606c81f3524038333356fc8f022cd6', 'v0.0.223'),
    ('shufflenet_g2_w1', '1332', '8784a32bb15e2bb49496ee6d151539d4eb085bbb', 'v0.0.241'),
    ('shufflenet_g3_w1', '1329', '0e213e7696a5ae086648152b9e28819798259081', 'v0.0.244'),
    ('shufflenet_g4_w1', '1310', 'ef2ff63e8fad961d1b38ba711e2b2ecadd078508', 'v0.0.245'),
    ('shufflenet_g8_w1', '1320', '796314f132292c36d09baa8486b5e40d974ecc4d', 'v0.0.250'),
    ('shufflenetv2_wd2', '1840', '9b4b0964301ba3f2e393c3d3b9a43de3bb480b05', 'v0.0.90'),
    ('shufflenetv2_w1', '1133', 'bcba973eb9f0c333564ed9761ecfd77d28326e5b', 'v0.0.133'),
    ('shufflenetv2_w3d2', '0927', '17a260398afbc6b27b9ab917d538e36993c12fb9', 'v0.0.288'),
    ('shufflenetv2_w2', '0822', 'a0209f14172e8c7c7c4a8e54307641ed69838beb', 'v0.0.301'),
    ('shufflenetv2b_wd2', '1783', 'ca8409ae44489695b468ceb7104e1cc63cb09873', 'v0.0.211'),
    ('shufflenetv2b_w1', '1101', '1caf1b22107357e3ed7409545eff6e815044bcb7', 'v0.0.211'),
    ('shufflenetv2b_w3d2', '0880', '265c3c7c077dd66f435bcc5f239010fd975f7006', 'v0.0.211'),
    ('shufflenetv2b_w2', '0810', '2149df381bcb370856cb4c7a27130d50a96b61f9', 'v0.0.242'),
    ('menet108_8x1_g3', '2031', 'a4d43433e2d9f770c406b3f780a8960609c0e9b8', 'v0.0.89'),
    ('menet128_8x1_g4', '1914', '5bb8f2287930abb3e921842f053d6592f7034ea7', 'v0.0.103'),
    ('menet160_8x1_g8', '2028', '09664de97e30e93189cf6d535c3a297b9c8c190e', 'v0.0.154'),
    ('menet228_12x1_g3', '1288', 'c2eeac242640ba862e04d9f7b67bcfe608b1c269', 'v0.0.131'),
    ('menet256_12x1_g4', '1217', 'b020cc33586896c2c8501c84e72a38818778c796', 'v0.0.152'),
    ('menet348_12x1_g3', '0936', '6795f0079484c1c4b4f65df1df5e68302861340a', 'v0.0.173'),
    ('menet352_12x1_g8', '1167', 'a9d9412dcebfaf7682c4c7cb8c7a1232f04bcce6', 'v0.0.198'),
    ('menet456_24x1_g3', '0780', '6645f5946ddda7039ac5fb4cfcac8a4e1338df52', 'v0.0.237'),
    ('mobilenet_wd4', '2217', 'fb7abda85e29c592f0196ff4a76b9ee2951c6e3c', 'v0.0.62'),
    ('mobilenet_wd2', '1330', 'aa86f3554b83e1a818b197e07cbc16585e1d15a3', 'v0.0.156'),
    ('mobilenet_w3d4', '1051', 'd200ad45590faa190c194ae9ca6853c19af97b63', 'v0.0.130'),
    ('mobilenet_w1', '0866', '9661b555d739c4bb2c519c598a96a1b3d288b006', 'v0.0.155'),
    ('fdmobilenet_wd4', '3052', '6c219205677d97f8c07479c7fdfe51990d608f84', 'v0.0.177'),
    ('fdmobilenet_wd2', '1969', '5678a212ba44317306e2960ddeed6a5c0489122f', 'v0.0.83'),
    ('fdmobilenet_w3d4', '1601', '2ea5eba9e1b8caf9235b71835971f868a9b0d1de', 'v0.0.159'),
    ('fdmobilenet_w1', '1312', 'e11d0dce083322e06e5ca296d2dfa5dff742d74a', 'v0.0.162'),
    ('mobilenetv2_wd4', '2412', '622733723bdd6b9df10723f16a465586be1c3d4b', 'v0.0.137'),
    ('mobilenetv2_wd2', '1443', 'c7086bcc628b74e2ed942631e1ed2d3fa8b2657b', 'v0.0.170'),
    ('mobilenetv2_w3d4', '1044', '29e9923c74c059abac6c5194c04570837510974a', 'v0.0.230'),
    ('mobilenetv2_w1', '0864', '5e487e824d18fc8f776b3103bab677ed1a81b6ab', 'v0.0.213'),
    ('mobilenetv3_large_w1', '0769', 'fc909b4c0fa19a789806c254977c652ad782184b', 'v0.0.411'),
    ('igcv3_wd4', '2829', '00072cafe96ba57f84a689d3016b85224b234983', 'v0.0.142'),
    ('igcv3_wd2', '1704', 'b8961ca335abd1d66eb2cf180eb14381ebdcc3ee', 'v0.0.132'),
    ('igcv3_w3d4', '1097', 'fb365b725beaf38429a98a52b88a36d3e423329b', 'v0.0.207'),
    ('igcv3_w1', '0899', '968237cbd0a55b43f8847b919fa3ba02a27bb595', 'v0.0.243'),
    ('mnasnet_b1', '0800', '9ce379b36ee4738719c82e44ef1917e0a846fbb8', 'v0.0.419'),
    ('mnasnet_a1', '0755', '8bf70a05b4d97ed149324c74fd400a3273a9478d', 'v0.0.419'),
    ('efficientnet_b0', '0722', '2bea741f87b9e0d85570bb3753597a11654f2f78', 'v0.0.364'),
    ('efficientnet_b1', '0626', 'd7a4bf8be529396c2375c93f50f355ee7968ab3f', 'v0.0.376'),
    ('efficientnet_b0b', '0669', '436cc024344cb2f4160bfbc4b5fb6c23d9f96987', 'v0.0.403'),
    ('efficientnet_b1b', '0564', 'f2eb3cd8d915f9eacc90cccf10ee02c5be7475b4', 'v0.0.403'),
    ('efficientnet_b2b', '0516', '9c08b8392236ca0654c195bb0020b412d23340b7', 'v0.0.403'),
    ('efficientnet_b3b', '0431', 'd1545ea07602b3e661c10b3ee246819ea80cf0b7', 'v0.0.403'),
    ('efficientnet_b4b', '0376', 'c7e29f57ea62639cebe936bff55335caca23bf42', 'v0.0.403'),
    ('efficientnet_b5b', '0334', '4365cf122b0a5b514347220898118b43e4d0e271', 'v0.0.403'),
    ('efficientnet_b6b', '0312', '7f3f3465e9c2538c36c7d97997a5fea5f1883719', 'v0.0.403'),
    ('efficientnet_b7b', '0311', 'b4aac2ceee6e22d67b5e61c46be3e34a9aedd06f', 'v0.0.403')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".keras", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $KERAS_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.h5".format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning("Mismatch in the content of model file detected. Downloading again.")
    else:
        logging.info("Model file not found. Downloading to {}.".format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + ".zip"
    _download(
        url="{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip".format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError("Downloaded file has different hash. Please try again.")


def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn("Unverified HTTPS request is being made (verify_ssl=False). Adding certificate verification"
                      " is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning("File {} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the `repo_url` is overridden, consider switching to "
                                      "the default repo.".format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, "s" if retries > 1 else ""))

    return fname


def _check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def _preprocess_weights_for_loading(layer,
                                    weights):
    """
    Converts layers weights.

    Parameters
    ----------
    layer : Layer
        Layer instance.
    weights : list of np.array
        List of weights values.

    Returns
    -------
    list of np.array
        A list of weights values.
    """
    is_channels_first = (K.image_data_format() == "channels_first")
    if ((K.backend() == "mxnet") and (not is_channels_first)) or (K.backend() == "tensorflow"):
        if layer.__class__.__name__ == "Conv2D":
            weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
        elif layer.__class__.__name__ == "DepthwiseConv2D":
            weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
    for i in range(len(weights)):
        assert (K.int_shape(layer.weights[i]) == weights[i].shape)
    return weights


def _load_weights_from_hdf5_group(f,
                                  layers):
    """
    Implements topological (order-based) weight loading.

    Parameters
    ----------
    f : File
        A pointer to a HDF5 group.
    layers : list of np.array
        List of target layers.
    """
    filtered_layers = []
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = load_attributes_from_hdf5_group(f, "layer_names")
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, "weight_names")
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError("You are trying to load a weight file "
                         "containing " + str(len(layer_names)) +
                         " layers into a model with " +
                         str(len(filtered_layers)) + " layers.")

    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, "weight_names")
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = _preprocess_weights_for_loading(
            layer=layer,
            weights=weight_values)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError("Layer #" + str(k) +
                             " (named `" + layer.name +
                             "` in the current model) was found to "
                             "correspond to layer " + name +
                             " in the save file. "
                             "However the new layer " + layer.name +
                             " expects " + str(len(symbolic_weights)) +
                             " weights, but the saved weights have " +
                             str(len(weight_values)) +
                             " elements.")
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)


def _load_weights_from_hdf5_group_by_name(f,
                                          layers):
    """
    Implements name-based weight loading.

    Parameters
    ----------
    f : File
        A pointer to a HDF5 group.
    layers : list of np.array
        List of target layers.
    """
    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, "layer_names")

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, "weight_names")
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            weight_values = _preprocess_weights_for_loading(
                layer=layer,
                weights=weight_values)
            if len(weight_values) != len(symbolic_weights):
                warnings.warn("Skipping loading of weights for layer {} due to mismatch in number of weights ({} vs"
                              " {}).".format(layer, len(symbolic_weights), len(weight_values)))
                continue
            # Set values.
            for i in range(len(weight_values)):
                symbolic_shape = K.int_shape(symbolic_weights[i])
                if symbolic_shape != weight_values[i].shape:
                    warnings.warn("Skipping loading of weights for layer {} due to mismatch in shape ({} vs"
                                  " {}).".format(layer, symbolic_weights[i].shape, weight_values[i].shape))
                    continue
                else:
                    weight_value_tuples.append((symbolic_weights[i],
                                                weight_values[i]))
    K.batch_set_value(weight_value_tuples)


def load_model(net,
               file_path,
               skip_mismatch=False):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    net : Model
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    skip_mismatch : bool, default False
        Whether to skip loading of layers with wrong names.
    """
    # if (K.backend() == "mxnet") and (K.image_data_format() == "channels_first"):
    #     net.load_weights(filepath=file_path, by_name=skip_mismatch)
    #     return
    with h5py.File(file_path, mode='r') as f:
        if ("layer_names" not in f.attrs) and ("model_weights" in f):
            f = f["model_weights"]
        if ("keras_version" not in f.attrs) or ("backend" not in f.attrs):
            raise ImportError("Unsupported version of Keras checkpoint file.")
        # original_keras_version = f.attrs["keras_version"].decode("utf8")
        original_backend = f.attrs["backend"].decode("utf8")
        assert (original_backend == "mxnet")
        if skip_mismatch:
            _load_weights_from_hdf5_group_by_name(
                f=f,
                layers=net.layers)
        else:
            _load_weights_from_hdf5_group(
                f=f,
                layers=net.layers)


def download_model(net,
                   model_name,
                   local_model_store_dir_path=os.path.join("~", ".keras", "models")):
    """
    Load model state dictionary from a file with downloading it if necessary.

    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    """
    load_model(
        net=net,
        file_path=get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path))
