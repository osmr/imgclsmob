"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file', 'load_state_dict', 'download_state_dict', 'init_variables_from_state_dict']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '1788', 'd3cd2a5a7dfb882c47153b5abffc2edfe8335838', 'v0.0.394'),
    ('alexnetb', '1853', '58a51cd1803929c52eed6e1c69a43328fcc1d1cb', 'v0.0.384'),
    ('zfnet', '1715', 'a18747efbec5ce849244a540651228e287d13296', 'v0.0.395'),
    ('zfnetb', '1482', '2624da317b57bec7d3ce73e7548caa6eca0a6ad6', 'v0.0.400'),
    ('vgg11', '1015', 'b87e9dbcbab308f671a69ef3ed67067e62c5429f', 'v0.0.381'),
    ('vgg13', '0946', 'f1411e1fdd5e75919d4e1a0f7b33ac06e0acb146', 'v0.0.388'),
    ('vgg16', '0830', 'e63ead2e896e1a5185840b8cc0973d0791b19d35', 'v0.0.401'),
    ('vgg19', '0768', 'cf2a33c6221e44432f38dd8bdcf1312efd208ae8', 'v0.0.420'),
    ('bn_vgg11', '0936', '4ff8667b2daba34bb4531744a50c000543e4e524', 'v0.0.339'),
    ('bn_vgg13', '0888', '0a49f8714fa647684940feb6cffb0a468290a5af', 'v0.0.353'),
    ('bn_vgg16', '0755', '9948c82dcb133081796d095ebd5cf8815d77f7d1', 'v0.0.359'),
    ('bn_vgg19', '0689', '8a3197c6a27aa4a271e31610ff6cc58c6b593a81', 'v0.0.360'),
    ('bn_vgg11b', '0979', '6a3890a42d0b7bab962582298ddaf6077eedf22d', 'v0.0.407'),
    ('bn_vgg13b', '1015', '999e47a6a5d4cb493d1af3e31de04d67d25176e8', 'v0.0.123'),
    ('bn_vgg16b', '0866', '1f8251aa987151e89a82f0f209c2a8bbde0f0c47', 'v0.0.123'),
    ('bn_vgg19b', '0817', '784e4c396e6de685727bc8fd30f5ed35c66a84a0', 'v0.0.123'),
    ('resnet10', '1390', '7fff13aee28ba7601c155907be64aff344530736', 'v0.0.248'),
    ('resnet12', '1300', '9539494f8fae66c454efed4e5abf26c273d3b285', 'v0.0.253'),
    ('resnet14', '1225', 'd1fb0f762258c9fd04a3ad469462f732333740fa', 'v0.0.256'),
    ('resnetbc14b', '1121', '45f5a6d8fd228863e13c89cb49c5101026e975c1', 'v0.0.309'),
    ('resnet16', '1086', '5ac8e7da9dcee268db9b7cf4ecfeceb8efca3005', 'v0.0.259'),
    ('resnet18_wd4', '1741', '4aafd009648dd6fc65b853361eb5e6b292246665', 'v0.0.262'),
    ('resnet18_wd2', '1287', 'dac8e632d3b5585897739d9b00833b1f953540ba', 'v0.0.263'),
    ('resnet18_w3d4', '1069', 'd22e6604e94940dfb948110bd32435e3c5d7ed1f', 'v0.0.266'),
    ('resnet18', '0956', 'b4fc7198d9bbcf6699b904824c839943871401bc', 'v0.0.153'),
    ('resnet26', '0838', 'f647811d7211d82344419b1590fb3ae73433efa7', 'v0.0.305'),
    ('resnetbc26b', '0757', '55c88013263af95b0d391069e34542a5d899cc7d', 'v0.0.313'),
    ('resnet34', '0742', '8faa0ab2cbb8ff4ad3bb62aa82da3dd1eb3ef05d', 'v0.0.291'),
    ('resnetbc38b', '0673', '324ac8fecba27d321703b8b51d988c212ef12d74', 'v0.0.328'),
    ('resnet50', '0605', '34177a2e963820ae5ee9c7b2bd233a2566928774', 'v0.0.329'),
    ('resnet50b', '0609', '4b68417369140303594ae69d5ac5891e9fe91267', 'v0.0.308'),
    ('resnet101', '0601', '3fc260bc67ab133b39f087862f5bc70cf6aa9442', 'v0.0.72'),
    ('resnet101b', '0507', '527dca370eb8a2a4a25025993f8ccce35b00c9ef', 'v0.0.357'),
    ('resnet152', '0535', 'b21844fcaea4e14a91fa17bfa870a3d056d258ea', 'v0.0.144'),
    ('resnet152b', '0485', '36964f4867125dd08fa722d4d639273d7d1874e1', 'v0.0.378'),
    ('preresnet10', '1401', '3a2eed3b9254d35ba546c9894cf9cc3c6d88aa5c', 'v0.0.249'),
    ('preresnet12', '1321', '0c424c407bd91c5135ec74660b5f001f07cec0df', 'v0.0.257'),
    ('preresnet14', '1216', 'fda0747fd40cad58e46dad53e68d3d06b8829829', 'v0.0.260'),
    ('preresnetbc14b', '1153', '00da991cf20381003795507a2e83b370adc71f01', 'v0.0.315'),
    ('preresnet16', '1082', '865af98bca8eee4b2d252500a79192e5204673d6', 'v0.0.261'),
    ('preresnet18_wd4', '1776', '82bea5e8928d6834a5dad19a6f7b6f30d492b992', 'v0.0.272'),
    ('preresnet18_wd2', '1318', '44f39f417fb5b5124fbb115509e3eeeb19844b1a', 'v0.0.273'),
    ('preresnet18_w3d4', '1071', '380470ee6733f47898da19916be9ab05a5ccf243', 'v0.0.274'),
    ('preresnet18', '0949', '692e6c11e738c11eaf818d60a214e7a905a873c1', 'v0.0.140'),
    ('preresnet26', '0833', '8de37e08f3c2dd054a1dc4099d4b398097999af6', 'v0.0.316'),
    ('preresnetbc26b', '0789', '993dd84a36d8f1417e2f5454ec5f3b3159f251c1', 'v0.0.325'),
    ('preresnet34', '0754', '9d5635846928420d41f7304e02a4d33160af45e7', 'v0.0.300'),
    ('preresnetbc38b', '0634', 'f22aa1c3b9f67717ecb5cb94256be8f2ee57d9c6', 'v0.0.348'),
    ('preresnet50', '0625', '06130b124a1abf96cc92f7d212ca9b524da02ddd', 'v0.0.330'),
    ('preresnet50b', '0631', '9fc00073139d763ef08e2fc810c2469c9b0182c9', 'v0.0.307'),
    ('preresnet101', '0572', 'cd61594e9e2fb758ca69a38baf31223351638c4f', 'v0.0.73'),
    ('preresnet101b', '0539', 'c0b9e129908051592393ba5e7939a4feb5b82b6c', 'v0.0.351'),
    ('preresnet152', '0529', 'b761f286ab284b916f388cc5d6af00e5ea049081', 'v0.0.73'),
    ('preresnet152b', '0500', '7ae9df4bbabbc12d32a35f4369d64269ba3c8e7b', 'v0.0.386'),
    ('preresnet200b', '0560', '881e0e2869428d89831bde0c7da219ed69236f16', 'v0.0.73'),
    ('preresnet269b', '0555', 'c799eaf246d3dccf72ac10cdec3f35bd8bf72e71', 'v0.0.239'),
    ('resnext14_16x4d', '1224', '3f603dde73c4581f60ada40499ed42d800847268', 'v0.0.370'),
    ('resnext14_32x2d', '1246', 'df7d6b8a824796742a0bb369d654135cd109dfb3', 'v0.0.371'),
    ('resnext14_32x4d', '1113', 'cac0dad52d391f9268c9fee6f95be59e14952fcc', 'v0.0.327'),
    ('resnext26_32x2d', '0849', '2dee5d79b8f093f1f6d1cf87b7f36e0481c6648f', 'v0.0.373'),
    ('resnext26_32x4d', '0717', '594567d27cea1f5e324a6ecfd93f209d30c148d9', 'v0.0.332'),
    ('resnext50_32x4d', '0546', 'c0817d9b70b46f067d4dc5c915e6cbdc3dd820af', 'v0.0.417'),
    ('resnext101_32x4d', '0493', 'de52ea63f204c839c176f6162ae73a19a33626c4', 'v0.0.417'),
    ('resnext101_64x4d', '0485', 'ddff97a9e6aa2ccd603a067c2158044cec8b8342', 'v0.0.417'),
    ('seresnet10', '1336', 'd4a0a9d3e2e2b4188aac06d3b6cc4132f05ac916', 'v0.0.354'),
    ('seresnet18', '0923', '7aa519d2ec4c721c61a9fd04a9b0ca745f12b24a', 'v0.0.355'),
    ('seresnet26', '0809', 'b2a8b74fe11edbfa798c35d882d6ebd5cfceb6ff', 'v0.0.363'),
    ('seresnetbc26b', '0681', '692ccde37b4dc19df0bc5b92256f0023228baf98', 'v0.0.366'),
    ('seresnetbc38b', '0578', '2d787dc45bd6775fa96b4048ab5ac191089a0ab0', 'v0.0.374'),
    ('seresnet50', '0643', 'e022e5b9e58e19c692d00394c85daa57ea943b82', 'v0.0.75'),
    ('seresnet50b', '0533', '539e58be15125cf7693cc6318d99592a2f956d48', 'v0.0.387'),
    ('seresnet101', '0589', '305d23018de942b25df59d8ac9d2dd14374d7d28', 'v0.0.75'),
    ('seresnet152', '0578', 'd06ab6d909129693da68c552b91f3f344795114f', 'v0.0.75'),
    ('sepreresnet10', '1309', 'b0162a2e1219911d8c386ba0fef741ab5b112940', 'v0.0.377'),
    ('sepreresnet18', '0941', '5606cb354b61974a97ae81807194638fc3ea0576', 'v0.0.380'),
    ('sepreresnetbc26b', '0634', 'd903397d7afafbf43b5c19da927601a128cafd0b', 'v0.0.399'),
    ('sepreresnetbc38b', '0564', '262a4a2e23d34ab244b107fe8397e776744e4fcb', 'v0.0.409'),
    ('seresnext50_32x4d', '0507', '982a4cb8190a4e7bb21d4582336c13d8363c4ece', 'v0.0.418'),
    ('seresnext101_32x4d', '0461', 'b84ec20adb9d67f56ac4cd6eb35b134f964c1936', 'v0.0.418'),
    ('seresnext101_64x4d', '0465', 'b16029e686fb50fd64ed59df22c9d3c5ed0470c1', 'v0.0.418'),
    ('senet16', '0803', '366c58ce2f47ded548734cf336d46b50517c78c4', 'v0.0.341'),
    ('senet28', '0594', '98ba8cc2068495fe192af40328f1838b1e835b6f', 'v0.0.356'),
    ('senet154', '0463', 'c86eaaed79c696a32ace4a8576fc0b50f0f93900', 'v0.0.86'),
    ('densenet121', '0688', 'e3bccdc5544f46352bb91671ac4cd7e2f788952b', 'v0.0.314'),
    ('densenet161', '0617', '9deca33a34a5c4a0a84f0a37920dbfd1cad85cb7', 'v0.0.77'),
    ('densenet169', '0606', 'fcbb5c869350e22cc79b15a0508f2f5598dacb90', 'v0.0.406'),
    ('densenet201', '0635', '5eda789595ba0b8b450705220704687fa8ea8788', 'v0.0.77'),
    ('darknet_tiny', '1751', '750ff8d9b17beb5ab88200aa787dfcb5b6ca8b36', 'v0.0.71'),
    ('darknet_ref', '1672', '3c8ed62a43b9e8934b4beb7c47ce4c7b2cdb7a64', 'v0.0.71'),
    ('darknet53', '0555', '49816dbf617b2cd14051c2d7cd0325ee3ebb63a2', 'v0.0.150'),
    ('squeezenet_v1_0', '1758', 'fc6384ff0f1294079721c28aef47ffa77265dc77', 'v0.0.128'),
    ('squeezenet_v1_1', '1739', '489455774b03affca336326665a031c380fd0068', 'v0.0.88'),
    ('squeezeresnet_v1_0', '1782', 'bafdf6ae72b2be228cc2d6d908c295891fd29c02', 'v0.0.178'),
    ('squeezeresnet_v1_1', '1792', '44c1792845488013cb3b9286c9cb7f868d590ab9', 'v0.0.79'),
    ('sqnxt23_w1', '2108', '6267020032ac7d6aa0905b916954864cdfea4934', 'v0.0.171'),
    ('sqnxt23v5_w1', '2077', 'ebc0c53dc0c39e72eb620b06c2eb07ba451fb28d', 'v0.0.172'),
    ('sqnxt23_w3d2', '1509', '8fbdcd6dde6a3fb2f8e8aab4d1eb828123becfb5', 'v0.0.210'),
    ('sqnxt23v5_w3d2', '1539', 'ae14d7b8685b23fcffeba96038e31255a7c718fa', 'v0.0.212'),
    ('sqnxt23_w2', '1235', 'ea1ae9b747fb40f670b32fad28844fdc2af5ea66', 'v0.0.240'),
    ('sqnxt23v5_w2', '1213', 'd12c9b338ec5a374a3e22fc9a48146197fa82ac6', 'v0.0.216'),
    ('shufflenet_g1_wd4', '3680', '3d9856357041fb69f4a6ddf0208e7821605487a9', 'v0.0.134'),
    ('shufflenet_g3_wd4', '3617', '8f00e642cfc2b7ab8b1a770513bb46190c3bcb7d', 'v0.0.135'),
    ('shufflenet_g1_wd2', '2231', 'd5356e3b04c4a30d568755807e996821098d8aae', 'v0.0.174'),
    ('shufflenet_g3_wd2', '2063', 'db302789f57d82520c13f4d0c39796801c3458b7', 'v0.0.167'),
    ('shufflenet_g1_w3d4', '1678', 'ca175843c5d78bf7d6c826142df810b1b721978b', 'v0.0.218'),
    ('shufflenet_g3_w3d4', '1613', 'f7a106be40b1cdcc68e1cf185451832aec3584fc', 'v0.0.219'),
    ('shufflenet_g1_w1', '1351', '2f36fdbc45ef00b49dd558b3b2e5b238be2e28ca', 'v0.0.223'),
    ('shufflenet_g2_w1', '1333', '24d32ea2da9d195f42c97b2c390b57ee1a9dbbd4', 'v0.0.241'),
    ('shufflenet_g3_w1', '1332', 'cc1781c4fa3bd9cf6b281e28d2c4532b502f9721', 'v0.0.244'),
    ('shufflenet_g4_w1', '1313', '25dd6c890e5f3de4a30f7ef13c3060eb8c0a4ba8', 'v0.0.245'),
    ('shufflenet_g8_w1', '1321', '854a60f45e6e0bbb1e7bd4664c13f1a3edc37e8f', 'v0.0.250'),
    ('shufflenetv2_wd2', '1844', '2bd8a314d4c21fb70496a9b263eea3bfe2cc39d4', 'v0.0.90'),
    ('shufflenetv2_w1', '1131', '6a728e21f405d52b0deade6878f4661089b47a51', 'v0.0.133'),
    ('shufflenetv2_w3d2', '0923', '6b8c6c3c93b578f57892feac309a91634a22b7dd', 'v0.0.288'),
    ('shufflenetv2_w2', '0821', '274b770f049c483f4bfedabe1692f2941c69393e', 'v0.0.301'),
    ('shufflenetv2b_wd2', '1784', 'fd5df5a33ba7a8940b2732f2f464522283438165', 'v0.0.158'),
    ('shufflenetv2b_w1', '1104', '6df32bad4c38e603dd75c89ba39c25d45162ab43', 'v0.0.161'),
    ('shufflenetv2b_w3d2', '0880', '9ce6d2b779f0f2483ffc8c8396a9c22af0ea712b', 'v0.0.203'),
    ('shufflenetv2b_w2', '0810', '164690eda8bf24de2f2835250646b8164b9de1dc', 'v0.0.242'),
    ('menet108_8x1_g3', '2032', '4e9e89e10f7bc055c83bbbb0e9f283f983546288', 'v0.0.89'),
    ('menet128_8x1_g4', '1915', '148105f444f44137b3df2d50ef63d811a9d1da82', 'v0.0.103'),
    ('menet160_8x1_g8', '2028', '7ff635d185d0228f147dc32c225da85c99763e9b', 'v0.0.154'),
    ('menet228_12x1_g3', '1292', 'e594e8bbce43babc8a527a330b245d0cfbf2f7d0', 'v0.0.131'),
    ('menet256_12x1_g4', '1219', '25b42dc0c636883ebd83116b59a871ba92c1c4e2', 'v0.0.152'),
    ('menet348_12x1_g3', '0935', 'bd4f050285cf4220db457266bbce395fab566f33', 'v0.0.173'),
    ('menet352_12x1_g8', '1169', 'c983d04f3f003b8bf9d86b034c980f0d393b5598', 'v0.0.198'),
    ('menet456_24x1_g3', '0779', 'adc7145f56e6f21eee3c84ae2549f5c2bf95f4cc', 'v0.0.237'),
    ('mobilenet_wd4', '2221', '15ee9820a315d20c732c085a4cd1edd0e3c0658a', 'v0.0.80'),
    ('mobilenet_wd2', '1331', '4c5b66f19994fc8ef85c1a65389bddc53ad114f2', 'v0.0.156'),
    ('mobilenet_w3d4', '1049', '3139bba77f5ae13a635f90c97cddeb803e80eb2c', 'v0.0.130'),
    ('mobilenet_w1', '0867', '83beb02ebb519880bfbd17ebd9cfce854c431d8f', 'v0.0.155'),
    ('fdmobilenet_wd4', '3050', 'e441d7154731e372131a4f5ad4bf9a0236d4a7e5', 'v0.0.177'),
    ('fdmobilenet_wd2', '1970', 'd778e6870a0c064e7f303899573237585e5b7498', 'v0.0.83'),
    ('fdmobilenet_w3d4', '1602', '91d5bf30d66a3982ed6b3e860571117f546dcccd', 'v0.0.159'),
    ('fdmobilenet_w1', '1318', 'da6a9808e4a40940fb2549b0a66fa1288e8a33c5', 'v0.0.162'),
    ('mobilenetv2_wd4', '2416', 'ae7e5137b9b9c01b35f16380afe7e1423541475e', 'v0.0.137'),
    ('mobilenetv2_wd2', '1446', '696501bd3e6df77a78e85756403a3da23839244b', 'v0.0.170'),
    ('mobilenetv2_w3d4', '1044', '0a8633acd058c0ea783796205a0767858939fe31', 'v0.0.230'),
    ('mobilenetv2_w1', '0862', '03daae54f799467152612138da07a8c221666d70', 'v0.0.213'),
    ('igcv3_wd4', '2835', 'b41fb3c75e090cc719962e1ca2debcbac241dc22', 'v0.0.142'),
    ('igcv3_wd2', '1705', 'de0b98d950a3892b6d15d1c3ea248d41a34adf00', 'v0.0.132'),
    ('igcv3_w3d4', '1096', 'b8650159ab15b118c0655002d9ce613b3a36dea1', 'v0.0.207'),
    ('igcv3_w1', '0903', 'a69c216fa5838dba316b01d347846812835650fe', 'v0.0.243'),
    ('mnasnet_b1', '0800', 'a21e7b11537a81d57be61b27761efa69b0b44728', 'v0.0.419'),
    ('mnasnet_a1', '0756', '2903749fb1ac67254487ccf1668cae064170ffd1', 'v0.0.419')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".tensorflow", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TENSORFLOW_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.tf.npz".format(
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
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

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


def load_state_dict(file_path):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    state_dict : dict
        Dictionary with values of model variables.
    """
    import numpy as np
    assert os.path.exists(file_path) and os.path.isfile(file_path)
    if file_path.endswith(".npy"):
        state_dict = np.load(file_path, encoding="latin1").item()
    elif file_path.endswith(".npz"):
        state_dict = dict(np.load(file_path))
    else:
        raise NotImplementedError
    return state_dict


def download_state_dict(model_name,
                        local_model_store_dir_path=os.path.join("~", ".tensorflow", "models")):
    """
    Load model state dictionary from a file with downloading it if necessary.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TENSORFLOW_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    state_dict : dict
        Dictionary with values of model variables.
    file_path : str
        Path to the file.
    """
    file_path = get_model_file(
        model_name=model_name,
        local_model_store_dir_path=local_model_store_dir_path)
    state_dict = load_state_dict(file_path=file_path)
    return state_dict, file_path


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
    import tensorflow as tf
    assert sess is not None
    if state_dict is None:
        raise Exception("The state dict is empty")
    dst_params = {v.name: v for v in tf.global_variables()}
    sess.run(tf.global_variables_initializer())
    for src_key in state_dict.keys():
        if src_key in dst_params.keys():
            assert (state_dict[src_key].shape == tuple(dst_params[src_key].get_shape().as_list()))
            sess.run(dst_params[src_key].assign(state_dict[src_key]))
        elif not ignore_extra:
            raise Exception("The state dict is incompatible with the model")
        else:
            print("Key `{}` is ignored".format(src_key))
