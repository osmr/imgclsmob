"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
from mxnet.gluon.utils import download, check_sha1

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('resnet10', '1555', 'cfb0a76d89d916adf3e167fe3002f18096f73b4e', 'v0.0.1'),
    ('resnet12', '1446', '9ce715b091d167dd5985d4474c0013b7fb358b08', 'v0.0.30'),
    ('resnet14', '1241', 'a8955ff3d4facd8e38e776fc72f1945806def16a', 'v0.0.40'),
    ('resnet16', '1110', '1be996d1d18da94d9e8dc0cd35d5135ef4da6bc9', 'v0.0.41'),
    ('resnet18_wd4', '2445', '28d15cf486f1e159d9f257ae8f6951f135fa6ccd', 'v0.0.47'),
    ('resnet18_wd2', '1496', 'd839c509bcb8c9c156478f2b91f23e96fc17df15', 'v0.0.46'),
    ('resnet18_w3d4', '1254', 'd654861258936f678b928a92d88fd205a20777af', 'v0.0.18'),
    ('resnet18', '0994', 'ae25f2b27603955a3888f15286692a45d7396045', 'v0.0.1'),
    ('resnet34', '0792', '5b875f4934da8d83d44afc30d8e91362d3103115', 'v0.0.1'),
    ('resnet50', '0687', '79fae958a0acd7a66d688f6453b2bbbc5fe8b3d3', 'v0.0.1'),
    ('resnet50b', '0644', '27a36c02aed870c0c455774f7fb853223f83abc8', 'v0.0.1'),
    ('resnet101', '0599', 'a6d3a5f4933794d56b61867c050ee730f6310f1b', 'v0.0.1'),
    ('resnet101b', '0560', '6517274e7aacd6b05b50da78cb1bf6b9ef85ab57', 'v0.0.1'),
    ('resnet152', '0561', 'd05971c8f10d991ffdbf10318e58f27c2d3471ef', 'v0.0.1'),
    ('resnet152b', '0537', '4f5bd8799404acd1e9e9c857c83877bdb43e299c', 'v0.0.1'),
    ('preresnet18', '0988', '5defff0e1820f8ec7266129afc70e9f686910a7f', 'v0.0.39'),
    ('preresnet34', '0811', 'f8fe98a25337d747b8687ffdbd1c83ce0d9b9a34', 'v0.0.2'),
    ('preresnet50', '0668', '4940c94b02cf25d015a76a0b09498433729c37b8', 'v0.0.2'),
    ('preresnet50b', '0664', '2fcfddb13fbb8d7f58fb949f137610bf3d99a892', 'v0.0.2'),
    ('preresnet101', '0575', 'e2887e539f2519c36aea0fc991d6503ed384c4fc', 'v0.0.2'),
    ('preresnet101b', '0588', '1015145a6228aa16583a975b9c33f879ee2a6fc0', 'v0.0.2'),
    ('preresnet152', '0532', '31505f719ad76f5aee59d37a695ac7a9b06230fc', 'v0.0.14'),
    ('preresnet152b', '0575', 'dc303191ea47ca258f5abadd203b5de24d059d1a', 'v0.0.2'),
    ('preresnet200b', '0564', '38f849a61f59924d85a9353923424889a77c93dc', 'v0.0.45'),
    ('resnext101_32x4d', '0579', '9afbfdbc5a420a9f56058be0bf80d12b21a627af', 'v0.0.10'),
    ('resnext101_64x4d', '0541', '0d4fd87b8de78c5c0295e1dcb9923a578dce7adb', 'v0.0.10'),
    ('seresnet50', '0644', '10954a846a56a387a6a222e260d95fb8a9bd68c3', 'v0.0.11'),
    ('seresnet101', '0589', '4c10238dd485a540a464bf1c39a8752d2da040b9', 'v0.0.11'),
    ('seresnet152', '0577', 'de6f099dd39f374390639ca8854b2954af3c59b9', 'v0.0.11'),
    ('seresnext50_32x4d', '0558', 'a49f8fb039973979afe2fc70974a8b07c7159bca', 'v0.0.12'),
    ('seresnext101_32x4d', '0500', 'cf1612601f319a0e75190ae756ae380b947dcb1a', 'v0.0.12'),
    ('senet154', '0465', 'dd2445078c0770c4a52cd22aa1d4077eb26f6132', 'v0.0.13'),
    ('densenet121', '0780', '49b72d04bace00bb1964b38cec13d19059a14e86', 'v0.0.3'),
    ('densenet161', '0618', '52e30516e566bdef53dcb417f86849530c83d0d1', 'v0.0.3'),
    ('densenet169', '0689', '281ec06b02f407b4523245622371da669a287044', 'v0.0.3'),
    ('densenet201', '0636', '65b5d389b1f2a18c62dc39f74960266c601fec76', 'v0.0.3'),
    ('condensenet74_c4_g4', '0864', 'cde68fa2fcc9197e336717a17753a15a6efd7596', 'v0.0.4'),
    ('condensenet74_c8_g8', '1049', '4cf4a08e7fb46f5821049dcae97ae442b0ceb546', 'v0.0.4'),
    ('dpn68', '0700', '3114719dccf3d9fa30bb7ab5a8c845815328e495', 'v0.0.17'),
    ('dpn98', '0528', 'fa5d6fca985afde21f6374e4a4d4df788d1b4c3a', 'v0.0.17'),
    ('dpn131', '0522', '35ac2f82e69264e0712dcb979da4d99675e2f2aa', 'v0.0.17'),
    ('darknet_tiny', '1746', '16501793621fbcb137f2dfb901760c1f621fa5ec', 'v0.0.69'),
    ('darknet_ref', '1668', '3011b4e14b629f80da54ab57bef305d588f748ab', 'v0.0.64'),
    ('squeezenet_v1_0', '1896', 'b69a46072eba69b5a1967b721a01d2e8c385e0ff', 'v0.0.19'),
    ('squeezenet_v1_1', '1739', 'd7a1483aaa1053c7cd0cf08529b2b87ed2781b35', 'v0.0.88'),
    ('squeezeresnet_v1_1', '1784', '26064b82773e7a7175d6038976a73abfcd5ed2be', 'v0.0.70'),
    ('shufflenetv2_wd2', '1830', '156953de22d0e749c987da4a58e0e53a5fb18291', 'v0.0.90'),
    ('shufflenetv2_w1', '1322', '04b52239c888e06079b327e932ca156f99b20687', 'v0.0.93'),
    ('shufflenetv2_w3d2', '1237', '08c013888fc1f782683eccdb201739b05dfd43aa', 'v0.0.65'),
    ('shufflenetv2_w2', '1210', '544b55d98384504305e9c9fe9458eb5e9dd53dbd', 'v0.0.84'),
    ('shufflenetv2c_wd2', '1811', '979ce7d96d21ec1082df4e2db145373390f11af2', 'v0.0.91'),
    ('shufflenetv2c_w1', '1138', '646f3b787f2d6e88594ad0a326d2f27af54382fc', 'v0.0.95'),
    ('menet108_8x1_g3', '2030', 'aa07f925180834389cfd3bf50cb22d2501225118', 'v0.0.89'),
    ('menet128_8x1_g4', '1913', '0c890a76fb23c0af50fdec076cb16d0f0ee70355', 'v0.0.103'),
    ('menet228_12x1_g3', '1399', '8c28d22f8355345f535ca8f94a472b001095aff5', 'v0.0.6'),
    ('menet256_12x1_g4', '1390', '4502f2230e16a8e43fd84960ec4d3690bd7bb582', 'v0.0.6'),
    ('menet348_12x1_g3', '1141', 'ac69b246629131d77bf5a0a454bda28f5c2e6bc0', 'v0.0.6'),
    ('menet352_12x1_g8', '1375', '85779b8a576540ec1082a433bd5ea1ab93def27a', 'v0.0.6'),
    ('menet456_24x1_g3', '1043', '6e777068761f9c45cd0527f3824ad3b5cf36b0b5', 'v0.0.6'),
    ('mobilenet_wd4', '2218', '3185cdd29b3b964ad51fdd7820bd65f091cf281f', 'v0.0.62'),
    ('mobilenet_wd2', '1481', '9f48baf607e2c589b4aa2505fa93b4c28553c212', 'v0.0.66'),
    ('mobilenet_w3d4', '1228', 'dc11727a3917f2c795c9f286ad9cf299a165fe85', 'v0.0.7'),
    ('mobilenet_w1', '1003', 'b4fb8f1b44a91f6636782a98d81470cadd152c19', 'v0.0.7'),
    ('fdmobilenet_wd4', '3138', '2fe432fd125497dc70fa88c92a6066c2e97be974', 'v0.0.68'),
    ('fdmobilenet_wd2', '1969', '242b9fa82d54f54f08b4bdbb194b7c89030e7bc4', 'v0.0.83'),
    ('fdmobilenet_w1', '1472', 'a525b206302aa5488186f2de7dd8e6159789a463', 'v0.0.8'),
    ('mobilenetv2_wd4', '2524', 'a2468611a7424dcb8fadf294e4f3bcf862c7f3fb', 'v0.0.9'),
    ('mobilenetv2_wd2', '1464', '02fe7ff2b176f9c2056ba3bf28d1a116cd1ecc95', 'v0.0.9'),
    ('mobilenetv2_w3d4', '1126', '152672f558b4f350f82056b4d09e6c79f54eaca9', 'v0.0.9'),
    ('mobilenetv2_w1', '0990', '4e1a3878e588fc84e6317e14f3437a018223b10a', 'v0.0.9'),
    ('xception', '0561', 'a0e538e1cce773f4ea918100468dd0fe0039ff24', 'v0.0.100'),
    ('inceptionv3', '0559', '6c087967685135a321ed66b9ad2277512e9b2868', 'v0.0.92'),
    ('inceptionv4', '0532', '39d99a0bae5b0fb34f3c934733dfa3721269401d', 'v0.0.98'),
    ('inceptionresnetv2', '0502', '0b93c6ef90a505d2021cb3e83102d7a32f358d8d', 'v0.0.99'),
    ('polynet', '0453', '742803144e5a2a6148212570726350da09adf3f6', 'v0.0.96'),
    ('nasnet_4a1056', '0795', '5c78908e38c531283d86f9cbe7e14c2afd85a7ce', 'v0.0.97'),
    ('nasnet_6a4032', '0424', '73cca5fee009db77412c5fca7c826b3563752757', 'v0.0.101'),
    ('pnasnet5large', '0436', '9cc6d6744a7de969a79a65f1c4abbc9774942ff9', 'v0.0.102')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.mxnet', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $MXNET_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.params'.format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to {}.'.format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + '.zip'
    download(
        url='{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip'.format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')
