"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '2132', 'cea565f1d8254d6dc3fdbc87568e90c34455a477', 'v0.0.108'),
    ('vgg11', '1179', '3cc057e61154ddbd152e138c31327ebd986d2b2f', 'v0.0.109'),
    ('vgg13', '1116', 'e835ca5af6ad9b9d65ffa4f19ccc544907ee4e13', 'v0.0.109'),
    ('vgg16', '0870', '8741ff5c98cd3e17bdc00a557b010c849e923b3c', 'v0.0.109'),
    ('vgg19', '0823', '18980884d7b7e46d0f564548e09af8ea8313789d', 'v0.0.109'),
    ('bn_vgg11b', '1060', '8964402b8870b2b2463b01e9ba9425737678c258', 'v0.0.110'),
    ('bn_vgg13b', '1019', '0121b0a47782b5b58c02baa148c88cdc848fc642', 'v0.0.110'),
    ('bn_vgg16b', '0863', 'cbaa2105e000ae844b4775390e9be3b30a23e02e', 'v0.0.110'),
    ('bn_vgg19b', '0816', 'dc5e37a5f6a1d5068b18011ad779062d7b4842cd', 'v0.0.110'),
    ('bninception', '0778', '99f685c2a38743e719eb0ed2ac99ee93f8898926', 'v0.0.139'),
    ('resnet10', '1549', 'b31f113596ba5fae687eb775e2dda81a293060d2', 'v0.0.22'),
    ('resnet12', '1448', '11acb729500299883bc9829028a168735275566b', 'v0.0.30'),
    ('resnet14', '1242', '4e65746b8a327f2fde5740669f5cd44dc7327e24', 'v0.0.40'),
    ('resnet16', '1107', 'b1d7fb7df91145155f6b1c45133c47ecb26996e9', 'v0.0.41'),
    ('resnet18_wd4', '2448', '58c4a0075a3a240d060a625cefe6e53bf8d28865', 'v0.0.47'),
    ('resnet18_wd2', '1499', '542ed773551add89346117be2430c9f818faeeb1', 'v0.0.46'),
    ('resnet18_w3d4', '1256', 'ce2011dfcddf9cac229d7e3a63b3764e15bcbc47', 'v0.0.22'),
    ('resnet18', '0997', '9862a84fbb34789888ffb631d64534294b312e20', 'v0.0.22'),
    ('resnet34', '0795', '0b392267b08907dc14023b24fd84df0268087002', 'v0.0.22'),
    ('resnet50', '0683', '9c795737b3ec3983de7139f817d40736fd7187fe', 'v0.0.22'),
    ('resnet50b', '0646', '225a550ed5f2d8bf0027ae7f105dbe39e041d5cc', 'v0.0.22'),
    ('resnet101', '0601', 'd8cddbea530e052e726d5a1007985beb10ec36eb', 'v0.0.22'),
    ('resnet101b', '0559', 'b5c3b4b65dd7e2c7278e35489281b7abf0fda42c', 'v0.0.22'),
    ('resnet152', '0567', '62d194fccb015a6e4517272dfa7e98a35ec6b6c6', 'v0.0.22'),
    ('resnet152b', '0539', '2b1757288ef04c89060c850d9ca725f0d589f4a5', 'v0.0.22'),
    ('preresnet18', '0954', '21e4811aa9c868bc4afb21ca773493322ba09e82', 'v0.0.140'),
    ('preresnet34', '0812', '829f5a239d51b9138d0b3d1aae5ae4a6082d9bc3', 'v0.0.23'),
    ('preresnet50', '0669', '40bd5e93861bf9ee8892cd766afbcc23a6d3b68c', 'v0.0.23'),
    ('preresnet50b', '0667', 'b7d221efa64231c2f3b83b197ddf570fb86a409b', 'v0.0.23'),
    ('preresnet101', '0575', 'f6f6789a895f681be08db6cb9ef184d9009a2f4b', 'v0.0.23'),
    ('preresnet101b', '0587', '4211c5abf0be8d849796a4af36729f74d90620d6', 'v0.0.23'),
    ('preresnet152', '0530', '021d99dc3004530a3a1f591e88807ce84e025033', 'v0.0.23'),
    ('preresnet152b', '0566', 'fdd337e701c06a928e0706ad98fa722508a4dabe', 'v0.0.23'),
    ('preresnet200b', '0560', 'f79bd952c08555e0d7bfbcfb2c8214da9c69a0c2', 'v0.0.45'),
    ('resnext101_32x4d', '0569', 'c6d1c30dcca4e83c48a2b77cfd36739a0192e244', 'v0.0.26'),
    ('resnext101_64x4d', '0543', 'dd8b7d963c2415ee1207f3705fbc33cb4ba46427', 'v0.0.26'),
    ('seresnet50', '0641', 'f3d68cfc8423b786c53390313cabfe0c4410f2d7', 'v0.0.24'),
    ('seresnet101', '0588', 'e45a9f8f09f1a7439e66032a0d79d7d5a20783b6', 'v0.0.24'),
    ('seresnet152', '0577', 'a089ba52930e9949313b9fba00a1b2e6e68f6ea4', 'v0.0.24'),
    ('seresnext50_32x4d', '0558', '5c435c1b730a0cea61b9657c8796f3c6b95ce9e8', 'v0.0.27'),
    ('seresnext101_32x4d', '0501', '98ea6fc4d36e742a01a0256707a5fa118be166dd', 'v0.0.27'),
    ('senet154', '0463', '381d2494a2ad725f62325188f94cd91c795c9902', 'v0.0.28'),
    ('airnet50_1x64d_r2', '0620', 'b6a9359d735916ff8f6192c631b7c646f489fc41', 'v0.0.120'),
    ('airnet50_1x64d_r16', '0650', '95da530f61ae4b0dda4b52c88f37bbc7cc674a03', 'v0.0.120'),
    ('airnext50_32x4d_r2', '0573', '160860f7a1750d759c36e6000080c839cda7ac56', 'v0.0.120'),
    ('bam_resnet50', '0697', 'a8c65533b4fd5e2ebf20c61d5d56936a9e1032b5', 'v0.0.124'),
    ('cbam_resnet50', '0640', 'b2314d9778b321fad2ecf3b350969038236deb96', 'v0.0.125'),
    ('pyramidnet101_a360', '0649', 'b68c786b43512e4297ce00756bd32f8beaa418ba', 'v0.0.104'),
    ('diracnet18v2', '1113', 'b85b43d13697dfbddbea6e46dea4766359fff7e5', 'v0.0.111'),
    ('diracnet34v2', '0948', '0245163a5c947bd6e07a743f17e6ca92c79c84da', 'v0.0.111'),
    ('densenet121', '0779', '06d5ebbf5b3f923ce8863268995ab5ed0f5b5019', 'v0.0.29'),
    ('densenet161', '0620', '6d05f3b9991bc570cb35fff22410d2065b667835', 'v0.0.29'),
    ('densenet169', '0686', '1978656b46c2b7de94c1e12350c74f492d683f7e', 'v0.0.29'),
    ('densenet201', '0629', '7770293931c03c2852115267dde3100d7140bbba', 'v0.0.29'),
    ('condensenet74_c4_g4', '0861', 'ef6077ec5348504346b3bcbaacbc308f825a9f87', 'v0.0.36'),
    ('condensenet74_c8_g8', '1043', '277fbfb898e0c8c7de8475184bcf5e651da10acc', 'v0.0.36'),
    ('peleenet', '1127', 'ef057fc99fda7df002d9654f0a74452e4b4b75d0', 'v0.0.141'),
    ('wrn50_2', '0613', 'd0cd9171917f04095ba8f4f48413a2ddd1ee5bc2', 'v0.0.113'),
    ('drnc26', '0788', '762c34c1f20d8ad76cec251cc0125936b608a3bc', 'v0.0.116'),
    ('drnc42', '0693', 'ec938cc429d3d0e54c34243c10be83ffae38023e', 'v0.0.116'),
    ('drnc58', '0629', '063ef19974f0158bcc6b9e4020729291462a08a3', 'v0.0.116'),
    ('drnd22', '0850', 'b25d475756dcfceb1321190b9cca6cc1f7e8e55a', 'v0.0.116'),
    ('drnd38', '0736', '153481d6f8d0b113981fc323f5b2c2ad6b2ad7f5', 'v0.0.116'),
    ('drnd54', '0623', '31e8eeb88bdbb07d8613a16471c8c5bd67ae823a', 'v0.0.116'),
    ('drnd105', '0584', 'c0d7657b2d3c4cf7d97ff407cd50dda5d1bd1880', 'v0.0.116'),
    ('dpn68', '0701', 'ad8cd4ec04a611726ee1ffcff69118a5587da691', 'v0.0.34'),
    ('dpn98', '0553', '9cd5733573f7a99062d16cd8850bb82d684704bb', 'v0.0.34'),
    ('dpn131', '0523', 'e37215991fa7e9f49245843d53de63ef1717f293', 'v0.0.34'),
    ('darknet_tiny', '1746', 'b04fa46318a78e977aa5a117786968d98d325871', 'v0.0.69'),
    ('darknet_ref', '1671', 'b2d5721f3a5f6f05cc785d57ff7a63fe82f6325e', 'v0.0.64'),
    ('squeezenet_v1_0', '1738', '4c55a6a5c7ae14b88a7989eea5a7dc60960120ef', 'v0.0.128'),
    ('squeezenet_v1_1', '1740', 'b236c2047fe1d9b283ccfaabb763143a214ecc33', 'v0.0.88'),
    ('squeezeresnet_v1_1', '1787', 'f40e60512a8b66f314f4d7ffab9b18dd31715b3a', 'v0.0.70'),
    ('sqnxt23_w1', '2213', 'e1404b066fe5d5641c1cc9380627610c2de6870d', 'v0.0.138'),
    ('shufflenet_g1_wd4', '3681', '15d3e7871b85cee9283663bbbc78dfe5e1a1a1db', 'v0.0.134'),
    ('shufflenet_g3_wd4', '3616', '064f7f7f1dd327f43e16adf5e4864a31e16d9ad9', 'v0.0.135'),
    ('shufflenetv2_wd2', '2073', 'c5e5a23c300c800d55e2f45e1dcb2e12907c0eae', 'v0.0.90'),
    ('shufflenetv2_w1', '1298', '3830a2da0701f2b31385aceeb828101008446812', 'v0.0.133'),
    ('shufflenetv2_w3d2', '1337', '66c1d6ed56e77d7bbf172e698e4a0d9f8a3bb442', 'v0.0.65'),
    ('shufflenetv2_w2', '1303', '349e42b513c3cf3fd7b0f9f647c645fce168f725', 'v0.0.84'),
    ('shufflenetv2b_wd2', '1856', '4d6e16de4798e2aa87f25cee1fbbac574955aec9', 'v0.0.112'),
    ('shufflenetv2c_wd2', '1814', '20fc1e3c18bc48b8c2f0ee0a0736b496c66e1b73', 'v0.0.94'),
    ('shufflenetv2c_w1', '1137', '2f59108aff47f73888bf8a374c8c89dfce951eef', 'v0.0.95'),
    ('menet108_8x1_g3', '2042', '9e3ff283ac81b4f4e6d4a5b11d8d54b63f4aa2f0', 'v0.0.89'),
    ('menet128_8x1_g4', '1919', 'f6fd56fae09d0c528c902d1381f7cf401590d130', 'v0.0.103'),
    ('menet228_12x1_g3', '1301', '39c25ca345751cac91395a602565796393fea60d', 'v0.0.131'),
    ('menet256_12x1_g4', '1391', 'ee68bd6fbb6c6c248a625435344bc615325d50a1', 'v0.0.33'),
    ('menet348_12x1_g3', '1140', '49feaea78bc6831b1c472d0aa52cbc38679918d5', 'v0.0.33'),
    ('menet352_12x1_g8', '1368', '2d523fac34b7863f0fab00fd5cf087b33c274708', 'v0.0.33'),
    ('menet456_24x1_g3', '1039', 'f68c36a2a19f1fe625a2b02cb855a42012a0a32b', 'v0.0.33'),
    ('mobilenet_wd4', '2216', '09c50ab8d72049a4aa9cae4bd1502859522b9a70', 'v0.0.62'),
    ('mobilenet_wd2', '1486', '90e62dd62af971cdbe9b8c47318d01342c1dcb37', 'v0.0.66'),
    ('mobilenet_w3d4', '1053', 'd7ec3192f88b7017d477fdb704ad6ad77a4c5cc1', 'v0.0.130'),
    ('mobilenet_w1', '1031', '3ecb405b83bbf772ef15ae304d0ccdebda7cb326', 'v0.0.21'),
    ('fdmobilenet_wd4', '3145', '6718fb0745135de28d98700e15fa66cae3d9bcfe', 'v0.0.68'),
    ('fdmobilenet_wd2', '1976', '6299d44272390440be808e58059219b0d57907e4', 'v0.0.83'),
    ('fdmobilenet_w1', '1374', '99c7854b46b971baec95c02ed5266b1a0468f1ae', 'v0.0.129'),
    ('mobilenetv2_wd4', '2411', '9fc398d348226c410659464d12b0fe6b7d4506e7', 'v0.0.137'),
    ('mobilenetv2_wd2', '1498', '4b767a983ab4b42f29f00ac63eb9a0a56b5af69e', 'v0.0.31'),
    ('mobilenetv2_w3d4', '1148', 'a6f852ea49ed066b2db2a43054c4e2fa7f28f8bb', 'v0.0.31'),
    ('mobilenetv2_w1', '1005', '3b6d1764934efd35d4cf402ea5194546dc5004e4', 'v0.0.31'),
    ('igcv3_wd2', '1704', '86246558ade35232344a4c448288ae3927143f9c', 'v0.0.132'),
    ('igcv3_w1', '0955', '1c00ac33b7326aef50a1b29c477ce75217b6c5ee', 'v0.0.126'),
    ('mnasnet', '1144', '688e523d02834b34f4a693a1b18e7a523483eb58', 'v0.0.117'),
    ('darts', '0897', '8986fe64b3f853704a88010f0a735a9e6e33bd97', 'v0.0.118'),
    ('xception', '0547', '7a5be9582fd7a4771ede5290645be394d66d29ca', 'v0.0.115'),
    ('inceptionv3', '0561', '4ddea4df44f132ffc9e2b22b1e7d686f8b59703b', 'v0.0.92'),
    ('inceptionv4', '0526', '02e53701d1bda64b057b41fa90d8e04a17d07f66', 'v0.0.105'),
    ('inceptionresnetv2', '0492', '3d3de82bb9db27b260603fe2f956ad929c3eb277', 'v0.0.107'),
    ('polynet', '0450', '6dc7028b0edc48c452f83dd38448b1242c554a5e', 'v0.0.96'),
    ('nasnet_4a1056', '0796', 'f09950c0f4a333007dc33049531534b8cd9f8521', 'v0.0.97'),
    ('nasnet_6a4032', '0422', 'd49d46631abda0ec7ac4a0076e6f8d05bf99b7d1', 'v0.0.101'),
    ('pnasnet5large', '0426', '3c2755dce80a29dea19b398dce514a640da2aaa3', 'v0.0.114')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.chainer', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $CHAINER_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.npz'.format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to {}.'.format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + '.zip'
    _download(
        url='{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip'.format(
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
        raise ValueError('Downloaded file has different hash. Please try again.')


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
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print('Downloading {} from {}...'.format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'
                                      ' The repo may be outdated or download may be incomplete. '
                                      'If the "repo_url" is overridden, consider switching to '
                                      'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, 's' if retries > 1 else ''))

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
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash
