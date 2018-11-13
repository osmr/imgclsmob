"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file', 'load_model', 'download_model']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('resnet10', '1585', 'ef8a3ae358543fbce38088ac4955f41ce860e45b', 'v0.0.1'),
    ('resnet12', '1480', 'c2263f735b9af6e692bccbacfbe7d7f357e7f57d', 'v0.0.30'),
    ('resnet14', '1271', '568c392e3e94041dd589d1b8164e337f90440d06', 'v0.0.40'),
    ('resnet16', '1138', '3a5aa7c0164af3b66faf5b9974352a86d6a02974', 'v0.0.41'),
    ('resnet18_wd4', '2483', '6ef2515c60ccf004a055954fdb56f46dd786ee20', 'v0.0.47'),
    ('resnet18_wd2', '1538', '671466b5e3b952eac4e8c60f57380d11bc1d3a05', 'v0.0.46'),
    ('resnet18_w3d4', '1285', '94713e0e1780a9f19b2a5f4575eb254f8a67b556', 'v0.0.18'),
    ('resnet18', '1021', 'b0d7daeaab950f2a7106c8062c7435627b3fe3da', 'v0.0.1'),
    ('resnet34', '0818', '6f947d409313c862a1ef22c46e29b09c85eb9abf', 'v0.0.1'),
    ('resnet50', '0705', 'f7a2027ee704dac27a9b5184276dd7fd7969b9e8', 'v0.0.1'),
    ('resnet50b', '0665', '89691746c7fc77f79e2ce4e939001c59c99c25f2', 'v0.0.1'),
    ('resnet101', '0622', 'ab0cf005bbe9b17e53f9e3c330c6147a8c80b3a5', 'v0.0.1'),
    ('resnet101b', '0581', 'd983e68295a38498cf7b2feb4dd0dc33102201b0', 'v0.0.1'),
    ('resnet152', '0582', 'af1a3bd5285762330a8e8a5e7ec1ba23ed429e55', 'v0.0.1'),
    ('resnet152b', '0550', '216604cf5b7014f1349a879270926ef57273f952', 'v0.0.1'),
    ('preresnet18', '1018', '98958fd2719a5b824345b004cc3b76804a6179fb', 'v0.0.39'),
    ('preresnet34', '0841', 'b4dd761f32f603e4ea352f73ab84c0db3d5299af', 'v0.0.2'),
    ('preresnet50', '0685', 'd81a7aca0384c6d65ee0e5c1f3ba854591466346', 'v0.0.2'),
    ('preresnet50b', '0687', '65be98fbe7b82c79bccd9c794ce9d9a3482aec9c', 'v0.0.2'),
    ('preresnet101', '0591', '4bacff796e113562e1dfdf71cfa7c6ed33e0ba86', 'v0.0.2'),
    ('preresnet101b', '0603', 'b1e37a09424dde15ecba72365d46b1f59abd479b', 'v0.0.2'),
    ('preresnet152', '0555', 'c842a030abbcc21a0f2a9a8299fc42204897a611', 'v0.0.14'),
    ('preresnet152b', '0591', '2c91ab2c8d90f3990e7c30fd6ee2184f6c2c3bee', 'v0.0.2'),
    ('preresnet200b', '0588', 'f7104ff306ed5de2c27f3c855051c22bda167981', 'v0.0.45'),
    ('resnext101_32x4d', '0611', 'cf962440f11fe683fd02ec04f2102d9f47ce38a7', 'v0.0.10'),
    ('resnext101_64x4d', '0575', '651abd029bcc4ce88c62e1d900a710f284a8281e', 'v0.0.10'),
    ('seresnet50', '0640', '8820f2af62421ce2e1df989d6e0ce7916c78ff86', 'v0.0.11'),
    ('seresnet101', '0589', '5e6e831b7518b9b8a049dd60ed1ff82ae75ff55e', 'v0.0.11'),
    ('seresnet152', '0576', '814cf72e0deeab530332b16fb9b609e574afec61', 'v0.0.11'),
    ('seresnext50_32x4d', '0554', '99e0e9aa4578af9f15045c1ceeb684a2e988628a', 'v0.0.12'),
    ('seresnext101_32x4d', '0505', '0924f0a2c1de90dc964c482b7aff6232dbef3600', 'v0.0.12'),
    ('senet154', '0461', '6512228c820897cd09f877527a553ca99d673956', 'v0.0.13'),
    ('densenet121', '0803', 'f994107a83aed162916ff89e2ded4c5af5bc6457', 'v0.0.3'),
    ('densenet161', '0644', 'c0fb22c83e8077a952ce1a0c9703d1a08b2b9e3a', 'v0.0.3'),
    ('densenet169', '0719', '271391051775ba9bbf458a6bd77af4b3007dc892', 'v0.0.3'),
    ('densenet201', '0663', '71ece4ad7be5d1e2aa4bbf6f1a6b32ac2562d847', 'v0.0.3'),
    ('condensenet74_c4_g4', '0828', '5ba550494cae7081d12c14b02b2a02365539d377', 'v0.0.4'),
    ('condensenet74_c8_g8', '1006', '3574d874fefc3307f241690bad51f20e61be1542', 'v0.0.4'),
    ('dpn68', '0727', '438492331840612ff1700e7b7d52dd6c0c683b47', 'v0.0.17'),
    ('dpn98', '0553', '52c55969835d56185afa497c43f09df07f58f0d3', 'v0.0.17'),
    ('dpn131', '0548', '0c53e5b380137ccb789e932775e8bd8a811eeb3e', 'v0.0.17'),
    ('darknet_tiny', '1784', '4561e1ada619e33520d1f765b3321f7f8ea6196b', 'v0.0.69'),
    ('darknet_ref', '1718', '034595b49113ee23de72e36f7d8a3dbb594615f6', 'v0.0.64'),
    ('squeezenet_v1_0', '1932', 'e4017303477daaaee6dfb687c17717c1c82c59f5', 'v0.0.19'),
    ('squeezenet_v1_1', '1772', '25b77bc39e35612abbe7c2344d2c3e1e6756c2f8', 'v0.0.88'),
    ('squeezeresnet_v1_1', '1821', 'c27ed88f1b19eb233d3925efc71c71d25e4c434e', 'v0.0.70'),
    ('shufflenetv2_wd2', '1865', '9c22238b5fa9c09541564e8ed7f357a5f7e8cd7c', 'v0.0.90'),
    ('shufflenetv2_w1', '1354', '73e7c9fdcc377303112b61e27d90923fc907ca9b', 'v0.0.93'),
    ('shufflenetv2_w3d2', '1269', '536ad5b11bc169022ea6752cc288fb468b2856bf', 'v0.0.65'),
    ('shufflenetv2_w2', '1249', 'b9f9e84cbf49cf63fe2a89e9c48a9fb107f591d7', 'v0.0.84'),
    ('shufflenetv2c_wd2', '1851', 'e1d36c5d8c6de9485c6c3bba6d0bdbed95226b68', 'v0.0.94'),
    ('shufflenetv2c_w1', '1161', '8cdbbcc14462424d97bdf250785d9f0cbbe16c58', 'v0.0.95'),
    ('menet108_8x1_g3', '2076', '6acc82e46dfc1ce0dd8c59668aed4a464c8cbdb5', 'v0.0.89'),
    ('menet128_8x1_g4', '1959', '48fa80fc363adb88ff580788faa8053c9d7507f3', 'v0.0.103'),
    ('menet228_12x1_g3', '1328', '27991387c8fa27f98785d1910a76e9a9a19f42c5', 'v0.0.6'),
    ('menet256_12x1_g4', '1326', 'e5d35476f4082dac61dd2d15a597d4365ad39793', 'v0.0.6'),
    ('menet348_12x1_g3', '1092', '66be1a1896fa0bea27290580e8b98057dfdbda2c', 'v0.0.6'),
    ('menet352_12x1_g8', '1308', 'e91ec72ce2d0c3c2bf2a3cba6719c6b23ea7c736', 'v0.0.6'),
    ('menet456_24x1_g3', '0993', 'cb9fd37660b6064f44a6c779a330a967b2b41c2d', 'v0.0.6'),
    ('mobilenet_wd4', '2249', '1ad5e8fe8674cdf7ffda8450095eb96d227397e0', 'v0.0.62'),
    ('mobilenet_wd2', '1514', 'cc15154a68a000aa0b6cafee2e4fdef0fd742893', 'v0.0.66'),
    ('mobilenet_w3d4', '1285', 'b8022faebe280b6e6571bec3a4bb6e293895a72d', 'v0.0.7'),
    ('mobilenet_w1', '1036', '34f7a0cb20c4c8d81359c1b720b2b864e1527d12', 'v0.0.7'),
    ('fdmobilenet_wd4', '3132', '0b242eff0420fdb03e388fc2eb69692ec51b3790', 'v0.0.8'),
    ('fdmobilenet_wd2', '2015', '414dbeedb2f829dcd8f94cd7fef10aae6829f06f', 'v0.0.83'),
    ('fdmobilenet_w1', '1405', 'a653887955849c6502641509832ee041857fa8fb', 'v0.0.8'),
    ('mobilenetv2_wd4', '2587', '189d4ea2b49e91a89d426799cd39f6fa8d8dd9cb', 'v0.0.9'),
    ('mobilenetv2_wd2', '1519', 'd0937a23f6fc60320656b6397a76ae1ee12edd95', 'v0.0.9'),
    ('mobilenetv2_w3d4', '1176', '1b966ff415a784c3ec642b628b170730f38352d2', 'v0.0.9'),
    ('mobilenetv2_w1', '1039', '7532eb72394c58005351c0986d6ae604528657df', 'v0.0.9'),
    ('xception', '0568', '9e15f3e161dadbe295830546873bdab34d13c3ec', 'v0.0.100'),
    ('inceptionv3', '0565', 'cf4061800bc1dc3b090920fc9536d8ccc15bb86e', 'v0.0.92'),
    ('inceptionv4', '0532', '6bdbf383118c588f4b3a0e0b1fbae6ae15c6d7a8', 'v0.0.98'),
    ('inceptionresnetv2', '0500', 'e236401fde0eaf7ed0a9d70f5166aea9213e20d1', 'v0.0.99'),
    ('polynet', '0452', '6a1b295dad3f261b48e845f1b283e4eef3ab5a0b', 'v0.0.96'),
    ('nasnet_4a1056', '0816', 'd21bbaf5e937c2e06134fa40e7bdb1f501423b86', 'v0.0.97'),
    ('nasnet_6a4032', '0421', 'f354d28f4acdde399e081260c3f46152eca5d27e', 'v0.0.101'),
    ('pnasnet5large', '0436', '275d1239da0edad178734697e4870161aa2c52d9', 'v0.0.102')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.torch', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.pth'.format(
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
    """
    Download an given URL

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
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
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


def _check_sha1(file_name, sha1_hash):
    """
    Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    file_name : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(file_name, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def load_model(net,
               file_path,
               ignore_extra=True):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


def download_model(net,
                   model_name,
                   local_model_store_dir_path=os.path.join('~', '.torch', 'models'),
                   ignore_extra=True):
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
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    load_model(
        net=net,
        file_path=get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path),
        ignore_extra=ignore_extra)
