import os, sys
import subprocess
import json

PACKAGES = ['mxnet']
PACKAGES_PIP = ['mxnet-cu92', 'gluoncv']


def get_pip_versions(package_list, python_version=''):
    """
    Get packages information by using 'pip show' command.
    
    :param package_list: list of package names
    :param python_version: python version ('2', '3', '') appended to 'pip'
    command.
    :return: dict of module: version_info pairs
    """
    module_versions = {}
    for module in package_list:
        try:
            out_bytes = subprocess.check_output(['pip{0}'.format(python_version),
                                                 'show', module])
            out_text = out_bytes.decode('utf-8').strip()
        except subprocess.CalledProcessError:
            out_text = None
        module_versions[module] = out_text
    
    return module_versions


def get_package_versions(package_list):
    """
    Get packages information by inspecting __version__ attribute.
    
    :param package_list: list of package names
    :return: dict of module: version_info pairs
    """
    module_versions = {}
    for module in package_list:
        try:
            module_versions[module] = __import__(module).__version__
        except ImportError:
            module_versions[module] = None
        except AttributeError:
            module_versions[module] = 'unknown'
    return module_versions


def get_pyenv_info(package_list, package_list_pip):
    """
    Get all available information about Python environment: packages
    information, Python version, current path, git revision
    
    :param package_list: list of package names to inspect only __version__
    attributes.
    :param package_list_pip: list of package names to inspect by 'pip show'.
    :return: dictionary attribute: version_info
    """

    # get current path
    pyenv_info = {'pwd': os.path.dirname(os.path.realpath(__file__))}
    # get python version
    python_version = sys.version_info[0]
    try:
        pyenv_info["python"] = "{0}.{1}.{2}".format(*sys.version_info[0:3])
    except:
        pyenv_info["python"] = "unknown"
    
    # get versions from __version__ string
    modules_versions = get_package_versions(package_list)
    pyenv_info.update(modules_versions)
    
    # get versions from pip
    modules_versions_pip = get_pip_versions(package_list_pip, python_version)
    pyenv_info.update(modules_versions_pip)
    
    # get git revision of the code
    try:
        out_bytes = subprocess.check_output(
            ['cd {0}; git log -n 1'.format(pyenv_info['pwd'])], shell=True)
        out_text = out_bytes.decode('utf-8')
    except:
        out_text = 'unknown'
    pyenv_info["git"] = out_text.strip()
    
    return pyenv_info


def prettyprint_dict2str(d):
    """
    Pretty print of dictionary d to json-formated string.
    """
    out_text = json.dumps(d, indent=4)
    return out_text


def get_env_stats():
    """
    Get env statistics.
    """
    package_versions = get_pyenv_info(PACKAGES, PACKAGES_PIP)
    return prettyprint_dict2str(package_versions)


if __name__ == "__main__":
    out_text = get_env_stats()
    print(out_text)

