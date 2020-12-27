"""
    Routines for logging environment setting.
"""

__all__ = ['get_env_stats']

import os
import sys
import subprocess
import platform
import json


def get_pip_versions(package_list,
                     python_version=""):
    """
    Get packages information by using 'pip show' command.

    Parameters:
    ----------
    package_list : list of str
        List of package names.
    python_version : str, default ''
        Python version ('2', '3', '') appended to 'pip' command.

    Returns:
    -------
    dict
        Dictionary with module descriptions.
    """
    module_versions = {}
    for module in package_list:
        try:
            out_bytes = subprocess.check_output([
                "pip{0}".format(python_version),
                "show", module])
            out_text = out_bytes.decode("utf-8").strip()
        except (subprocess.CalledProcessError, OSError):
            out_text = None
        module_versions[module] = out_text
    return module_versions


def get_package_versions(package_list):
    """
    Get packages information by inspecting __version__ attribute.

    Parameters:
    ----------
    package_list : list of str
        List of package names.

    Returns:
    -------
    dict
        Dictionary with module descriptions.
    """
    module_versions = {}
    for module in package_list:
        try:
            module_versions[module] = __import__(module).__version__
        except ImportError:
            module_versions[module] = None
        except AttributeError:
            module_versions[module] = "unknown"
    return module_versions


def get_pyenv_info(packages,
                   pip_packages,
                   python_ver,
                   pwd,
                   git,
                   sys_info=True):
    """
    Get all available information about Python environment: packages information, Python version, current path,
    git revision.

    Parameters:
    ----------
    packages : list of str
        list of package names to inspect only __version__.
    pip_packages : list of str
        List of package names to inspect by 'pip show'.
    python_ver : bool
        Whether to show python version.
    pwd : bool
        Whether to show pwd.
    git : bool
        Whether to show git info.
    sys_info : bool, default True
        Whether to show platform info.

    Returns:
    -------
    dict
        Dictionary with module descriptions.
    """
    pyenv_info = {}

    python_version = sys.version_info[0]

    # get versions from __version__ string
    modules_versions = get_package_versions(packages)
    pyenv_info.update(modules_versions)

    # get versions from pip
    if type(pip_packages) == list and len(pip_packages) > 0 and pip_packages[0]:
        modules_versions_pip = get_pip_versions(pip_packages, python_version)
        pyenv_info.update(modules_versions_pip)

    if python_ver:
        # set python version
        try:
            pyenv_info["python"] = "{0}.{1}.{2}".format(*sys.version_info[0:3])
        except BaseException:
            pyenv_info["python"] = "unknown"

    if pwd:
        # set current path
        pyenv_info["pwd"] = os.path.dirname(os.path.realpath(__file__))

    if git:
        # set git revision of the code
        try:
            if os.name == "nt":
                command = "cmd /V /C \"cd {} && git log -n 1\"".format(pyenv_info["pwd"])
            else:
                command = ["cd {}; git log -n 1".format(pyenv_info["pwd"])]
            out_bytes = subprocess.check_output(command, shell=True)
            out_text = out_bytes.decode("utf-8")
        except BaseException:
            out_text = "unknown"
        pyenv_info["git"] = out_text.strip()

    if sys_info:
        pyenv_info["platform"] = platform.platform()

    return pyenv_info


def pretty_print_dict2str(d):
    """
    Pretty print of dictionary d to json-formated string.

    Parameters:
    ----------
    d : dict
        Dictionary with module descriptions.

    Returns:
    -------
    str
        Resulted string.
    """
    out_text = json.dumps(d, indent=4)
    return out_text


def get_env_stats(packages,
                  pip_packages,
                  python_ver=True,
                  pwd=True,
                  git=True,
                  sys_info=True):
    """
    Get environment statistics.

    Parameters:
    ----------
    packages : list of str
        list of package names to inspect only __version__.
    pip_packages : list of str
        List of package names to inspect by 'pip show'.
    python_ver : bool
        Whether to show python version.
    pwd : bool, default True
        Whether to show pwd.
    git : bool, default True
        Whether to show git info.
    sys_info : bool, default True
        Whether to show platform info.

    Returns:
    -------
    str
        Resulted string with information.
    """
    package_versions = get_pyenv_info(packages, pip_packages, python_ver, pwd, git, sys_info)
    return pretty_print_dict2str(package_versions)
