"""
    Routines for logging subsystem initialization.
"""

__all__ = ['initialize_logging']

import os
import sys
import logging
from .env_stats import get_env_stats


def prepare_logger(logging_dir_path,
                   logging_file_name):
    """
    Prepare logger.

    Parameters:
    ----------
    logging_dir_path : str
        Path to logging directory.
    logging_file_name : str
        Name of logging file.

    Returns
    -------
    Logger
        Logger instance.
    bool
        If the logging file exist.
    """
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # sh = logging.StreamHandler()
    # logger.addHandler(sh)
    log_file_exist = False
    if logging_dir_path is not None and logging_dir_path:
        log_file_path = os.path.join(logging_dir_path, logging_file_name)
        if not os.path.exists(logging_dir_path):
            os.makedirs(logging_dir_path)
            log_file_exist = False
        else:
            log_file_exist = (os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        if log_file_exist:
            logging.info("--------------------------------")
    return logger, log_file_exist


def initialize_logging(logging_dir_path,
                       logging_file_name,
                       script_args,
                       log_packages,
                       log_pip_packages):
    """
    Initialize logging subsystem.

    Parameters:
    ----------
    logging_dir_path : str
        Path to logging directory.
    logging_file_name : str
        Name of logging file.
    script_args : ArgumentParser
        Main script arguments.
    log_packages : bool
        Whether to log packages info.
    log_pip_packages : bool
        Whether to log pip-packages info.

    Returns
    -------
    Logger
        Logger instance.
    bool
        If the logging file exist.
    """
    logger, log_file_exist = prepare_logger(
        logging_dir_path=logging_dir_path,
        logging_file_name=logging_file_name)
    logging.info("Script command line:\n{}".format(" ".join(sys.argv)))
    logging.info("Script arguments:\n{}".format(script_args))
    packages = log_packages.replace(" ", "").split(",") if type(log_packages) == str else log_packages
    pip_packages = log_pip_packages.replace(" ", "").split(",") if type(log_pip_packages) == str else log_pip_packages
    if (log_packages is not None) and (log_pip_packages is not None):
        logging.info("Env_stats:\n{}".format(get_env_stats(
            packages=packages,
            pip_packages=pip_packages)))
    return logger, log_file_exist
