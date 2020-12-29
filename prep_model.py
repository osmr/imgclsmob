"""
    Script for preparing the model for publication.
"""

import os
import argparse
import subprocess
import shutil
import re
import hashlib
import zipfile
import pandas as pd


def parse_args():
    """
    Parse python script parameters.

    Returns:
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(description="Prepare model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model name")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="model weights (Gluon) file path")
    args = parser.parse_args()
    return args


def calc_sha1(file_name):
    """
    Calculate sha1 hash of the file content.

    Parameters:
    ----------
    file_name : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns:
    -------
    str
        sha1 hex digest.
    """
    sha1 = hashlib.sha1()
    with open(file_name, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()


def post_process(dst_dir_path,
                 model_name,
                 model_file_path,
                 log_file_path,
                 dst_model_file_ext,
                 log_line_num):
    """
    Post-process weight/log files.

    Parameters:
    ----------
    dst_dir_path : str
        Destination dir path.
    model_name : str
        Model name.
    model_file_path : str
        Model file path.
    log_file_path : str
        Log file path.
    dst_model_file_ext : str
        Destination model file extension.
    log_line_num : int
        Log file last line number for analysis.

    Returns:
    -------
    top5_err : str
        top5 error value.
    sha1_value : str
        sha1 hex digest.
    """
    with open(log_file_path, "r") as f:
        log_file_tail = f.read().splitlines()[log_line_num]
    top5_err = re.findall(r"\d+\.\d+", re.findall(r", err-top5=\d+\.\d+", log_file_tail)[0])[0].split(".")[1]

    sha1_value = calc_sha1(model_file_path)

    dst_model_file_name = "{}-{}-{}.{}".format(model_name, top5_err, sha1_value[:8], dst_model_file_ext)
    dst_model_file_path = os.path.join(dst_dir_path, dst_model_file_name)
    os.rename(model_file_path, dst_model_file_path)
    os.rename(log_file_path, dst_model_file_path + ".log")

    with zipfile.ZipFile(dst_model_file_path + ".zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(filename=dst_model_file_path, arcname=dst_model_file_name)
    os.remove(dst_model_file_path)

    return top5_err, sha1_value


def process_fwk(prep_info_dict,
                dst_framework,
                dst_dir_path,
                model_name,
                model_file_path,
                log_file_path):
    """
    Process weights on specific framework.

    Parameters:
    ----------
    prep_info_dict : dict
        Dictionary with preparation meta-info.
    dst_dir_path : str
        Destination dir path.
    model_name : str
        Model name.
    model_file_path : str
        Model file path.
    log_file_path : str
        Log file path.
    dst_framework : str
        Destination framework.
    """
    if dst_framework == "gluon":
        dst_model_file_ext = "params"
        eval_script = "eval_gl"
        num_gpus = 1
        calc_flops = "--calc-flops"
        log_line_num = -3
    elif dst_framework == "pytorch":
        dst_model_file_ext = "pth"
        eval_script = "eval_pt"
        num_gpus = 1
        calc_flops = "--calc-flops"
        log_line_num = -3
    elif dst_framework == "chainer":
        dst_model_file_ext = "npz"
        eval_script = "eval_ch"
        num_gpus = 0
        calc_flops = ""
        log_line_num = -2
    elif dst_framework == "tf2":
        dst_model_file_ext = "tf2.h5"
        eval_script = "eval_tf2"
        num_gpus = 1
        calc_flops = ""
        log_line_num = -2
    else:
        raise ValueError("Unknown framework: {}".format(dst_framework))

    dst_raw_log_file_path = os.path.join(dst_dir_path, "train.log")
    shutil.copy2(log_file_path, dst_raw_log_file_path)

    dst_raw_model_file_path = os.path.join(dst_dir_path, "{}.{}".format(model_name, dst_model_file_ext))

    if dst_framework == "gluon":
        shutil.copy2(model_file_path, dst_raw_model_file_path)
    else:
        command = "python3 convert_models.py --src-fwk=gluon --dst-fwk={dst_framework} --src-model={model_name}" \
                  " --dst-model={model_name} --src-params={model_file_path} --dst-params={dst_raw_model_file_path}" \
                  " --save-dir={dst_dir_path}"
        subprocess.call([command.format(
            dst_framework=dst_framework,
            model_name=model_name,
            model_file_path=model_file_path,
            dst_raw_model_file_path=dst_raw_model_file_path,
            dst_dir_path=dst_dir_path)], shell=True)

    command = "python3 {eval_script}.py --model={model_name} --resume={dst_raw_model_file_path}" \
              " --save-dir={dst_dir_path} --num-gpus={num_gpus} --batch-size=100 -j=4 {calc_flops}"
    subprocess.call([command.format(
        eval_script=eval_script,
        model_name=model_name,
        dst_raw_model_file_path=dst_raw_model_file_path,
        dst_dir_path=dst_dir_path,
        num_gpus=num_gpus,
        calc_flops=calc_flops)], shell=True)

    if dst_framework == "gluon":
        shutil.copy2(dst_raw_log_file_path, log_file_path)

    top5_err, sha1_value = post_process(
        dst_dir_path=dst_dir_path,
        model_name=model_name,
        model_file_path=dst_raw_model_file_path,
        log_file_path=dst_raw_log_file_path,
        dst_model_file_ext=dst_model_file_ext,
        log_line_num=log_line_num)

    prep_info_dict["Type"].append(dst_framework)
    prep_info_dict["Top5"].append(top5_err)
    prep_info_dict["Sha1"].append(sha1_value)


def main():
    args = parse_args()
    model_name = args.model

    model_file_path = os.path.expanduser(args.resume)
    if not os.path.exists(model_file_path):
        raise Exception("Model file doesn't exist: {}".format(model_file_path))
    root_dir_path = os.path.dirname(model_file_path)

    log_file_path = os.path.join(root_dir_path, "train.log")
    if not os.path.exists(log_file_path):
        raise Exception("Log file doesn't exist: {}".format(log_file_path))

    dst_dir_path = os.path.join(root_dir_path, "_result")
    if not os.path.exists(dst_dir_path):
        os.mkdir(dst_dir_path)

    prep_info_dict = {
        "Type": [],
        "Top5": [],
        "Sha1": [],
    }

    dst_frameworks = ["gluon", "pytorch", "chainer", "tf2"]
    # dst_frameworks = ["tf2"]
    for dst_framework in dst_frameworks:
        process_fwk(
            prep_info_dict=prep_info_dict,
            dst_framework=dst_framework,
            dst_dir_path=dst_dir_path,
            model_name=model_name,
            model_file_path=model_file_path,
            log_file_path=log_file_path)

    prep_info_df = pd.DataFrame(prep_info_dict)
    prep_info_df.to_csv(
        os.path.join(root_dir_path, "prep_info.csv"),
        sep="\t",
        index=False)


if __name__ == '__main__':
    main()
