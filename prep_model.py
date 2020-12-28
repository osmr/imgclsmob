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
                 line_count):
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
    line_count : int
        Log file last line count for analysis.
    """
    with open(log_file_path, "r") as f:
        log_file_tail = f.read().splitlines()[-line_count:]
    top5_err = re.findall(r"\d+\.\d+", re.findall(r", err-top5=\d+\.\d+", log_file_tail)[0])[0].split(".")[1]

    model_file_ext = ".".join(os.path.basename(model_file_path).split(".")[1:])

    sha1_value = calc_sha1(model_file_path)

    dst_model_file_name = "{}-{}-{}.{}".format(model_name, top5_err, sha1_value[:8], model_file_ext)
    dst_model_file_path = os.path.join(dst_dir_path, dst_model_file_name)
    os.rename(model_file_path, dst_model_file_path)
    os.rename(log_file_path, dst_model_file_path + ".log")

    with zipfile.ZipFile(dst_model_file_path + ".zip", "w") as zf:
        zf.write(dst_model_file_path)

    return top5_err, sha1_value


def process_pt(model_name,
               root_dir_path,
               model_file_path,
               log_file_path):
    """
    Parse python script parameters.

    Parameters:
    ----------
    model_name : str
        Model name.
    root_dir_path : str
        Root dir path.
    model_file_path : str
        Model file path.
    log_file_path : str
        Log file path.
    """
    pt_dir_path = os.path.join(root_dir_path, "pt-{}".format(model_name))
    os.mkdir(pt_dir_path)
    shutil.copy2(log_file_path, pt_dir_path)
    pt_model_file_path = os.path.join(pt_dir_path, "{}.pth".format(model_name))
    command = "python3 convert_models.py --src-fwk=gluon --dst-fwk=pytorch --src-model={model_name}" \
              " --dst-model={model_name} --src-params={model_file_path} --dst-params={pt_model_file_path}" \
              " --save-dir={pt_dir_path}"
    subprocess.call([command.format(
        model_name=model_name,
        model_file_path=model_file_path,
        pt_model_file_path=pt_model_file_path,
        pt_dir_path=pt_dir_path)], shell=True)
    command = "python3 eval_pt.py --model={model_name} --resume={pt_model_file_path} --save-dir={pt_dir_path}" \
              " --num-gpus=1 --batch-size=100 -j=4 --calc-flops"
    subprocess.call([command.format(
        model_name=model_name,
        pt_model_file_path=pt_model_file_path,
        pt_dir_path=pt_dir_path)], shell=True)

    pt_log_file_path = os.path.join(pt_dir_path, "train.log")
    line_count = 4
    with open(pt_log_file_path, "r") as f:
        log_file_tail = f.read().splitlines()[-line_count:]
    top5_err = re.findall(r"\d+\.\d+", re.findall(r", err-top5=\d+\.\d+", log_file_tail)[0])[0].split(".")[1]

    sha1_value = calc_sha1(pt_model_file_path)

    dst_pt_model_file_path = os.path.join(pt_dir_path, "{}-{}-{}.pth".format(model_name, top5_err, sha1_value[:8]))
    os.rename(pt_model_file_path, dst_pt_model_file_path)


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

    command = "python3 eval_gl.py --model={model_name} --resume={model_file_path} --save-dir={root_dir_path}" \
              " --num-gpus=1 --batch-size=100 -j=4 --calc-flops"
    subprocess.call([command.format(
        model_name=model_name,
        model_file_path=model_file_path,
        root_dir_path=root_dir_path)], shell=True)

    dst_dir_path = os.path.join(root_dir_path, "_result")
    os.mkdir(dst_dir_path)

    gl_log_file_path = os.path.join(dst_dir_path, "train.log")
    shutil.copy2(log_file_path, gl_log_file_path)

    gl_model_file_path = os.path.join(dst_dir_path, "{}.params".format(model_name))
    shutil.copy2(model_file_path, gl_model_file_path)

    types = []
    top5s = []
    sha1s = []

    top5_err, sha1_value = post_process(
        dst_dir_path=dst_dir_path,
        model_name=model_name,
        model_file_path=gl_model_file_path,
        log_file_path=gl_log_file_path,
        line_count=4)
    types.append("gl")
    top5s.append(top5_err)
    sha1s.append(sha1_value)

    slice_df_dict = {
        "Type": types,
        "Top5": top5s,
        "Sha1": sha1s,
    }
    slice_df = pd.DataFrame(slice_df_dict)
    slice_df.to_csv(
        os.path.join(root_dir_path, "info.csv"),
        sep="\t",
        index=False)

    # process_pt(
    #     model_name=model_name,
    #     root_dir_path=root_dir_path,
    #     model_file_path=model_file_path,
    #     log_file_path=log_file_path)


if __name__ == '__main__':
    main()
