"""
    Script for preparing the model for publication.
"""

import os
import argparse
import subprocess
import shutil


def parse_args():
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

    gl_dir_path = os.path.join(root_dir_path, "gl-{}".format(model_name))
    os.mkdir(gl_dir_path)
    shutil.copy2(log_file_path, gl_dir_path)
    gl_model_file_path = os.path.join(gl_dir_path, "{}.params".format(model_name))
    shutil.copy2(model_file_path, gl_model_file_path)

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


if __name__ == '__main__':
    main()
