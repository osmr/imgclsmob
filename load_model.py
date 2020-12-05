"""
    Script for downloading model weights.
"""

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Download model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model name")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    from gluon.utils import prepare_model as prepare_model_gl
    prepare_model_gl(
        model_name=args.model,
        use_pretrained=True,
        pretrained_model_file_path="",
        dtype=np.float32)

    from pytorch.utils import prepare_model as prepare_model_pt
    prepare_model_pt(
        model_name=args.model,
        use_pretrained=True,
        pretrained_model_file_path="",
        use_cuda=False)

    from chainer_.utils import prepare_model as prepare_model_ch
    prepare_model_ch(
        model_name=args.model,
        use_pretrained=True,
        pretrained_model_file_path="")

    from tensorflow2.utils import prepare_model as prepare_model_tf2
    prepare_model_tf2(
        model_name=args.model,
        use_pretrained=True,
        pretrained_model_file_path="",
        use_cuda=False)


if __name__ == '__main__':
    main()
