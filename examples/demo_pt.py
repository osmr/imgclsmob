"""
    Script for evaluating trained model on PyTorch / ImageNet-1K (demo mode).
"""

import math
import argparse
import numpy as np
import cv2
import torch
from gluoncv.data import ImageNet1kAttr
from pytorchcv.model_provider import get_model as ptcv_get_model


def parse_args():
    """
    Create python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an ImageNet-1K model on PyTorch (demo mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="path to testing image")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of gpus to use")
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="size of the input for model")
    parser.add_argument(
        "--resize-inv-factor",
        type=float,
        default=0.875,
        help="inverted ratio for input image crop")
    parser.add_argument(
        "--mean-rgb",
        nargs=3,
        type=float,
        default=(0.485, 0.456, 0.406),
        help="Mean of RGB channels in the dataset")
    parser.add_argument(
        "--std-rgb",
        nargs=3,
        type=float,
        default=(0.229, 0.224, 0.225),
        help="STD of RGB channels in the dataset")

    args = parser.parse_args()
    return args


def main():
    """
    Main body of script.
    """
    args = parse_args()

    # Load a testing image:
    image = cv2.imread(args.image, flags=cv2.IMREAD_COLOR)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)

    # Resize image with keeping aspect ratio:
    resize_value = int(math.ceil(float(args.input_size) / args.resize_inv_factor))
    h, w = image.shape[:2]
    if not ((w == resize_value and w <= h) or (h == resize_value and h <= w)):
        resize_size = (resize_value, int(resize_value * h / w)) if w < h else (int(resize_value * w / h), resize_value)
        image = cv2.resize(image, dsize=resize_size, interpolation=cv2.INTER_LINEAR)

    # Center crop of the image:
    h, w = image.shape[:2]
    th, tw = args.input_size, args.input_size
    ih = int(round(0.5 * (h - th)))
    jw = int(round(0.5 * (w - tw)))
    image = image[ih:(ih + th), jw:(jw + tw), :]
    # cv2.imshow("image2", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Convert image to a float tensor and normalize it:
    x = image.astype(np.float32)
    x = x / 255.0
    x = (x - np.array(args.mean_rgb)) / np.array(args.std_rgb)

    # Create `use_cuda` flag:
    use_cuda = (args.num_gpus > 0)

    # Convert the tensor to a Pytorch tensor:
    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0)
    x = torch.FloatTensor(x)
    if use_cuda:
        x = x.cuda()

    # Create model with loading pretrained weights:
    net = ptcv_get_model(args.model, pretrained=True)
    net.eval()
    if use_cuda:
        net = net.cuda()

    # Evaluate the network:
    y = net(x)
    probs = torch.nn.Softmax(dim=-1)(y)

    # Show results:
    top_k = 5
    probs_np = probs.cpu().detach().numpy().squeeze(axis=0)
    top_k_inds = probs_np.argsort()[::-1][:top_k]
    classes = ImageNet1kAttr().classes
    print("The input picture is classified to be:")
    for k in range(top_k):
        print("{idx}: [{class_name}], with probability {prob:.3f}.".format(
            idx=(k + 1),
            class_name=classes[top_k_inds[k]],
            prob=probs_np[top_k_inds[k]]))


if __name__ == "__main__":
    main()
