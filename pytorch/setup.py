from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pytorchcv',
    description='Image classification and segmentation models for PyTorch',
    url='https://github.com/osmr/imgclsmob',
    author='Oleg Sémery',
    author_email='osemery@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='machine-learning deep-learning neuralnetwork image-classification pytorch imagenet cifar svhn vgg resnet '
             'pyramidnet diracnet densenet condensenet wrn drn dpn darknet fishnet espnetv2 xdensnet squeezenet '
             'squeezenext shufflenet menet mobilenet igcv3 mnasnet darts xception inception polynet nasnet pnasnet ror '
             'proxylessnas dianet efficientnet mixnet image-segmentation voc ade20k cityscapes coco pspnet deeplabv3 '
             'fcn',
    packages=find_packages(exclude=['datasets', 'metrics', 'others', '*.others', 'others.*', '*.others.*']),
    install_requires=['numpy', 'requests'],
    python_requires='>=3.10',
    include_package_data=True,
)
