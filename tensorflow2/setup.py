from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tf2cv',
    version='0.0.15',
    description='Image classification models for TensorFlow 2.0',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/osmr/imgclsmob',
    author='Oleg Sémery',
    author_email='osemery@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='machine-learning deep-learning neuralnetwork image-classification tensorflow imagenet vgg resnet resnext '
             'senet densenet darknet squeezenet squeezenext shufflenet menet mobilenent igcv3 mnasnet',
    packages=find_packages(exclude=['datasets', 'metrics', 'others', '*.others', 'others.*', '*.others.*']),
    include_package_data=True,
    install_requires=['numpy', 'requests'],
)
