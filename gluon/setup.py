from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gluoncv2',
    version='0.0.10',
    description='Image classification models for Gluon',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/osmr/imgclsmob',
    author='osmr',
    author_email='osemery@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    keywords='machine-learning deep-learning neuralnetwork image-classification imagenet mxnet gluon',
    packages=find_packages(exclude=['others', '*.others', 'others.*', '*.others.*']),
    include_package_data=True,
    install_requires=['numpy'],
)
