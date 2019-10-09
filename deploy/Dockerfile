FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="osemery@gmail.com"

RUN apt update
RUN apt install -y python3-pip
RUN apt install -y ipython3 git htop mc wget
RUN apt install -y libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade mxnet-cu100
RUN pip3 install --upgrade torch torchvision
RUN pip3 install --upgrade chainer cupy-cuda100 chainercv
#RUN pip3 install --upgrade keras-mxnet
RUN pip3 install --upgrade tensorflow-gpu tensorpack
RUN pip3 install --upgrade keras
RUN pip3 install --upgrade pandas Pillow tqdm opencv-python
#RUN pip3 install --upgrade gluoncv2 pytorchcv

ADD bootstrap_eval.sh /root/
RUN chmod ugo+x /root/bootstrap_eval.sh
CMD /root/bootstrap_eval.sh