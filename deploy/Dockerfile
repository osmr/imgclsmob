FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
LABEL maintainer="osemery@gmail.com"

RUN apt update
RUN apt install -y python3-pip
RUN apt install -y ipython3 git htop mc wget
RUN apt install -y libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade mxnet-cu100 gluoncv2
RUN pip3 install --upgrade pandas Pillow tqdm opencv-python

ADD bootstrap_eval_gl.sh /root/
RUN chmod ugo+x /root/bootstrap_eval_gl.sh
CMD /root/bootstrap_eval_gl.sh