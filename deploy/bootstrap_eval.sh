#!/bin/sh

git clone https://github.com/osmr/imgclsmob.git

mkdir imgclsmob_data
cd imgclsmob_data

#mkdir imagenet_rec
#cd imagenet_rec
#wget http://soleka.sadmin.ru/SOLEKA/val.idx
#wget http://soleka.sadmin.ru/SOLEKA/val.rec

mkdir imagenet
cd imagenet
wget http://soleka.sadmin.ru/SOLEKA/imagenet_val.zip
unzip imagenet_val.zip
rm imagenet_val.zip

cd ../../imgclsmob

#python3 eval_gl.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --use-pretrained --calc-flops
python3 eval_gl.py --dataset=ImageNet1K --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --use-pretrained --calc-flops
python3 eval_pt.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --use-pretrained --calc-flops
python3 eval_ch.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --use-pretrained
#python3 eval_ke.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --use-pretrained
python3 eval_tf.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --use-pretrained

cp ~/.mxnet/models/resnet18-0951-98a2545b.params ~/imgclsmob_data/resnet18/
python3 convert_models.py --src-fwk=gluon --dst-fwk=pytorch --src-model=resnet18 --dst-model=resnet18 --src-params=../imgclsmob_data/resnet18/resnet18-0951-98a2545b.params --dst-params=../imgclsmob_data/resnet18/resnet18.pth --save-dir=../imgclsmob_data/resnet18/
python3 eval_pt.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --resume=../imgclsmob_data/resnet18/resnet18.pth --calc-flops
python3 convert_models.py --src-fwk=gluon --dst-fwk=chainer --src-model=resnet18 --dst-model=resnet18 --src-params=../imgclsmob_data/resnet18/resnet18-0951-98a2545b.params --dst-params=../imgclsmob_data/resnet18/resnet18.npz --save-dir=../imgclsmob_data/resnet18/
python3 eval_ch.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --resume=../imgclsmob_data/resnet18/resnet18.npz
#python3 convert_models.py --src-fwk=gluon --dst-fwk=keras --src-model=resnet18 --dst-model=resnet18 --src-params=../imgclsmob_data/resnet18/resnet18-0951-98a2545b.params --dst-params=../imgclsmob_data/resnet18/resnet18.h5 --save-dir=../imgclsmob_data/resnet18/
#python3 eval_ke.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --resume=../imgclsmob_data/resnet18/resnet18.h5
python3 convert_models.py --src-fwk=gluon --dst-fwk=tensorflow --src-model=resnet18 --dst-model=resnet18 --src-params=../imgclsmob_data/resnet18/resnet18-0951-98a2545b.params --dst-params=../imgclsmob_data/resnet18/resnet18.tf.npz --save-dir=../imgclsmob_data/resnet18/
python3 eval_tf.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --resume=../imgclsmob_data/resnet18/resnet18.tf.npz