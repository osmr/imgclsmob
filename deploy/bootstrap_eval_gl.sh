#!/bin/sh

git clone https://github.com/osmr/imgclsmob.git

mkdir imgclsmob_data
cd imgclsmob_data
mkdir imagenet_rec
cd imagenet_rec
wget http://soleka.sadmin.ru/SOLEKA/val.idx
wget http://soleka.sadmin.ru/SOLEKA/val.rec


cd ../../imgclsmob
python3 eval_gl.py --num-gpus=1 --model=resnet18 --batch-size=100 -j=4 --save-dir=../imgclsmob_data/resnet18/ --use-pretrained --calc-flops