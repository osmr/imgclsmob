#!/bin/sh

git clone https://github.com/osmr/imgclsmob.git

mkdir imgclsmob_data
cd imgclsmob_data
mkdir imagenet_rec
cd imagenet_rec
wget http://soleka.sadmin.ru/SOLEKA/val.idx
wget http://soleka.sadmin.ru/SOLEKA/val.rec
wget http://soleka.sadmin.ru/SOLEKA/train.idx
wget http://soleka.sadmin.ru/SOLEKA/train.rec


cd ../../imgclsmob
python3 train_gl.py --num-gpus=1 --model=resnet18 --save-dir=../imgclsmob_data/resnet18/ --batch-size=320 --batch-size-scale=4 -j=12 --num-epochs=200 --lr=0.5 --lr-mode=cosine --wd=0.0001 --warmup-epochs=5 --warmup-mode=cosine --mixup --label-smoothing --gamma-wd-mult=0.0001 --beta-wd-mult=0.0001 --bias-wd-mult=0.01 --attempt=1 --start-epoch=1
python3 train_gl.py --dataset=ImageNet1K --num-gpus=1 --model=resnet18 --save-dir=../imgclsmob_data/resnet18/ --batch-size=320 --batch-size-scale=4 -j=12 --num-epochs=200 --lr=0.5 --lr-mode=cosine --wd=0.0001 --warmup-epochs=5 --warmup-mode=cosine --mixup --label-smoothing --gamma-wd-mult=0.0001 --beta-wd-mult=0.0001 --bias-wd-mult=0.01 --attempt=1 --start-epoch=1