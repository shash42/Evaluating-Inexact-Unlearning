#!/bin/bash

chmod +x src/train.py
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) \
python3 src/train.py --log-dir='logs/args-debug' --exp-name='Conf-C0-C1-4000_LR[5e-3, 0.1]_Batch64_62eps' \
--dataset='cifar10' --model='resnet20' --num-classes=10 \
--confname='C0-C1-10' --num-change=2000 --exch-classes 2 3 \