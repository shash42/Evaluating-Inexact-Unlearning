#!/bin/bash

chmod +x src/unlearn.py
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)\
 python3 src/unlearn.py\
 --path-o='logs/args-debug/Conf-C0-C1-4000_LR[5e-3, 0.1]_Batch64_62eps/cifar10_resnet20_seed-1_Nf-4000_split-0.2_ep-62_bs-64_lr-[0_005-0_1]_wd-5e-05_original.pt'\
 --path-r='logs/args-debug/Conf-C0-C1-4000_LR[5e-3, 0.1]_Batch64_62eps/cifar10_resnet20_seed-1_Nf-4000_split-0.2_ep-62_bs-64_lr-[0_005-0_1]_wd-5e-05_retrain.pt'\
 --path-oarg='logs/args-debug/Conf-C0-C1-4000_LR[5e-3, 0.1]_Batch64_62eps/args_og.txt'\
 --path-rarg='logs/args-debug/Conf-C0-C1-4000_LR[5e-3, 0.1]_Batch64_62eps/args_re.txt'\
 --init-checkpoint='logs/args-debug/Conf-C0-C1-4000_LR[5e-3, 0.1]_Batch64_62eps/cifar10_resnet20_seed-1_Nf-4000_split-0.2_ep-62_bs-64_lr-[0_005-0_1]_wd-5e-05_init.pt'\
 --num-classes=10 --exch-classes 2 3\
 --golatkar=False --name-go='Golatkar'\
 --retrfinal=True --name-rf='RetrFinal_L[1, 7, 2]_62ep' --epochs-rf=62 --minL-rf=1 --maxL-rf=7 --stepL-rf=2\
 --finetune=True --name-ft='Finetune_LR[0.005, 1]_30ep' --epochs-ft=30 --minlr-ft=0.005 --maxlr-ft=0.1\