#!/bin/bash
#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH -x gnode58,gnode17
#SBATCH --mem=40G
#SBATCH --mail-user=shashwat.goel@research.iiit.ac.in
#SBATCH --mail-type=ALL

chmod +x src/learn.py
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)\
python3 src/learn.py\
 --log-dir='logs/cifar100-resnet20' --exp-name="126ep_[5e-3_0.1]" --procedure='pretrain'\
 --epochs=126 --minlr=5e-3 --maxlr=0.1\
 --dataset='cifar100' --model='resnet20' --num-classes=100 --validsplit=0