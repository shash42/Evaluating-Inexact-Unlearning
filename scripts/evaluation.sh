#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH -x gnode58,gnode17,gnode67
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --mail-user=shashwat.goel@research.iiit.ac.in
#SBATCH --mail-type=ALL

#[TODO]: 
#1. CHECK TRAIN/UNLEARN PATHS POINT TO RIGHT CHECKPOINTS WHEN EVALUATING!!
#2. CHECK SEED/LOGS MATCH REQUIREMENT
#3. CHECK NO-OGRE 

cat scripts/evaluation.sh
#ARGS
logdir='logs/Final-cifar100-resnet110'
dataset='cifar100'
model='resnet110'
num_classes=100
minlr_og=5e-3
maxlr_og=0.1

confA=47
confB=52
num_change=200
epochs_og=126
expname="Conf-C$confA-C$confB-$num_change[$minlr_og, $maxlr_og]_Batch64_${epochs_og}eps"

results="$logdir/$expname/FinetuneFinal_L[10, 90, 20]_62ep"
path_cust="$results/$1"
logname="eval-$1"

pathsout_tr="$logdir/$expname/train-paths.txt"
path_oarg=$(sed -n 1p "$pathsout_tr")
path_rarg=$(sed -n 2p "$pathsout_tr")
path_o=$(sed -n 3p "$pathsout_tr")
path_r=$(sed -n 4p "$pathsout_tr")

chmod +x src/evaluation.py
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs)\
 python3 src/evaluation.py\
 --path-o="$path_o"\
 --path-r="$path_r"\
 --path-oarg="$path_oarg"\
 --path-rarg="$path_rarg"\
 --num-classes=$num_classes --exch-classes $confA $confB\
 --custom --path-cust="$path_cust" --result-folder-cust="$results" --logname-cust="$logname"