#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH -x gnode58,gnode17
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --mail-user=shashwat.goel@research.iiit.ac.in
#SBATCH --mail-type=ALL

# WAIT AT BOTTOM 
# IS EVALUATION OF ORIGINAL/UNLEARNT NECC.? IF NOT, DONT DO IT!

# chmod +x scripts/cifar10-exch.sh
# chmod +x scripts/cifar100-exch.sh
# bash scripts/cifar10-exch.sh 3 5 2000 &
# bash scripts/cifar100-exch.sh 47 52 200 &

# chmod +x scripts/cifar10-CRT.sh
# chmod +x scripts/cifar100-CRT.sh
chmod +x scripts/cifar10-noise.sh
chmod +x scripts/cifar100-noise.sh
# bash scripts/cifar10-CRT.sh -1 4000 &
# bash scripts/cifar100-CRT.sh -1 400 &
bash scripts/cifar10-noise.sh 400 &
bash scripts/cifar100-noise.sh 4 &

# chmod +x scripts/cifar10-exch.sh
# chmod +x scripts/cifar100-exch.sh
# bash scripts/cifar10-exch.sh 0 4 200 &
# bash scripts/cifar10-exch.sh 3 5 2000 &
# bash scripts/cifar100-exch.sh 0 1 40 &
# bash scripts/cifar100-exch.sh 47 52 200 &

# chmod +x scripts/cifar10-noise.sh
# bash scripts/cifar10-noise.sh 2000 &
# bash scripts/cifar10-noise.sh 800 &
# bash scripts/cifar10-noise.sh 200 &
# bash scripts/cifar10-noise.sh 80 &

# chmod +x scripts/cifar100-noise.sh
# bash scripts/cifar100-noise.sh 200 &
# bash scripts/cifar100-noise.sh 80 &
# bash scripts/cifar100-noise.sh 20 &
# bash scripts/cifar100-noise.sh 8 &

# chmod +x scripts/cifar10-exch.sh
# bash scripts/cifar10-exch.sh 0 4 2000 &
# bash scripts/cifar10-exch.sh 1 9 2000 &
# bash scripts/cifar10-exch.sh 2 8 2000 &
# bash scripts/cifar10-exch.sh 6 7 2000 &
# bash scripts/cifar10-exch.sh 3 5 2000 &

# chmod +x scripts/cifar10-exch.sh
# bash scripts/cifar10-exch.sh 0 4 80 &
# bash scripts/cifar10-exch.sh 0 4 800 &
# bash scripts/cifar10-exch.sh 3 5 80 &
# bash scripts/cifar10-exch.sh 3 5 800 &

# chmod +x scripts/cifar100-exch.sh
# bash scripts/cifar100-exch.sh 0 1 20 &
# bash scripts/cifar100-exch.sh 0 1 80 &
# bash scripts/cifar100-exch.sh 0 1 200 &
# bash scripts/cifar100-exch.sh 47 52 20 &
# bash scripts/cifar100-exch.sh 47 52 40 &
# bash scripts/cifar100-exch.sh 47 52 80 &


# chmod +x scripts/cifar100-exch.sh
# chmod +x scripts/cifar100-noise.sh
# bash scripts/cifar100-exch.sh 0 1 200 &
# bash scripts/cifar100-exch.sh 47 52 20 &
# bash scripts/cifar100-exch.sh 0 1 20 &
# bash scripts/cifar100-noise.sh 200 &
# bash scripts/cifar100-noise.sh 20 &

wait