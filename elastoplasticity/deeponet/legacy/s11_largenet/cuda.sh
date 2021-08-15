#!/bin/bash

# Request 1 CPU core
#SBATCH --mem=25G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gcondo
#SBATCH --gres=gpu:1

#SBATCH -t 80:30:00
#SBATCH -o MyCUDAJob-%j.out
###SBATCH --exclude=gpu1204
#SBATCH --nodelist=gpu2101

# Compile CUDA program and run
#/users/myin2/Environments/pytorch_cuda11/bin

module load python/3.7.4 cuda/11.1.1 cudnn/8.1.0
#~/Environments/pytorch_cuda11/bin/python main.py --epochs 50000
source ~/Environments/pytorch_cuda11/bin/activate

python3 -u main.py --epochs 20000 #--batch-size 3000000
