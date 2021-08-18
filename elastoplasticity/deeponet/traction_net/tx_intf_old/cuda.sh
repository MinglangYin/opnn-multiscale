#!/bin/bash

# Request 1 CPU core
#SBATCH --mem=12G
#SBATCH -A gk-condo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH -t 6:00:00
#SBATCH -o MyCUDAJob-%j.out
###SBATCH --exclude=gpu1204

# Compile CUDA program and run
~/Environments/PyTorch-GPU/bin/python main.py --epochs 50000
