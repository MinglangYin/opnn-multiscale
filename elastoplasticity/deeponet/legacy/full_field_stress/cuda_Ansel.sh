#!/bin/bash

#SBATCH -t 24:00:00
#SBATCH --mem=12g
#SBATCH -A gk-condo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J explore_distribution
#SBATCH -o output-%j.out
#SBATCH -e error-%j.out
#SBATCH --constraint=titanrtx

module load cuda/9.1.85.1 intel/2017.0 mpi/openmpi_4.0.5_intel_2020.2_cuda_9.1.85_slurm20

OMP_NUM_THREADS=1 mpirun ./lmp_meso -in Reference.in
