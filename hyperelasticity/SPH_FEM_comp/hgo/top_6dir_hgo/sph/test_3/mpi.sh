#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=3:20:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH -N 4
###SBATCH --tasks-per-node=8

# Specify a job name:
#SBATCH -J MyMPIJob

# Specify an output file
#SBATCH -o MyMPIJob-%j.out
#SBATCH -e MyMPIJob-%j.out

# Run a command
module load gcc/6.3 mpi/openmpi_4.0.1_gcc eigen/3.3.2
srun --mpi=pmix -n 4 ./lmp_mpi -in in.hgo3d_uniaxial

