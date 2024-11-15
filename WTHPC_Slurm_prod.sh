#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
#
#SBATCH --nodes=1                     # Number of nodes; all cores per node are allocated to the job
#SBATCH --gres=gpu:1                  # Request one GPU from the node
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=GFLOWNET_ILU
#SBATCH --time=200:00:00                # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 1 hour)
#SBATCH --partition=compute-gpu       # partition - use gpus
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#
#SBATCH --mail-type=ALL		      # Recieve email when your job starts and completes 
#
###################################
#
# Add shell commands (load modules, create directories, compile code, etc.)
#
module purge
module load gcc/8.2.0
module load slurm/20.11.9
module load cuda11.7/toolkit/11.7.1
module load cuda11.7/blas/11.7.1
module load cuda11.7/fft/11.7.1
module load spack19/python/3.10.8

source /home/al1087536/gflownet_ilu/venv/bin/activate

#
# Launch job
#
srun python train.py
