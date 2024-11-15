#!/bin/bash
## Do not put any commands or blank lines before the #SBATCH lines
#
#SBATCH --nodes=2                     # Request 2 nodes
#SBATCH --gres=gpu:1                  # Request one GPU per node
#SBATCH --ntasks-per-node=1           # Only one task per node (one experiment per node)
#SBATCH --job-name=GFLOWNET_ILU
#SBATCH --time=99:00:00               # Wall clock time
#SBATCH --partition=compute-gpu       # Use GPU partition
#SBATCH --output=slurm-%A_%N.out      # Separate output file per node
#SBATCH --error=slurm-%A_%N.err       # Separate error file per node
#SBATCH --mail-type=ALL               # Receive email notifications

###################################
# Load modules and environment
module purge
module load gcc/8.2.0
module load slurm/20.11.9
module load cuda11.7/toolkit/11.7.1
module load cuda11.7/blas/11.7.1
module load cuda11.7/fft/11.7.1
module load spack19/python/3.10.8

source /home/al1087536/gflownet_ilu/venv/bin/activate

###################################
# Define all hyperparameter combinations
###################################

learning_rates=(2e-4 7e-5 2e-5)
number_epochs=(50 100)
no_sampling_batches=(2)
schedule_patiences=(5 10)

# Create an array of all combinations
combinations=()
for lr in "${learning_rates[@]}"; do
  for epoch in "${number_epochs[@]}"; do
    for sampling_batch in "${no_sampling_batches[@]}"; do
      for patience in "${schedule_patiences[@]}"; do
        combinations+=("$lr $epoch $sampling_batch $patience")
      done
    done
  done
done

# Determine which subset of combinations to run on each node
total_combinations=${#combinations[@]}
half_combinations=$((total_combinations / 2))

if [ "$SLURM_NODEID" -eq 0 ]; then
  start_index=0
  end_index=$((half_combinations - 1))
else
  start_index=$half_combinations
  end_index=$((total_combinations - 1))
fi

# Run each combination sequentially on the assigned node
for i in $(seq $start_index $end_index); do
  read lr epoch sampling_batch patience <<< "${combinations[$i]}"
  echo "Running experiment with lr=$lr, epochs=$epoch, sampling_batch=$sampling_batch, patience=$patience"
  # Each srun command will wait for the previous one to finish due to the for loop
  srun --exclusive python train.py --lr $lr --epochs $epoch --no_sampling_batch $sampling_batch --patience $patience
done
