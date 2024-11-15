#!/bin/bash
#SBATCH --nodes=2                     # Allocate 2 nodes
#SBATCH --gres=gpu:1                  # Request one GPU per node
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=GFLOWNET_ILU
#SBATCH --time=99:00:00
#SBATCH --array=0-1                   # One job array per node
#SBATCH --partition=compute-gpu
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

module purge
module load gcc/8.2.0
module load slurm/20.11.9
module load cuda11.7/toolkit/11.7.1
module load cuda11.7/blas/11.7.1
module load cuda11.7/fft/11.7.1
module load spack19/python/3.10.8

source /home/al1087536/gflownet_ilu/venv/bin/activate

# Define hyperparameter arrays
learning_rates=(2e-5 2e-6)
number_epochs=(50)
no_sampling_batches=(1)
schedule_patience=(5)

# Generate all hyperparameter combinations
combinations=()
for lr in "${learning_rates[@]}"; do
    for epochs in "${number_epochs[@]}"; do
        for batch in "${no_sampling_batches[@]}"; do
            for patience in "${schedule_patience[@]}"; do
                combinations+=("$lr $epochs $batch $patience")
            done
        done
    done
done

# Split combinations between nodes based on array task
num_combinations=${#combinations[@]}
combinations_per_node=$(( (num_combinations + 1) / 2 ))

# Determine the start and end indices for this node's combinations
start_index=$(( SLURM_ARRAY_TASK_ID * combinations_per_node ))
end_index=$(( start_index + combinations_per_node - 1 ))
if (( end_index >= num_combinations )); then
    end_index=$(( num_combinations - 1 ))
fi

# Run each combination sequentially on this node
for (( i=start_index; i<=end_index; i++ )); do
    # Extract hyperparameters for the current combination
    combination=(${combinations[i]})
    lr=${combination[0]}
    epochs=${combination[1]}
    no_sampling_batch=${combination[2]}
    patience=${combination[3]}

    # Run the Python script with the selected hyperparameters
    srun python train.py --lr "$lr" --epochs "$epochs" --no_sampling_batch "$no_sampling_batch" --patience "$patience"
done
