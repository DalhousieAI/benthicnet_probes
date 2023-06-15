#!/bin/bash
#SBATCH --time=00-12:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=v100l:4     # Number of GPUs per node to request
#SBATCH --tasks-per-node=4          # Number of processes to spawn per node
#SBATCH --cpus-per-task=8           # Number of CPUs per GPU
#SBATCH --mem=128G                  # Memory per node
#SBATCH --output=../logs/%x_%A-%a_%n-%t.out
#SBATCH --job-name=hl_full_resnet50
#SBATCH --account=def-ttt			# Use default account

GPUS_PER_NODE=4

# Exit if any command hits an error
set -e

#Store the time at which the script was launched
start_time="$SECONDS"

# Multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(hostname -s)  # Store the master node’s IP address in the MASTER_ADDR environment variable.
export MAIN_HOST="$MASTER_ADDR"

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# Get the address of an open socket
source "../slurm/get_socket.sh"

srun python ../main.py \
    --train_cfg "../cfgs/cnn/resnet50_hl.json" \
    --csv "../data_csv/size_full_benthicnet.csv" \
    --nodes "$SLURM_JOB_NUM_NODES" \
    --gpus "$GPUS_PER_NODE" \
    --name "hl_full_resnet50"
