#!/bin/bash
#SBATCH --time=00-03:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=a100:4      # Number of GPUs per node to request
#SBATCH --tasks-per-node=4          # Number of processes to spawn per node
#SBATCH --cpus-per-task=12          # Number of CPUs per GPU
#SBATCH --mem=498G                  # Memory per node
#SBATCH --output=../logs/%x_%A-%a_%n-%t.out
#SBATCH --job-name=ohp_rn50
#SBATCH --account=def-ttt			# Use default account

GPUS_PER_NODE=4

ENC_PTH=$1
NAME=$2
SEED=${3:-0}
CSV=${4:-"/lustre06/project/6012565/isaacxu/benthicnet_probes/data_csv/one_hots/substrate_depth_2_data/substrate_depth_2_data.csv"}

# Exit if any command hits an error
set -e

# Set and activate the virtual environment
ENVNAME=pl_env
source ~/venvs/pl_env/bin/activate

# Multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(hostname -s)  # Store the master node’s IP address in the MASTER_ADDR environment variable.
export MAIN_HOST="$MASTER_ADDR"

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# Get the address of an open socket
source "../slurm/get_socket.sh"

# Copy and extract data over to the node
source "../slurm/copy_and_extract_data.sh"

srun python ../main_one_hot.py \
    --train_cfg "../cfgs/cnn/resnet50_ohp.json" \
    --csv "$CSV" \
    --nodes "$SLURM_JOB_NUM_NODES" \
    --gpus "$GPUS_PER_NODE" \
    --enc_pth "$ENC_PTH" \
    --name "$NAME" \
    --seed "$SEED" \
