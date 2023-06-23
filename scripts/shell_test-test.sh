#!/bin/bash
#SBATCH --time=00-01:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=a100:1      # Number of GPUs per node to request
#SBATCH --tasks-per-node=1          # Number of processes to spawn per node
#SBATCH --cpus-per-task=12          # Number of CPUs per GPU
#SBATCH --mem=256G                  # Memory per node
#SBATCH --output=../logs/%x_%A-%a_%n-%t.out
#SBATCH --job-name=hp_test_test
#SBATCH --account=def-ttt			# Use default account

GPUS_PER_NODE=1

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

srun python ../main.py \
    --train_cfg "../cfgs/cnn/resnet50_hp_1024_test.json" \
    --enc_pth "../pretrained_encoders/100K_benthicnet_resnet50_checkpoint_epoch=99-val_loss=0.1433.ckpt" \
    --csv "../data_csv/size_10K_benthicnet.csv" \
    --nodes "$SLURM_JOB_NUM_NODES" \
    --gpus "$GPUS_PER_NODE" \
    --test_mode true \
    --name "hp_test_test"
