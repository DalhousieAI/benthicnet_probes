#!/bin/bash
#SBATCH --time=00-01:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=a100:4      # Number of GPUs per node to request
#SBATCH --tasks-per-node=4          # Number of processes to spawn per node
#SBATCH --cpus-per-task=12          # Number of CPUs per GPU
#SBATCH --mem=498G                  # Memory per node
#SBATCH --output=../logs/%x_%A-%a_%n-%t.out
#SBATCH --job-name=rn50_training
#SBATCH --account=      			      # Use default account

GPUS_PER_NODE=4

# Default values
ONE_HOT="False"
SEED=0
CSV="../data_csv/benthicnet_nn.csv"
FINE_TUNE="False"
TEST_MODE="False"
FROM_SCRATCH="False"
FROM_SCRATCH_400="False"

# Parse options
while getopts ":o:e:n:s:c:f:t:g:i:j:" opt; do
  case $opt in
    o) ONE_HOT="$OPTARG" ;;
    e) ENC_PTH="$OPTARG" ;;
    n) NAME="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    c) CSV="$OPTARG" ;;
    f) FINE_TUNE="$OPTARG" ;;
    t) TEST_MODE="$OPTARG" ;;
    g) GPUS_PER_NODE="$OPTARG" ;;
    i) FROM_SCRATCH="$OPTARG" ;;
    j) FROM_SCRATCH_400="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Default is hierarchical learning
PYTHON_SCRIPT="../main.py"
TRAIN_CFG="../cfgs/cnn/resnet50_hp.json"
if [ "$FINE_TUNE" = "True" ]; then
  TRAIN_CFG="../cfgs/cnn/resnet50_hft.json"
fi
# From scratch scenario, use different config
if [ "$FROM_SCRATCH" = "True" ]; then
  TRAIN_CFG="../cfgs/cnn/resnet50_hl.json"
fi
# From scratch 400 scenario, use different config
if [ "$FROM_SCRATCH_400" = "True" ]; then
  TRAIN_CFG="../cfgs/cnn/resnet50_hl_400e.json"
fi

# One-hot scenario, use different script and configs
if [ "$ONE_HOT" = "True" ]; then
  PYTHON_SCRIPT="../main_one_hot.py"
  TRAIN_CFG="../cfgs/cnn/resnet50_ohp.json"
  if [ "$CSV" = "../data_csv/benthicnet_nn.csv" ]; then
    CSV="../data_csv/one_hots/substrate_depth_2_data/substrate_depth_2_data.csv"
  fi
  if [ "$FINE_TUNE" = "True" ]; then
    TRAIN_CFG="../cfgs/cnn/resnet50_ohft.json"
  fi
    # From scratch scenario, use different config
  if [ "$FROM_SCRATCH" = "True" ]; then
    TRAIN_CFG="../cfgs/cnn/resnet50_ohl.json"
  fi
  # From scratch 400 scenario, use different config
  if [ "$FROM_SCRATCH_400" = "True" ]; then
    TRAIN_CFG="../cfgs/cnn/resnet50_ohl_400e.json"
  fi
fi

# Exit if any command hits an error
set -e

# Set and activate the virtual environment
ENVNAME=pl_env
source ~/venvs/pl_env/bin/activate

# Multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname -s)
export MAIN_HOST="$MASTER_ADDR"

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# Get the address of an open socket
source "../slurm/get_socket.sh"

# Copy and extract data over to the node
source "../slurm/copy_and_extract_data.sh"

# Build and execute Python command
cmd="srun python $PYTHON_SCRIPT"
cmd+=" --train_cfg \"$TRAIN_CFG\""
cmd+=" --csv \"$CSV\""
cmd+=" --nodes \"$SLURM_JOB_NUM_NODES\""
cmd+=" --gpus \"$GPUS_PER_NODE\""
cmd+=" --name \"$NAME\""
cmd+=" --seed \"$SEED\""

if [ "$FINE_TUNE" = "True" ]; then
  cmd+=" --fine_tune \"$FINE_TUNE\""
fi

if [ "$TEST_MODE" = "True" ]; then
  cmd+=" --test_mode \"$TEST_MODE\""
fi

if [ -n "$ENC_PTH" ]; then
  cmd+=" --enc_pth \"$ENC_PTH\""
fi

echo "Submitting command: $cmd"

eval $cmd
