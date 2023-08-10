#!/bin/bash
#SBATCH --time=00-01:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --tasks-per-node=1          # Number of processes to spawn per node
#SBATCH --cpus-per-task=1           # Number of CPUs per GPU
#SBATCH --mem=36G                   # Memory per node
#SBATCH --output=../logs/%x_%A-%a_%n-%t.out
#SBATCH --job-name=prob_imgs
#SBATCH --account=def-ttt			# Use default account

# Exit if any command hits an error
set -e

# Set and activate the virtual environment
ENVNAME=pl_env
source ~/venvs/pl_env/bin/activate

srun python ../check_for_problem_imgs.py \
