#!/bin/bash

#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="llama3 train"
#SBATCH --output="/projects/bckf/dphilipov/teach-recreate/logs/llama.out"

#SBATCH --partition="gpuA40x4"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 16:00:00

# Run training script
/projects/bckf/dphilipov/teach-recreate/llama.bash
