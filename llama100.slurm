#!/bin/bash

#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="teach-recreate-train"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/llama100.out"

#SBATCH --partition="gpuA100x4"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 48:00:00

# Run training script
/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/llama.bash
