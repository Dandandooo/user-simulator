#!/bin/bash

#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="llama3 eval"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/llama.out"

#SBATCH --partition="gpuA40x4"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 36:00:00

# Run training script
/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/llama.bash
