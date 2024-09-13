#!/bin/bash

#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="llama3 eval"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/llamatune-scratch.out"

#SBATCH --partition="gpuA100x4"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 24:00:00
python llamatune-scratch.py