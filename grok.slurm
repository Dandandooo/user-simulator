#!/bin/bash

#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="grok"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/grok.out"

#SBATCH --partition="gpuA100x4"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 48:00:00

PYTHONPATH=$PROJECT python src/grokking_bert/train.py
