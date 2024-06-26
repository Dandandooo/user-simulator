#!/bin/bash

#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="tuning llm"
#SBATCH --output="/projects/bckf/dphilipov/teach-recreate/logs/llamatune.out"

#SBATCH --partition="gpuA40x4"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 36:00:00

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/projects/bckf/dphilipov/teach-recreate/ python llamatune.py