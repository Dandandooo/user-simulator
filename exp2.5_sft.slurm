#!/u/dphilipov/miniforge3/envs/Tur/bin/python
#SBATCH --chdir=/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp2.5: sft"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp2.5_sft.out"

#SBATCH --partition="gpuA100x4"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 36:00:00

import sys
sys.path.append('/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator')

import wandb
from src.model.llms2 import LoraLM

wandb.init(project="UserSimulator")

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
dataset = "sel-move_0-shot_100pc-obs"

model = LoraLM(model_name, dataset_name=dataset)

model.fine_tune(epochs=1)
