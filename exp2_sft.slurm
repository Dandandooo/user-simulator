#!/u/dphilipov/miniforge3/envs/Tur/bin/python
#SBATCH --chdir=/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp2: sft"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp2_sft.out"

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

model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
# adapter_name = "llm_models/Meta-Llama-3.1-8B-Instruct-bnb-4bit/no-move_0-shot_100pc-obs"
dataset = "no-move_0-shot_100pc-obs"

model = LoraLM(model_name, dataset_name=dataset)

wandb.init(project="UserSimulator")

model.fine_tune(epochs=1)
