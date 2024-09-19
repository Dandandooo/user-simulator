#!/u/dphilipov/miniforge3/envs/Tur/bin/python
#SBATCH --chdir=/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp3: sft-eval"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp3_sft_eval.out"

#SBATCH --partition="gpuA40x4"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 24:00:00

import sys
sys.path.append("/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator")

import wandb
from src.model.llms2 import LoraLM

dataset_name = "sel-move_0-shot_20pc-obs"
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

from glob import glob
adapter_name = "llm_models/Meta-Llama-3.1-8B-Instruct-bnb-4bit/sel-move_0-shot_20pc-obs"
adapter_name = sorted(glob(adapter_name + "/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))[-1]

wandb.init(project="UserSimulator")

model = LoraLM(model_name, adapter_name=adapter_name)
model.fetch_dataset(dataset_name)

model.answer_dataset(f"experiment_results/exp3/sft.jsonl", dataset_name, split="test")
