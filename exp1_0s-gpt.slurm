#!/u/dphilipov/miniforge/envs/Tur/bin/python
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp1: 0s gpt"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp1_0s-gpt.out"

#SBATCH --partition="gpuA100x4"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 36:00:00

import os
from src.model.llms2 import AzureLM

dataset_name = "0-shot_100pc-obs"
model_name = "gpt-4o-mini"
endpoint = "https://uiuc-convai.openai.azure.com/"
api_key = os.getenv("AZURE_OPENAI_KEY_4o")

client = AzureLM(endpoint, api_key, model_name)
client.fetch_dataset(dataset_name)

client.answer_dataset(f"experiment_results/exp1/0s-gpt.jsonl", dataset_name, split="test")
