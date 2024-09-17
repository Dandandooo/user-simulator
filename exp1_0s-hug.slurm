#!/u/dphilipov/miniforge3/envs/Tur/bin/python
#SBATCH --chdir=/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp1: 0s hug"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp1_0s-hug.out"

#SBATCH --partition="gpuA40x4"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 24:00:00

import sys
sys.path.append("/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator")

from src.model.llms2 import HugLM

dataset_name = "0-shot_100pc-obs"
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

model = HugLM(model_name)
model.fetch_dataset(dataset_name)

model.answer_dataset(f"experiment_results/exp1/0s-hug.jsonl", dataset_name, split="test")
