#!/u/dphilipov/miniforge3/envs/Tur/bin/python
#SBATCH --chdir=/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp2: sft-eval"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp2_sft_eval.out"

#SBATCH --partition="gpuA40x4"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 24:00:00

import sys
sys.path.append("/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator")

from src.model.llms2 import LoraLM

dataset_name = "no-move_0-shot_100pc-obs"
model_name = ""
adapter_name = "llm_models/Meta-Llama-3.1-8B-Instruct-bnb-4bit/no-move_0-shot_100pc-obs"

model = LoraLM(model_name)
model.fetch_dataset(dataset_name)

model.answer_dataset(f"experiment_results/exp2/sft.jsonl", dataset_name, split="test")
