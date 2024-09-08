#!/u/dphilipov/miniforge/envs/Tur/bin/python
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp2: fs hug"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp2_fs-hug.out"

#SBATCH --partition="gpuA100x4"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 36:00:00

from src.model.llms2 import HugLM

dataset_name = "no-move_5-shot_100pc-obs"
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

model = HugLM(model_name)
model.fetch_dataset(dataset_name)

model.answer_dataset(f"experiment_results/exp2/fs-hug.jsonl", dataset_name, split="test")
