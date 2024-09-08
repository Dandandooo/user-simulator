#!/u/dphilipov/miniforge/envs/Tur/bin/python
#SBATCH --account="bckf-delta-gpu"
#SBATCH --job-name="exp6: sft"
#SBATCH --output="/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator/logs/exp6_sft.out"

#SBATCH --partition="gpuA100x4"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

#SBATCH --exclusive
#SBATCH -t 36:00:00
