srun --account=bckf-delta-gpu --partition=gpuA40x4-interactive --time=1:00:00 --mem=64G --gpus-per-node=1 \
	./train.bash
