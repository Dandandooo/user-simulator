#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/taiga/illinois/collab/eng/cs/conv-ai/UserSimulator python src/prompt_llm/runllama.py
