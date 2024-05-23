#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/projects/bckf/dphilipov/teach-recreate/ python src/prompt_llm/runllama.py
