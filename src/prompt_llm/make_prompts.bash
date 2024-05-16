#!/usr/bin/env bash

# To limit the unfair weighting of observes
MAX_PERCENT_OBSERVE=35

# How many random prompts to make for the llm
NUM_PROMPTS=32

# Number of examples to give the llm for each prompt
NUM_EXAMPLES=5

# This will make it random
EXAMPLE_LENGTH=None

# Omit observations from given examples, not the task. Comment out to disable
OMIT_OBS=--no_obs

# Split the data between train and test (valid_seen)
SPLIT_DATA=--split_dataset

# I'm using $INTERPRETER because my conda enviroonments broke and I don't know why
$INTERPRETER src/prompt_llm/user_sim.py --include_das               $OMIT_OBS $SPLIT_DATA --num_prompts=$NUM_PROMPTS --num=$NUM_EXAMPLES --length=$EXAMPLE_LENGTH --save_path=./llm_prompts_data/turns/da/action/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
$INTERPRETER src/prompt_llm/user_sim.py --include_das --predict_das $OMIT_OBS $SPLIT_DATA --num_prompts=$NUM_PROMPTS --num=$NUM_EXAMPLES --length=$EXAMPLE_LENGTH --save_path=./llm_prompts_data/turns/da/predict/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
$INTERPRETER src/prompt_llm/user_sim.py                             $OMIT_OBS $SPLIT_DATA --num_prompts=$NUM_PROMPTS --num=$NUM_EXAMPLES --length=$EXAMPLE_LENGTH --save_path=./llm_prompts_data/turns/no_da/action/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
$INTERPRETER src/prompt_llm/user_sim.py               --predict_das $OMIT_OBS $SPLIT_DATA --num_prompts=$NUM_PROMPTS --num=$NUM_EXAMPLES --length=$EXAMPLE_LENGTH --save_path=./llm_prompts_data/turns/no_da/predict/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
