#!/usr/bin/env bash

MAX_PERCENT_OBSERVE=35
NUM_PROMPTS=32

$INTERPRETER src/prompt_llm/user_sim.py --include_das --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/da/action/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
$INTERPRETER src/prompt_llm/user_sim.py --include_das --predict_das --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/da/predict/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
$INTERPRETER src/prompt_llm/user_sim.py --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/no_da/action/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
$INTERPRETER src/prompt_llm/user_sim.py --predict_das --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/no_da/predict/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
