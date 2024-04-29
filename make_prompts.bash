#!/usr/bin/env bash

MAX_PERCENT_OBSERVE=35
NUM_PROMPTS=32

python src/prompt_llm/user_sim.py --give_task --include_das --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/da/action/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
python src/prompt_llm/user_sim.py --give_task --include_das --predict_das --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/da/predict/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
python src/prompt_llm/user_sim.py --give_task --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/no_da/action/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
python src/prompt_llm/user_sim.py --give_task --predict_das --num_prompts=$NUM_PROMPTS --save_path=./llm_prompts_data/turns/no_da/predict/generated.txt --save_answer --max_percent_observe=$MAX_PERCENT_OBSERVE
