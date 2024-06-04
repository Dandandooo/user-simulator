from src.model.llms import HugLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "unsloth/llama-3-8b-bnb-4bit"

llama3 = HugLM(model_name)

llama3.data.add("zero", "/projects/bckf/dphilipov/teach-recreate/llm_prompts_data/turns/valid_no_move/")
llama3.data.add("five", "/projects/bckf/dphilipov/teach-recreate/llm_prompts_data/turns/valid_5_no_move/")

llama3.save_answers("five", "/projects/bckf/dphilipov/teach-recreate/llm_prompt_sessions/llama_no-train/five_no_move/")
llama3.save_answers("zero", "/projects/bckf/dphilipov/teach-recreate/llm_prompt_sessions/llama_no-train/zero_no_move/")
