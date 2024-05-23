from src.model.causal import HugLM

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "unsloth/llama-3-8b-bnb-4bit"

llama3 = HugLM(model_name)

llama3.data.add("zero", "llm_prompts_data/turns/valid_no_move/")
llama3.data.add("five", "llm_prompts_data/turns/valid_5_no_move/")

llama3.save_answers("five", "llm_prompt_sessions/llama_no-train/five_no_move/")
llama3.save_answers("zero", "llm_prompt_sessions/llama_no-train/zero_no_move/")
