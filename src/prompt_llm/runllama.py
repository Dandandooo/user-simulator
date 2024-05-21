from src.model.causal import HugLM

llama3 = HugLM("meta-llama/Meta-Llama-3-8B-Instruct")

llama3.data.add("zero", "llm_prompts_data/turns/valid_no_move/")
llama3.data.add("five", "llm_prompts_data/turns/valid_5_no_move/")

llama3.save_answers("zero", "llm_prompt_sessions/llama_no-train/zero_no_move/")
llama3.save_answers("five", "llm_prompt_sessions/llama_no-train/five_no_move/")
