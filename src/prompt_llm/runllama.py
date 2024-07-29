from src.model.llms import HugLM

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

llama3 = HugLM(model_name)

llama3.data.load("0_no_move")
llama3.save_answers("0_no_move", "test", "llm_prompt_sessions/llama_no-train/0_no_move.json")

# llama3.data.load("0_no_move")
# llama3.save_answers("0_no_move", "test", "llm_prompt_sessions/llama_no-train/0_no_move.json")
