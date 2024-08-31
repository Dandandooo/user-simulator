from src.model.llms import HugLM

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

llama3 = HugLM(model_name)

dataset = "0_no_move"

llama3.data.load(dataset)
llama3.save_answers(dataset, "test", f"llm_prompt_sessions/llama31_no-train/{dataset}.json")
