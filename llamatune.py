from src.model.llms import LoraLM

# model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_name = "unsloth/gemma-2b-it-bnb-4bit"

llm = LoraLM(model_name)

llm.train()
