from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig

"""
This script is designed for trl < 9.0.0
"""

# from accelerate import PartialState
# device_string = PartialState().process_index

model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
# model_name = "meta-llama/Meta-Llama-3-8b-Instruct"

# Promising alternative?
# model_name = "unsloth/Qwen2-7B-Instruct-bnb-4bit"

# TODO: omit most of the observes

print("Loading dataset")
dataset = "5_no_move"
data = load_dataset("Dandandooo/user-sim", dataset)

print("Initializing Model")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir=".cache", 
    use_cache=True, 
    force_download=False, 
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,)
)

print("Initializing Trainer")
save_model = f"Dandandooo/user-sim__llama-3-8b-it__{dataset}"
save_dir = f"llm_models/user-sim__llama-3-8b-it__{dataset}"

args = TrainingArguments(
    output_dir=save_dir,
    resume_from_checkpoint=save_dir, # can do save_model, but I want to resume locally
    # torch_compile=True,
    push_to_hub=True,
    hub_model_id=save_model,
    per_device_train_batch_size=2,  # Hopefully this won't overflow the memory
    gradient_accumulation_steps=3,
    bf16=True,
    save_steps=5000,
)


def format_func(data):
    return [f"### Instruction: {prompt}\n ### Response: {answer}" for prompt, answer in zip(data["prompt"], data["answer"])]


lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    lora_dropout=0.1,
    bias='none',
    task_type="CAUSAL_LM",
    use_rslora=True,  # Huggingface said "shown to work better"
)


trainer = SFTTrainer(
    model,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    peft_config=lora_config,
    formatting_func=format_func,
    tokenizer=tokenizer,
    args=args,
)
print("Trainer initialized!")


if __name__ == "__main__":
    trainer.train()
