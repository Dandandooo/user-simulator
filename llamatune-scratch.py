from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig

"""
This script is designed for trl < 9.0.0
"""

# from accelerate import PartialState
# device_string = PartialState().process_index

model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# Promising alternative?
# model_name = "unsloth/Qwen2-7B-Instruct-bnb-4bit"

print("Loading dataset")
data = load_dataset("Dandandooo/user-sim", "5_no_move")

print("Initializing Model")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=".cache", use_cache=True, force_download=False, device_map="auto")

print("Initializing Trainer")
save_model = "Dandandooo/user-sim__llama-3-8b-it"
save_dir = "llm_models/user-sim__llama-3-8b-it"

args = TrainingArguments(
    output_dir=save_dir,
    # resume_from_checkpoint=save_model,
    # torch_compile=True,
    push_to_hub=True,
    hub_model_id=save_model,
    per_device_train_batch_size=1,  # Hopefully this won't overflow the memory
    gradient_accumulation_steps=8,
    bf16=True,
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

trainer.train()
