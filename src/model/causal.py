from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model
from glob import glob
import os

# from src.data.dataclass import TeachData
from transformers import DataCollatorForLanguageModeling

import torch

class EvalLM:
    def __init__(self, model_name="google/gemma-2b"):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", truncation_side="left")

        self.data = None

    def set_data(self, folder: str, prompt_glob: str, answer_glob: str):
        prompts = glob(os.path.join(folder, prompt_glob))
        answers = glob(os.path.join(folder, answer_glob))
        self.data = list(zip(prompts, answers))

    def answer(self, prompt: str):
        tokenized = self.data.tokenizer(prompt, return_tensors="pt")
        self.model

    def load_data(self, folder: str, prompt_glob: str, answer_glob: str):
        prompts = glob(os.path.join(folder, prompt_glob))
        answers = glob(os.path.join(folder, answer_glob))
        paths = list(zip(prompts, answers))
        self.data = [(open(prompt).read(), open(answer).read()) for prompt, answer in paths]

    def eval(self, input_ids):
        if self.data is None:
            raise ValueError("Data not set")
        pass



class LoraLM(EvalLM):
    def __init__(self, model_name="google/gemma-2b"):
        #TODO change model to use the bitsandbytes integration (CUDA ONLY)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            # init_lora_weights="gaussian",
            bias='none',
            task_type="CAUSAL_LM",
        )

        self.peft_model = get_peft_model(self.model, self.lora_config)

        # self.trainer = Trainer(
        #     model=self.peft_model,
        #     args=TrainingArguments(
        #         output_dir="results",
        #         overwrite_output_dir=True,
        #         num_train_epochs=1,
        #         per_device_train_batch_size=1,
        #         save_steps=1,
        #         save_total_limit=2,
        #     ),
        #     data_collator=DataCollatorForLanguageModeling(tokenizer=self.data.tokenizer, mlm=False),
        # )
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

if __name__ == "__main__":
    model = LoraLM()
    model.peft_model.print_trainable_parameters()
