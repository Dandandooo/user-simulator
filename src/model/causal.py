from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from openai import AzureOpenAI

from tqdm import tqdm
from functools import lru_cache
import os
import re
import json

# import bitsandbytes

# from src.data.dataclass import TeachData

import torch
import numpy as np


class LLMDataset:
    def __init__(self):
        self.data = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def add(self, dataset_name, directory, prompt_regex=r"turn_\d+\.txt", answer_regex=r"turn_\d+_answer\.txt"):
        prompt_re = re.compile(prompt_regex)
        answer_re = re.compile(answer_regex)

        if dataset_name not in self.data:
            self.data[dataset_name] = {}
        for folder in os.listdir(directory):
            prompts = filter(prompt_re.match, os.listdir(directory + folder))
            answers = filter(answer_re.match, os.listdir(directory + folder))
            self.data[dataset_name][folder] = list(zip(prompts, answers))


class BaseLM:
    def __init__(self):
        self.test_folder = "llm_prompts_data/turns/valid/"
        self.data = LLMDataset()
        self.data.add("test", self.test_folder)

    @lru_cache  # Memoizing to reduce cost
    def answer(self, prompt: str) -> str:
        raise NotImplementedError

    def answer_folder(self, folder: list[tuple[str, str]]) -> list[dict[str, str]]:
        for prompt, answer in folder:
            result = self.answer(prompt)
            yield {
                "prompt": prompt,
                "answer": answer,
                "response": result
            }

    def answer_dataset(self, dataset_name: str) -> dict:
        for file_id, folder in self.data[dataset_name].items():
            yield file_id, list(self.answer_folder(folder))

    def save_answers(self, dataset_name: str, dest_folder: str):
        for file_id, folder in self.answer_dataset(dataset_name):
            with open(f"{dest_folder}/{file_id}_result.json", "w") as f:
                json.dump(folder, f)

    def tp_etc(self, dataset_name: str) -> tuple[int, int, int, int, int, int]:
        true_observed = 0
        false_observed = 0
        true_speak = 0
        false_speak = 0
        correct_speak = 0
        incorrect_speak = 0

        # Returns a lazy generator
        data = self.answer_dataset(dataset_name)

        print("Getting metrics for dataset")
        for _, folder in tqdm(data, total=len(self.data[dataset_name])):
            for result in folder:
                answer, response = result["answer"], result["response"]
                if answer == "OBSERVE":
                    if response == "OBSERVE":
                        true_observed += 1
                    else:
                        false_speak += 1
                else:
                    if response == "OBSERVE":
                        false_observed += 1
                    else:
                        true_speak += 1
                        if response == answer:
                            correct_speak += 1
                        else:
                            incorrect_speak += 1

        return true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak

    def eval(self, dataset_name: str) -> dict:
        true_observed, false_observed, true_speak, false_speak, correct_speak, incorrect_speak = self.tp_etc(dataset_name)
        return {
            "accuracy": (true_observed + true_speak) / (true_observed + false_observed + true_speak + false_speak),
            "f-score": 2 * correct_speak / (2 * correct_speak + incorrect_speak),
            "conv_matrix": np.array([[true_observed, false_speak], [false_observed, true_speak]]),
            "da_accuracy": correct_speak / (correct_speak + incorrect_speak) if (incorrect_speak + correct_speak) != 0 else 0,
        }


class GPT4LM(BaseLM):
    def __init__(self, api_key=os.getenv("AZURE_OPENAI_KEY_4"),
                 azure_endpoint="https://uiuc-convai-sweden.openai.azure.com/", role="user"):
        super().__init__()
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-15-preview"
        )

        self.role = role

    @lru_cache
    def answer(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model="UIUC-ConvAI-Sweden-GPT4",  # model = "deployment_name"
            messages=[{"role": self.role, "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return completion.choices[0].message.content


# To work with huggingface models
class HugLM(BaseLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", load_in_8bit=True, **model_kwargs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=load_in_8bit,
            **model_kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @lru_cache
    def answer(self, prompt: str) -> str:
        tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        result = self.model.generate(tokenized["input_ids"], max_length=1000)
        decoded: str = self.tokenizer.decode(result[0], skip_special_tokens=True)
        return decoded


# To use the LoRA fine-tuning method for huggingface models
class LoraLM(HugLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it"):
        super().__init__(model_name, load_in_8bit=True, gradient_checkpointing=True, use_cache=True)

        self.model.enable_input_require_grads()

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=8,
            lora_dropout=0.1,
            bias='none',
            task_type="CAUSAL_LM",
        )

        self.peft_model = get_peft_model(self.model, self.lora_config)

        # TODO: write trainer for lora

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
