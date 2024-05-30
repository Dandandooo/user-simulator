from transformers import AutoModelForCausalLM, TFAutoModelForCausalLM, FlaxAutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from openai import AzureOpenAI
import ollama

from functools import cache

from tqdm import tqdm
import os
import re
import json

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

    def add(self, dataset_name, directory, prompt_regex=r".*turn_\d+\.txt", answer_regex=r".*turn_\d+_answer\.txt"):
        prompt_re = re.compile(prompt_regex)
        answer_re = re.compile(answer_regex)

        op = lambda e: open(e).read()

        if dataset_name not in self.data:
            self.data[dataset_name] = {}
        for folder in os.listdir(directory):
            folder_files = os.listdir(os.path.join(directory, folder))
            folder_files = [os.path.join(directory, folder, name) for name in folder_files]
            prompts = list(map(op, filter(prompt_re.match, folder_files)))
            answers = list(map(op, filter(answer_re.match, folder_files)))
            self.data[dataset_name][folder] = list(zip(prompts, answers))


class BaseLM:
    def __init__(self):
        self.data = LLMDataset()

    def answer(self, prompt: str) -> str:
        raise NotImplementedError

    def answer_folder(self, folder: list[tuple[str, str]]) -> list[dict[str, str]]:
        responses = []
        for prompt, answer in tqdm(folder):
            result = self.answer(prompt)
            responses.append({
                "prompt": prompt,
                "answer": answer,
                "response": result
            })
        return responses

    def answer_dataset(self, dataset_name: str) -> list[tuple[str, list[dict]]]:
        print(f'Answering "{dataset_name}" dataset')
        responses = []
        for i, (file_id, folder) in enumerate(self.data[dataset_name].items()):
            print(f"Answering {file_id} ({i + 1}/{len(self.data[dataset_name])})")
            responses.append((file_id, list(self.answer_folder(folder))))
        return responses

    def save_answers(self, dataset_name: str, dest_folder: str):
        for file_id, answered_folder in self.answer_dataset(dataset_name):
            with open(os.path.join(dest_folder, f"{file_id}_result.json"), "w") as f:
                json.dump(answered_folder, f)

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

        self.role = "user"

    @cache
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


class OllamaLM(BaseLM):
    def __init__(self, model_name="gemma:instruct"):
        super().__init__()
        self.model_name = model_name

    def answer(self, prompt: str) -> str:
        return ollama.generate(prompt=prompt, model=self.model_name)


# To work with huggingface models
class HugLM(BaseLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", backend="torch", **model_kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, padding_side="left", use_fast=True)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        # Autoconfig does not map devices correctly
        config = {
            "device_map": "auto",
            "use_cache": True,
            "attn_implementation": "flash_attention_2",
            **model_kwargs
        }

        # todo: try adding some Seq2SeqLM models because GPT4 said it fits better
        match backend:
            case "torch" | "pt":
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **config)
            case "tensorflow" | "tf":
                self.model = TFAutoModelForCausalLM.from_pretrained(model_name, **config)
            case "flax" | "jax":
                self.model = FlaxAutoModelForCausalLM.from_pretrained(model_name, **config)
            case _:
                raise ValueError(f"Backend {backend} not supported")

        print(f"Running {model_name} on {self.model.device}")

    @cache
    def answer(self, prompt: str) -> str:
        tokenized = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        result = self.model.generate(**tokenized)
        decoded = self.tokenizer.decode(result[0], skip_special_tokens=True)
        return decoded

    # This is an alternative to "answer_folder" that batches up the prompts towards the LLM.
    # Be careful with max memory you can allocate to the model, as this can cause problems.
    def answer_folder_many(self, folder: list[tuple[str, str]]) -> list[dict[str, str]]:
        prompts: list[str] = [prompt for prompt, _ in folder]
        return [{
            "prompt": prompt,
            "answer": answer,
            "response": response,
            } for response, (prompt, answer) in zip(self.answer(prompts), folder)]


# To use the LoRA fine-tuning method for huggingface models
class LoraLM(HugLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it"):
        super().__init__(model_name, load_in_8bit=False, load_in_4bit=True, gradient_checkpointing=True, use_cache=True)

        self.model.enable_input_require_grads()

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=8,
            lora_dropout=0.1,
            bias='none',
            task_type="CAUSAL_LM",
            use_rslora=True,  # Huggingface said "shown to work better"
        )

        self.peft_model = get_peft_model(self.model, self.lora_config)

        # TODO: write trainer for lora

        self.trainer = Trainer(
            model=self.peft_model,
            args=TrainingArguments(
                output_dir="results",
                overwrite_output_dir=True,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                save_steps=1,
                save_total_limit=2,
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.data.tokenizer, mlm=False),
        )

        self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

    def train(self, dataset_name: str, epochs: int = 1):
        self.peft_model.train()
        # self.trainer.train()




if __name__ == "__main__":
    model = LoraLM()
    model.peft_model.print_trainable_parameters()
