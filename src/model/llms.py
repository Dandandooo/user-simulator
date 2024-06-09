from transformers import AutoModelForCausalLM, TFAutoModelForCausalLM, FlaxAutoModelForCausalLM, pipeline
from transformers import AutoTokenizer
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, prepare_model_for_kbit_training
from openai import AzureOpenAI
import ollama
from datasets import Dataset, load_dataset

from tqdm import tqdm
import os
import re
import json

import torch
import numpy as np

from src.prompt_llm.gpt4_entire_eval import get_last_turn_type, calc_score


class LLMDataset:
    def __init__(self):
        self.data: dict[str, Dataset] = {}
        self.locations = {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def load(self, dataset_name):
        data = load_dataset(f"Dandandooo/user-sim/{dataset_name}")
        self.data[dataset_name] = data

    def add(self, dataset_name, directory, prompt_regex=r".*turn_\d+\.txt", answer_regex=r".*turn_\d+_answer\.txt"):
        prompt_re = re.compile(prompt_regex)
        answer_re = re.compile(answer_regex)

        op = lambda e: open(e).read()

        if dataset_name not in self.data:
            self.data[dataset_name] = {}
            self.locations[dataset_name] = None
        for folder in tqdm(os.listdir(directory)):
            folder_files = os.listdir(os.path.join(directory, folder))
            folder_files = [os.path.join(directory, folder, name) for name in folder_files]
            prompts = list(map(op, filter(prompt_re.match, folder_files)))
            answers = list(map(op, filter(answer_re.match, folder_files)))
            self.data[dataset_name][folder] = list(zip(prompts, answers))

    def where(self, dataset_name):
        return self.locations[dataset_name]


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
        where_stored = self.data.where(dataset_name)
        for i, (file_id, folder) in enumerate(self.data[dataset_name].items()):
            if where_stored is not None and (file := open(os.path.join(where_stored, f"{file_id}_result.json"), "r")).read():
                yield file_id, json.load(file)
                continue
            print(f"Answering {file_id} ({i + 1}/{len(self.data[dataset_name])})")
            yield file_id, list(self.answer_folder(folder))

    def save_answers(self, dataset_name: str, dest_folder: str):
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        self.data.locations[dataset_name] = dest_folder
        for file_id, answered_folder in self.answer_dataset(dataset_name):
            with open(os.path.join(dest_folder, f"{file_id}_result.json"), "w") as f:
                json.dump(answered_folder, f)
            print("Saved", file_id)

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
                if "OBSERVE" in answer:
                    if "OBSERVE" in response:
                        true_observed += 1
                    else:
                        false_speak += 1
                else:
                    if "OBSERVE" in response:
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
        ollama.pull(model_name)

    def answer(self, prompt: str) -> str:
        return ollama.generate(prompt=prompt, model=self.model_name)['response']


class HugPipeline(BaseLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", task="text-generation"):
        super().__init__()
        self.model_name = model_name
        self.task = task

        self.pipeline = pipeline(task, model=model_name)

    def answer(self, prompt: str) -> str:
        return self.pipeline(prompt)[0]['generated_text']


# To work with huggingface models
class HugLM(BaseLM):
    def __init__(self, model_name="google/gemma-1.1-2b-it", backend="torch", **model_kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, padding_side="left", use_fast=True)
        # self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # Autoconfig does not map devices correctly
        config = {
            "device_map": "auto",
            "use_cache": True,
            # "attn_implementation": "flash_attention_2",
            "cache_dir": ".cache",
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
            case "peft":
                self.model = AutoPeftModelForCausalLM.from_pretrained(model_name, **config)
            case _:
                raise ValueError(f"Backend {backend} not supported")

        # to account for pad token
        # self.model.resize_token_embeddings(len(self.tokenizer))

        print(f"Running {model_name} on {self.model.device}")

    def answer(self, prompt: str) -> str:
        tokenized = self.tokenizer(prompt, padding=True, return_tensors="pt").to(self.model.device)
        result = self.model.generate(**tokenized, max_new_tokens=32)
        decoded = self.tokenizer.decode(result[0], skip_special_tokens=True).removeprefix(prompt).strip()
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
    def __init__(self, model_name="google/gemma-1.1-2b-it", save_name=None, resume=False):
        save_name = model_name.split('/')[-1] if save_name is None else save_name

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=8,
            lora_dropout=0.1,
            bias='none',
            task_type="CAUSAL_LM",
            use_rslora=True,  # Huggingface said "shown to work better"
        )

        extra_config = {
            "model_name": model_name,
            "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        }

        if os.path.exists(os.path.join("llm_models", save_name)):
            if resume:
                extra_config["backend"] = "peft"
                extra_config["model_name"] = save_name
            else:
                raise FileExistsError(f"Model {save_name} already exists. You chose not to resume training.")

        super().__init__(
            **extra_config
        )

        if not resume:
            self.model = get_peft_model(self.model, self.lora_config)

        if not os.path.exists(os.path.join("llm_training_sessions", model_name.split('/')[-1])):
            os.mkdir(f"llm_training_sessions/{model_name.split('/')[-1]}")

        self.training_args = TrainingArguments(
            output_dir=f"llm_training_sessions/{model_name.split('/')[-1]}/temp",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            save_steps=1,
            save_total_limit=2,
        )

        self.model.print_trainable_parameters()

        # self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

    def train(self, train_dataset_name: str = "train", val_dataset_name: str = "valid_unseen", eval_dataset_name: str = "valid_seen", ):
        self.model.train()

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.data[train_dataset_name],
            eval_dataset=self.data[eval_dataset_name],
            tokenizer=self.tokenizer,
            # max_seq_length=
            peft_config=self.lora_config,
            args=self.training_args,
        )

        trainer.train()


if __name__ == "__main__":
    model = LoraLM()
    model.model.print_trainable_parameters()
